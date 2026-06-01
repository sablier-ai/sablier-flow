"""Evaluate a *family* of backtest strategies against the Sablier
realistic null, producing the strategy-family DSR + Probability of
Backtest Overfitting (PBO).

The single-strategy case is well served by ``@sablier_flow.augment`` —
one decorator, one backtest function, one Sharpe vs N synthetic
Sharpes. The interesting case is the *family-of-strategies* regime:
``E[max_n SR_n]`` across M strategy variants under the realistic null
vs the analytical Bailey-López-de-Prado null. This is the API for
that case.

Usage::

    import sablier_flow

    strategies = {
        f"ma_{fast}_{slow}": (lambda f, s: lambda df: my_backtest(df, fast=f, slow=s))(fast, slow)
        for fast, slow in [(5, 20), (10, 30), (20, 60), (30, 90)]
    }

    report = sablier_flow.evaluate_family(
        strategies, real_prices, n_paths=100, horizon=252,
    )

    print(report.deflated_sharpe.realistic, report.deflated_sharpe.analytical)
    print(report.pbo)              # PBO via CSCV
    report.to_html("audit.html")

What's computed:

  - **Per-strategy real Sharpe** (one number per strategy)
  - **Per-(strategy, path) synthetic Sharpe** (M × N matrix)
  - **`max_m SR_m`** on the real and on each synthetic path
  - **DSR realistic + analytical** for the family-best strategy
  - **PBO via CSCV** [Bailey et al. 2015] on the *real* history alone

PBO is computed by splitting the real return series into S contiguous
chunks, taking every C(S, S/2) train/test partition, identifying the
best-in-sample strategy on each, and reporting the fraction of
partitions where that strategy ranks below median out-of-sample.
"""

from __future__ import annotations

import itertools
import logging
import math
import sys
import warnings
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from sablier_flow.analytics.deflated_sharpe import DeflatedSharpeReport

__all__ = ["FamilyReport", "evaluate_family", "probability_of_backtest_overfitting"]

logger = logging.getLogger(__name__)

_BacktestFn = Callable[..., "float | dict[str, float]"]


# ============================================================================
# Output type
# ============================================================================


@dataclass(frozen=True)
class FamilyReport:
    """Strategy-family overfit-audit report.

    Built from a family of M backtest functions evaluated on real data
    + on the same N synthetic alternative-history paths. Reports the
    family DSR + PBO directly — not the naive percentile rank of a
    single backtest.
    """

    strategy_names: tuple[str, ...]
    """Names of the strategies in the family, in evaluation order."""

    primary_metric: str
    """Metric used to define ``best`` and to drive DSR/PBO."""

    real_metrics: tuple[dict[str, float], ...]
    """Per-strategy backtest result on the real data. Order matches
    ``strategy_names``. Each entry is whatever the strategy returned
    (scalar wrapped to ``{primary_metric: v}`` if it returned a scalar)."""

    synthetic_metrics: tuple[tuple[dict[str, float], ...], ...]
    """Per-path × per-strategy backtest result on the synthetic
    alternative histories. Shape ``(n_paths, n_strategies)`` of dicts."""

    real_max_value: float
    """``max_m SR_m`` on real data (or ``min_m`` for lower-is-better)."""

    real_argmax_strategy: str
    """Name of the family-best strategy on real data."""

    synthetic_max_values: np.ndarray
    """Per-path ``max_m SR_m`` over the family. Shape ``(n_paths,)``.
    This is the distribution of best-of-N Sharpes under the realistic
    null."""

    deflated_sharpe: DeflatedSharpeReport
    """Family DSR under the realistic + analytical nulls. The DSR test
    is on ``max_m SR_m`` rather than a single strategy's SR; ``n_trials``
    in the underlying call is set to ``len(strategies)``."""

    pbo: float
    """Probability of Backtest Overfitting via CSCV on the real history
    [Bailey-Borwein-LdP-Zhu 2015]. In [0, 1]: 0 = best in-sample strategy
    always wins out-of-sample, 1 = always loses, 0.5 = no signal.
    Computed independently of the synthetic null."""

    pbo_n_partitions: int
    """How many CSCV partitions contributed to the PBO estimate.
    For S splits, ``C(S, S/2)`` partitions."""

    pbo_cscv_splits: int
    """Number of contiguous chunks the real history was split into for
    CSCV. SDK floor is 16; lower values under-detect overfit."""

    n_paths: int
    """How many synthetic paths went into the family evaluation."""

    per_strategy_real_metric: dict[str, float] = field(default_factory=dict)
    """``{strategy_name: real_metric}`` on the primary metric. Same
    information as ``real_metrics`` but flattened to the primary metric
    for direct iteration in customer code."""

    per_strategy_overfit_score: dict[str, float] = field(default_factory=dict)
    """``{strategy_name: percentile_rank}`` — each strategy's real
    metric expressed as its empirical CDF position in its **own**
    synthetic distribution.

    **This is a directional descriptor, NOT a per-strategy overfit
    verdict.** Same caveat as :class:`RobustnessReport` carries: a
    high percentile can mean "selected from a search" OR "single
    fixed strategy happened to perform well in this realization."
    The per-strategy percentile alone cannot tell the two apart. Use
    the **PBO** field (luck-safe by construction across CSCV
    partitions) for the family-level overfit verdict — that's the
    statistic the customer should gate deploy decisions on. The
    per-strategy percentile is best read as "which variants in the
    family pulled the most weight in the real realization?" — a
    diagnostic, not a verdict.

    Interpretation (higher_is_better):
      - close to 0.5 — strategy's real metric is a typical draw from
        its own synthetic distribution
      - ≥ 0.85 — strategy's real metric is in the top 15% of its own
        synth distribution (it pulled hard in the real window — could
        be skill, luck, or overfit; PBO tells you which)
      - ≤ 0.15 — strategy's real metric is unusually low for what its
        own dynamics produce in alt-histories"""

    per_strategy_synthetic_median: dict[str, float] = field(default_factory=dict)
    """``{strategy_name: median of per-path Sharpe}``. Lets a UI render
    "real vs synth median" per strategy for the attribution table."""

    failures: tuple[str, ...] = ()
    """Stringified exceptions from synthetic backtests that raised. The
    failed paths are dropped from the synthetic distribution."""

    notes: tuple[str, ...] = ()
    """Non-fatal warnings."""

    @property
    def verdict(self) -> str:
        """Bucketed family-level verdict combining the realistic-null
        DSR and the CSCV PBO:

          - ``'significant'``       — realistic DSR ≥ 0.95 AND (PBO
            non-finite or PBO < 0.6). The family beats the realistic
            null AND the in-sample-best generalizes.
          - ``'defensible'``        — realistic DSR ∈ [0.50, 0.95) with
            no overfit-selection PBO. Better than median best-of-N but
            not unambiguous.
          - ``'overfit_selection'`` — PBO ≥ 0.6. Grid-search results
            don't generalize across CSCV train/test splits; the
            family-level overfit signal trumps the DSR magnitude.
          - ``'looks_like_noise'``  — realistic DSR < 0.50. The real
            best-of-N sits below the synthetic median.
        """
        if np.isfinite(self.pbo) and self.pbo >= 0.6:
            return "overfit_selection"
        dsr_realistic = self.deflated_sharpe.realistic
        if dsr_realistic >= 0.95:
            return "significant"
        if dsr_realistic >= 0.50:
            return "defensible"
        return "looks_like_noise"

    @property
    def acceptable(self) -> bool:
        """True when the family DSR clears the realistic-null 0.95 bar
        AND the CSCV PBO does not flag overfit selection. Both gates
        matter: a high DSR with a high PBO means the in-sample-best
        wasn't the out-of-sample-best on most splits and shouldn't
        drive a deploy decision."""
        if np.isfinite(self.pbo) and self.pbo >= 0.6:
            return False
        return self.deflated_sharpe.realistic >= 0.95

    def most_overfit_variants(self, *, top: int = 5) -> list[tuple[str, float]]:
        """Return the strategies whose real metric most outperformed
        their own synth distributions, as ``(name, percentile)`` pairs
        sorted descending.

        Useful after a "best looks like noise" family verdict to
        attribute which variants pulled the result. A high score here
        means the strategy succeeded on the realised history more than
        its own dynamics produce on alt-histories — that can be skill,
        luck, OR overfit; the PBO field (luck-safe via CSCV) is the
        only family-level statistic that distinguishes them. Read this
        as a directional attribution, not a per-strategy overfit
        verdict.
        """
        items = [
            (name, score)
            for name, score in self.per_strategy_overfit_score.items()
            if np.isfinite(score)
        ]
        items.sort(key=lambda kv: kv[1], reverse=True)
        return items[: max(top, 0)]

    def summary(self) -> str:
        """One plain-English sentence translating the family DSR + PBO for a
        non-PhD reader.

        Reads cleanly in a Slack message, PR description, or fund CI log.
        DSR + PBO numbers stay accessible through the corresponding
        fields for the quant who wants them.
        """
        metric = self.primary_metric
        best_name = self.real_argmax_strategy
        best_val = self.real_max_value
        dsr_realistic = self.deflated_sharpe.realistic
        thr_realistic = self.deflated_sharpe.threshold_sr_realistic
        n_strats = len(self.strategy_names)

        # Verdict bucket on the realistic-null DSR.
        # DSR ≥ 0.95: significant under realistic null
        # DSR ∈ [0.50, 0.95): better than median but not unambiguous
        # DSR < 0.50: real best looks worse than the synthetic best-of-N
        if dsr_realistic >= 0.95:
            verdict = (
                f"Significant: best-of-{n_strats} ({best_name}, {metric} {best_val:+.3f}) "
                f"clears the 95% bar under the realistic null. "
            )
        elif dsr_realistic >= 0.50:
            verdict = (
                f"Defensible: best-of-{n_strats} ({best_name}, {metric} {best_val:+.3f}) "
                f"beats {dsr_realistic:.0%} of synthetic best-of-{n_strats} draws — "
                f"better than chance but not unambiguous. "
            )
        else:
            verdict = (
                f"Looks like noise: best-of-{n_strats} ({best_name}, {metric} {best_val:+.3f}) "
                f"sits at the {dsr_realistic:.0%} percentile of the synthetic "
                f"best-of-{n_strats} distribution — random selection on alt-histories "
                f"routinely produces higher {metric}. "
            )

        verdict += (
            f"You'd need {metric} ≥ {thr_realistic:+.3f} to clear the realistic-null "
            f"95% bar with this family."
        )

        # PBO note — flag if the in-sample-best doesn't generalize on real data
        if np.isfinite(self.pbo):
            if self.pbo >= 0.6:
                verdict += (
                    f" PBO via CSCV = {self.pbo:.2f} (≥ 0.6) — grid-search "
                    "results don't generalise across train/test splits of the "
                    "real history; re-tune on a wider universe."
                )
            elif self.pbo <= 0.2:
                verdict += (
                    f" PBO via CSCV = {self.pbo:.2f} — the in-sample-best "
                    "strategy is also the out-of-sample-best on most splits "
                    "(good sign)."
                )
            else:
                verdict += f" PBO via CSCV = {self.pbo:.2f}."
        return verdict

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary."""
        return {
            "strategy_names": list(self.strategy_names),
            "primary_metric": self.primary_metric,
            "real_max_value": self.real_max_value,
            "real_argmax_strategy": self.real_argmax_strategy,
            "deflated_sharpe": self.deflated_sharpe.to_dict(),
            "pbo": self.pbo,
            "pbo_n_partitions": self.pbo_n_partitions,
            "pbo_cscv_splits": self.pbo_cscv_splits,
            "n_paths": self.n_paths,
            "n_strategies": len(self.strategy_names),
            "n_failures": len(self.failures),
        }


# ============================================================================
# Runtime-estimate helpers
# ============================================================================

# Per-strategy-evaluation cost heuristic (seconds). Tuned against the
# persona-simulation backtests (lightweight pandas rolling/Sharpe);
# customers with heavier engines will see longer actual wall-clock but
# the estimate's purpose is to flag "this will take a while" not to be
# precise.
_PER_STRATEGY_SECONDS = 0.001

# Threshold above which progress=True is auto-flipped on. The
# silent-hang risk gets real at n_strategies × cscv_splits > 50 because
# CSCV does C(S, S/2) partition evaluations, not just S.
_PROGRESS_AUTO_THRESHOLD = 50

# Wall-clock threshold above which we emit a UserWarning so even
# someone with stderr piped to /dev/null sees the heads-up.
_LONG_RUNTIME_WARN_SECONDS = 30.0


def _cscv_partition_count(splits: int) -> int:
    """C(splits, splits/2) — exact partition count for CSCV."""
    if splits < 2 or splits % 2 != 0:
        return 0
    return math.comb(splits, splits // 2)


def _estimate_family_runtime(
    *, n_strategies: int, cscv_splits: int, n_rows: int, n_paths: int
) -> tuple[int, int, float]:
    """A-priori wall-clock estimate for evaluate_family.

    Returns (n_partitions, n_strategy_evals, seconds). The heuristic is:

      - synthetic-side: n_paths × n_strategies backtests (each on a
        ~horizon-row DataFrame).
      - CSCV-side: per-strategy on each unique chunk-union the
        partitions touch; bounded by 2 × n_partitions × n_strategies but
        the memo in probability_of_backtest_overfitting collapses most
        partitions to repeats of the same chunk-set. Use the unbounded
        estimate as the conservative ceiling.

    A per-strategy-backtest cost of ``_PER_STRATEGY_SECONDS`` is folded
    in. The estimate is intentionally conservative — we want it to flag
    long runs before they hang, not to be tight.
    """
    n_partitions = _cscv_partition_count(cscv_splits)
    if n_partitions > 20_000:
        n_partitions = 20_000  # matches the cap in probability_of_backtest_overfitting
    synth_evals = n_paths * n_strategies
    cscv_evals = 2 * n_partitions * n_strategies
    total = synth_evals + cscv_evals
    seconds = total * _PER_STRATEGY_SECONDS
    return n_partitions, total, seconds


def _emit_runtime_estimate_line(
    where: str,
    *,
    n_partitions: int,
    n_strategy_evals: int,
    seconds: float,
) -> None:
    """Stderr one-liner used by both evaluate_family and PBO.

    Reports the partition count + strategy-evaluation count only.
    Wall-clock is intentionally omitted because the per-strategy cost
    depends entirely on what's inside the customer's backtest function
    (pandas rolling on 252 rows is microseconds; a vectorbt sweep on
    intraday data is seconds). A printed prediction would be misleading
    in either direction. The counts let the caller form their own
    expectation against their per-backtest timing."""
    print(
        f"{where}: ~{n_partitions} partitions, ~{n_strategy_evals} "
        "strategy evaluations (use progress=True for live updates).",
        file=sys.stderr,
    )
    if seconds > _LONG_RUNTIME_WARN_SECONDS:
        warnings.warn(
            f"{where} workload is large ({n_strategy_evals} strategy evaluations); "
            "consider progress=True and/or executor='thread'.",
            UserWarning,
            stacklevel=3,
        )


_FLOW_KWARGS_ALIAS = "alternative_versions_kwargs"


def _resolve_flow_kwargs(raw_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Accept both ``**flow_kwargs`` and the legacy
    ``alternative_versions_kwargs={...}`` dict form.

    If the alias is present its dict is merged INTO the top-level
    kwargs; same-name keys in the top-level kwargs win (caller-explicit
    over inherited). No DeprecationWarning — we're unifying, not
    deprecating, and there are no customers on the old name.
    """
    alias = raw_kwargs.pop(_FLOW_KWARGS_ALIAS, None)
    if alias is None:
        return raw_kwargs
    if not isinstance(alias, Mapping):
        raise TypeError(
            f"{_FLOW_KWARGS_ALIAS} must be a mapping, got {type(alias).__name__}"
        )
    merged: dict[str, Any] = dict(alias)
    merged.update(raw_kwargs)
    return merged


# ============================================================================
# Public entry point
# ============================================================================


def evaluate_family(
    strategies: Mapping[str, _BacktestFn],
    real_data: pd.DataFrame,
    *,
    model_id: str | None = None,
    n_paths: int = 100,
    primary_metric: str | None = None,
    higher_is_better: bool = True,
    pbo_cscv_splits: int = 16,
    executor: Literal["serial", "thread"] = "serial",
    max_workers: int | None = None,
    progress: bool | None = None,
    raise_on_failure: bool = False,
    **flow_kwargs: Any,
) -> FamilyReport:
    """Evaluate a family of backtest strategies against the Sablier
    realistic null and report the family DSR + PBO.

    Generates the synthetic alternative-history paths **once** and runs
    every strategy on every path. The synthetic data round-trip dominates
    runtime, so amortising it across M strategies is materially faster
    than calling ``@augment`` M times.

    Cost model
    ----------
    Without ``model_id`` this call trains a fresh flow model and
    generates ``n_paths`` paths on top — same cost as a standalone
    ``sf.fit`` + ``sf.generate``, where the fit step dominates by
    roughly two orders of magnitude. Iterative strategy research should
    fit once via ``sf.fit`` and then pass ``model_id=fit.model_id`` here
    to skip the fit on every call, paying only for generation
    (a handful of credits per pass).

    Parameters
    ----------
    strategies
        Mapping of ``name → backtest_fn``. Each ``backtest_fn`` is a
        callable ``f(prices: pd.DataFrame) -> float | dict[str, float]``
        — same shape as ``@augment`` expects.
    real_data
        The real-history DataFrame, same format as for
        :func:`alternative_versions`.
    model_id
        Reuse a previously fitted model instead of training a fresh
        one. Pass the ``model_id`` returned from a prior ``sf.fit(...)``
        call. When supplied, the fit step is skipped — the function
        goes straight to ``sf.generate(model_id, ...)`` — which saves
        the dominant cost. Default ``None`` runs a fresh fit on
        ``real_data``.
    n_paths
        Number of synthetic alternative histories.
    primary_metric
        If strategies return dicts, which metric drives DSR + PBO.
        Defaults to ``"sharpe"`` if present, else the first key.
    higher_is_better
        Whether higher values on ``primary_metric`` are better.
    pbo_cscv_splits
        Number of contiguous chunks to split the real history into for
        the CSCV procedure. SDK floor is S = 16. Larger S → more
        partitions → tighter PBO estimate but more compute (each
        partition runs every strategy on every chunk).
    executor
        ``"serial"`` (default) or ``"thread"`` for the M × N synthetic
        backtests. Threads help when individual backtests are I/O-bound.
    max_workers
        Worker count for the thread executor.
    progress
        Stream a progress counter to stderr (every ~5% of synthetic
        paths). ``None`` (default) auto-enables it for heavy runs
        (``n_strategies × cscv_splits > 50``); pass ``True`` / ``False``
        to force.
    raise_on_failure
        If True, a single synthetic-backtest exception aborts the call.
        Default False: failures go to ``report.failures`` and the
        DSR is computed on whatever succeeded.
    **flow_kwargs
        Passed through to :func:`fit` / :func:`generate` (``features``,
        ``horizon``, ``seed``, ``like``, ``anchor_data``, ``api_key``,
        ``endpoint``, ``verify``, etc.). When ``model_id`` is set, the
        fit-only kwargs (``features``, ``train_split``, ``embargo_days``)
        are ignored — those were locked in at the original fit. For
        symmetry with older call sites, ``alternative_versions_kwargs={...}``
        is accepted as an alias dict and merged in (same-named explicit
        kwargs win).
    """
    if not strategies:
        raise ValueError("strategies must be non-empty")
    if pbo_cscv_splits < 4 or pbo_cscv_splits % 2 != 0:
        raise ValueError(
            f"pbo_cscv_splits must be an even integer ≥ 4, got {pbo_cscv_splits}"
        )
    if pbo_cscv_splits < 16:
        # PBO via CSCV under-detects overfit on noise grids when S<16 —
        # the partition count C(S, S/2) is too small to discriminate the
        # in-sample-best from luck.
        warnings.warn(
            f"pbo_cscv_splits={pbo_cscv_splits}<16 gives unstable estimates; "
            "use >=16 (SDK floor; smaller S under-detects overfit on "
            "noise grids).",
            DeprecationWarning,
            stacklevel=2,
        )

    # Accept the legacy alias ``alternative_versions_kwargs={...}`` alongside
    # the **flow_kwargs sink so older call sites don't break.
    flow_kwargs = _resolve_flow_kwargs(flow_kwargs)

    strategy_names: tuple[str, ...] = tuple(strategies)
    strategy_fns: tuple[_BacktestFn, ...] = tuple(strategies.values())
    m_strategies = len(strategy_fns)

    # --- 0. A-priori runtime estimate + progress auto-flip --------------
    # Print a single stderr line so users see "this will take a while"
    # before the call hangs silently. Auto-flip progress=True when the
    # workload crosses the threshold where the silent-hang risk gets real.
    n_partitions_est, n_strategy_evals_est, est_seconds = _estimate_family_runtime(
        n_strategies=m_strategies,
        cscv_splits=pbo_cscv_splits,
        n_rows=len(real_data),
        n_paths=n_paths,
    )
    _emit_runtime_estimate_line(
        "evaluate_family",
        n_partitions=n_partitions_est,
        n_strategy_evals=n_strategy_evals_est,
        seconds=est_seconds,
    )
    if progress is None:
        progress = (m_strategies * pbo_cscv_splits) > _PROGRESS_AUTO_THRESHOLD

    # --- 1. Real-data backtests ----------------------------------------
    real_metrics = tuple(
        _normalise_result(fn(real_data), primary_metric) for fn in strategy_fns
    )

    # Decide the primary metric now (auto-pick from first real result)
    if primary_metric is None:
        keys = list(real_metrics[0])
        primary_metric = "sharpe" if "sharpe" in keys else keys[0]
    for i, m in enumerate(real_metrics):
        if primary_metric not in m:
            raise ValueError(
                f"strategy {strategy_names[i]!r} did not return "
                f"primary_metric={primary_metric!r}"
            )

    # --- 2. Generate synthetic alternative histories -------------------
    # Two paths:
    #   (a) model_id supplied  → reuse a prior fit, pay only generate
    #   (b) model_id is None   → train + generate (the train step dominates the cost)
    # The fit is the dominant cost by a huge margin; iterative research
    # workflows should fit once externally and pass model_id here every
    # subsequent call.
    from sablier_flow.client.client import fit as _fit
    from sablier_flow.client.client import generate as _generate

    # Split family-level kwargs into the fit + generate sides.
    fit_kwargs: dict[str, Any] = {}
    gen_kwargs: dict[str, Any] = {}
    for key, value in flow_kwargs.items():
        if key in {"features", "train_split", "embargo_days"}:
            fit_kwargs[key] = value
        elif key in {"anchor_data", "like"}:
            gen_kwargs[key] = value
        elif key in {"horizon", "seed"}:
            # horizon + seed are useful on both sides; pass through.
            fit_kwargs[key] = value
            gen_kwargs[key] = value
        else:
            gen_kwargs[key] = value

    if model_id is not None:
        # Reuse-path: skip the fit entirely. fit-only kwargs (features,
        # train_split, embargo_days) were locked in at original fit time
        # — silently drop them rather than fail loudly so a caller can
        # pass the same kwargs dict to both the fit path and the reuse
        # path without conditional logic.
        resolved_model_id = model_id
    else:
        fit_res = _fit(real_data, **fit_kwargs)
        resolved_model_id = fit_res.model_id
    gen = _generate(resolved_model_id, n_paths=n_paths, **gen_kwargs)

    # 1.0.20 — horizon-mismatch warning. The customer's backtest was
    # just run on the FULL ``real_data`` above (line ~ "real_metrics
    # = …"), producing a Sharpe over ``len(real_data)`` bars. The
    # synthetic side will be ``n_paths × gen.horizon`` bars. When those
    # two lengths differ, the real and synthetic Sharpes are computed
    # on different sample sizes — fine for PBO (CSCV walks the real
    # series alone), but a methodological footgun for the DSR-vs-synth
    # comparison: the synthetic distribution is a poor null for a real
    # Sharpe computed on 14× as many bars. Emit a UserWarning so the
    # customer can either window ``real_data`` to ``gen.horizon`` or
    # pass ``like=real_data.iloc[-gen.horizon:]`` to ``generate``.
    if len(real_data) != gen.horizon:
        warnings.warn(
            f"evaluate_family: real_data has {len(real_data)} bars but the "
            f"synthetic horizon is {gen.horizon}. The real Sharpe is computed "
            f"over {len(real_data)} bars while each synthetic Sharpe is "
            f"computed over {gen.horizon} bars — different sample sizes mean "
            f"the synthetic distribution is a biased null for the DSR-vs-real "
            f"comparison. PBO is unaffected. To match horizons, either pass "
            f"`like=real_data.iloc[-{gen.horizon}:]` (truncates synth to your "
            f"window) or window `real_data` yourself to {gen.horizon} bars "
            f"before calling.",
            UserWarning,
            stacklevel=2,
        )

    # --- 3. Materialise synth DataFrames once, run every strategy on each
    synth_dfs = gen.as_dataframes(index=_aligned_index(real_data, gen.horizon))

    synthetic_metrics, failures = _run_family_batch(
        strategy_fns=strategy_fns,
        strategy_names=strategy_names,
        datasets=synth_dfs,
        primary_metric=primary_metric,
        executor=executor,
        max_workers=max_workers,
        progress=progress,
        raise_on_failure=raise_on_failure,
    )

    # --- 4. Per-path max_m SR_m ----------------------------------------
    # For higher-is-better, max; for lower-is-better, min — but we use
    # the same "extreme" reductor and let the DSR module's interpretation
    # be conditioned on higher_is_better via the caller's choice.
    primary_real = np.array(
        [m[primary_metric] for m in real_metrics], dtype=np.float64
    )
    if higher_is_better:
        real_extremum = float(np.max(primary_real))
        argmax_idx = int(np.argmax(primary_real))
    else:
        real_extremum = float(np.min(primary_real))
        argmax_idx = int(np.argmin(primary_real))

    synth_max_per_path: list[float] = []
    for path_results in synthetic_metrics:
        if not path_results:
            continue
        vals = np.array(
            [m.get(primary_metric, np.nan) for m in path_results], dtype=np.float64
        )
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        synth_max_per_path.append(
            float(np.max(vals) if higher_is_better else np.min(vals))
        )
    synthetic_max_values = np.array(synth_max_per_path, dtype=np.float64)

    if synthetic_max_values.size == 0:
        raise RuntimeError(
            "Every synthetic path failed for at least one strategy — "
            "cannot compute family DSR. See report.failures."
        )

    # --- 5. Family DSR (realistic + analytical) ------------------------
    from sablier_flow.analytics.deflated_sharpe import deflated_sharpe as _dsr

    dsr = _dsr(
        observed_sr=real_extremum,
        synthetic_sharpes=synthetic_max_values,
        n_trials=m_strategies,
    )

    # --- 5b. Per-strategy attribution -----------------------------------
    # For each strategy m, compute the empirical CDF position of its real
    # metric in its OWN synthetic distribution. Lets the customer narrow
    # "the family is overfit" → "this specific variant is pulling it."
    per_strategy_real_metric: dict[str, float] = {}
    per_strategy_overfit_score: dict[str, float] = {}
    per_strategy_synthetic_median: dict[str, float] = {}
    for s_idx, name in enumerate(strategy_names):
        per_strategy_real_metric[name] = float(primary_real[s_idx])
        path_vals = np.array(
            [
                synthetic_metrics[p_idx][s_idx].get(primary_metric, np.nan)
                for p_idx in range(len(synthetic_metrics))
            ],
            dtype=np.float64,
        )
        path_vals = path_vals[np.isfinite(path_vals)]
        if path_vals.size == 0:
            per_strategy_overfit_score[name] = float("nan")
            per_strategy_synthetic_median[name] = float("nan")
            continue
        if higher_is_better:
            per_strategy_overfit_score[name] = float(
                np.mean(path_vals < primary_real[s_idx])
            )
        else:
            per_strategy_overfit_score[name] = float(
                np.mean(path_vals > primary_real[s_idx])
            )
        per_strategy_synthetic_median[name] = float(np.median(path_vals))

    # --- 6. PBO via CSCV on the real history ---------------------------
    pbo_value, n_partitions = probability_of_backtest_overfitting(
        strategies, real_data,
        primary_metric=primary_metric,
        higher_is_better=higher_is_better,
        cscv_splits=pbo_cscv_splits,
        executor=executor,
        max_workers=max_workers,
    )

    notes: list[str] = []
    if n_paths < 100:
        notes.append(
            f"only {n_paths} synthetic paths — DSR is high-variance. "
            "Aim for ≥ 500 paths for stable estimates."
        )
    if failures:
        notes.append(f"{len(failures)} synthetic backtests failed (see failures field).")

    return FamilyReport(
        strategy_names=strategy_names,
        primary_metric=primary_metric,
        real_metrics=real_metrics,
        synthetic_metrics=tuple(tuple(p) for p in synthetic_metrics),
        real_max_value=real_extremum,
        real_argmax_strategy=strategy_names[argmax_idx],
        synthetic_max_values=synthetic_max_values,
        deflated_sharpe=dsr,
        pbo=pbo_value,
        pbo_n_partitions=n_partitions,
        pbo_cscv_splits=pbo_cscv_splits,
        n_paths=int(synthetic_max_values.size),
        per_strategy_real_metric=per_strategy_real_metric,
        per_strategy_overfit_score=per_strategy_overfit_score,
        per_strategy_synthetic_median=per_strategy_synthetic_median,
        failures=tuple(failures),
        notes=tuple(notes),
    )


# ============================================================================
# PBO via CSCV
# ============================================================================


def probability_of_backtest_overfitting(
    strategies: Mapping[str, _BacktestFn],
    real_data: pd.DataFrame,
    *,
    primary_metric: str = "sharpe",
    higher_is_better: bool = True,
    cscv_splits: int = 16,
    executor: Literal["serial", "thread"] = "serial",
    max_workers: int | None = None,
) -> tuple[float, int]:
    """Compute the Bailey-Borwein-López-de-Prado-Zhu PBO via CSCV.

    Procedure (Bailey et al. 2015):
      1. Split the real history into ``S`` contiguous chunks.
      2. For each ``C(S, S/2)`` partition of chunks into train/test:
         (a) evaluate every strategy on the train partition;
         (b) evaluate every strategy on the test partition;
         (c) identify the in-sample best strategy;
         (d) compute its rank percentile out-of-sample.
      3. PBO = fraction of partitions where the in-sample best is
         out-of-sample below median (rank < 0.5).

    Returns
    -------
    (pbo, n_partitions)
    """
    if cscv_splits < 4 or cscv_splits % 2 != 0:
        raise ValueError(f"cscv_splits must be even ≥ 4, got {cscv_splits}")
    if cscv_splits < 16:
        # PBO via CSCV under-detects overfit at S<16 — partition count
        # C(S, S/2) is too small to discriminate in-sample-best from luck
        # on a noise grid.
        warnings.warn(
            f"cscv_splits={cscv_splits}<16 gives unstable estimates; "
            "use >=16 (SDK floor; smaller S under-detects overfit on "
            "noise grids).",
            DeprecationWarning,
            stacklevel=2,
        )

    # A-priori runtime estimate — one stderr line + UserWarning if heavy,
    # same shape as evaluate_family so customers aren't surprised by a
    # silent hang on a big grid.
    n_strategies = len(strategies)
    n_partitions_est, n_strategy_evals_est, est_seconds = _estimate_family_runtime(
        n_strategies=n_strategies,
        cscv_splits=cscv_splits,
        n_rows=len(real_data),
        n_paths=0,  # PBO doesn't touch synthetic paths
    )
    _emit_runtime_estimate_line(
        "probability_of_backtest_overfitting",
        n_partitions=n_partitions_est,
        n_strategy_evals=n_strategy_evals_est,
        seconds=est_seconds,
    )

    n_rows = len(real_data)
    if n_rows < cscv_splits * 4:
        # Not enough data for meaningful CSCV — return NaN with a small partition count.
        return float("nan"), 0

    # Split into S equal-sized contiguous chunks (drop any short tail).
    chunk_size = n_rows // cscv_splits
    chunks: list[Any] = [
        real_data.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(cscv_splits)
    ]
    half = cscv_splits // 2
    chunk_indices = list(range(cscv_splits))
    partitions = list(itertools.combinations(chunk_indices, half))

    list(strategies)
    strategy_fns = list(strategies.values())

    # Cap partitions for very large S to keep runtime bounded — S = 16
    # gives C(16, 8) = 12,870 partitions which is fine, but S = 20 gives
    # 184,756 and explodes if each backtest is slow.
    max_partitions = 20_000
    if len(partitions) > max_partitions:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(partitions), size=max_partitions, replace=False)
        partitions = [partitions[i] for i in idx]

    # Cache per-chunk × per-strategy results so we don't re-run backtests
    # — the dominating cost. C(S, S/2) partitions all draw from the same
    # S chunks; pre-computing once is O(S × M) instead of O(C(S, S/2) × M).
    import pandas as pd

    np.full((cscv_splits, len(strategy_fns)), np.nan)

    def _run_chunk(i: int) -> tuple[int, list[float]]:
        # Each chunk is too short to backtest meaningfully on its own —
        # the strategy needs context across multiple chunks. Build it
        # lazily inside the iteration below instead.
        return i, [np.nan] * len(strategy_fns)

    # Run each strategy on each train and test partition. The partitions
    # are unions of chunks, so we have to evaluate strategies on the
    # *unioned* DataFrames, not per-chunk. Use a memo on the frozenset
    # of chunk indices.
    memo: dict[frozenset[int], list[float]] = {}

    def _eval_partition(chunk_set: frozenset[int]) -> list[float]:
        if chunk_set in memo:
            return memo[chunk_set]
        df = pd.concat([chunks[i] for i in sorted(chunk_set)], axis=0)
        results: list[float] = []
        for fn in strategy_fns:
            try:
                out = fn(df)
                if isinstance(out, dict):
                    results.append(float(out.get(primary_metric, np.nan)))
                else:
                    results.append(float(out))
            except Exception as exc:
                logger.warning("pbo strategy raised on chunk-set %s: %s", chunk_set, exc)
                results.append(np.nan)
        memo[chunk_set] = results
        return results

    n_below_median = 0
    n_counted = 0
    for train_indices in partitions:
        train_set = frozenset(train_indices)
        test_set = frozenset(chunk_indices) - train_set

        train_metrics = np.array(_eval_partition(train_set), dtype=np.float64)
        test_metrics = np.array(_eval_partition(test_set), dtype=np.float64)

        if not (np.all(np.isfinite(train_metrics)) and np.all(np.isfinite(test_metrics))):
            continue

        if higher_is_better:
            best_in_sample = int(np.argmax(train_metrics))
        else:
            best_in_sample = int(np.argmin(train_metrics))

        # Rank percentile of best-in-sample strategy out-of-sample
        # (lower-is-better metrics: rank by reversed order).
        oos_score = test_metrics[best_in_sample]
        if higher_is_better:
            rank = float(np.mean(test_metrics <= oos_score))
        else:
            rank = float(np.mean(test_metrics >= oos_score))
        if rank < 0.5:
            n_below_median += 1
        n_counted += 1

    pbo = (n_below_median / n_counted) if n_counted else float("nan")
    return float(pbo), n_counted


# ============================================================================
# Internals
# ============================================================================


def _aligned_index(real_data: Any, horizon: int) -> Any:
    """Pick a sensible DatetimeIndex of length ``horizon`` for the
    synthetic DataFrames.

    Three cases:
      - real index length matches horizon → use it as-is (most common
        when ``generate(model_id, like=window)`` was called)
      - real has a DatetimeIndex but length differs → bdate_range starting
        one business day after the real history ends (synthetic plays
        the role of "the future the real data didn't show us")
      - otherwise → return ``None`` and let the adapter use its default
    """
    idx = getattr(real_data, "index", None)
    if idx is None:
        return None
    if len(idx) == horizon:
        return idx
    import pandas as _pd
    if isinstance(idx, _pd.DatetimeIndex) and len(idx) > 0:
        start = idx[-1] + _pd.offsets.BDay(1)
        return _pd.bdate_range(start=start, periods=horizon)
    return None


def _normalise_result(
    result: float | dict[str, float],
    primary_metric: str | None,
) -> dict[str, float]:
    """Wrap a scalar result into a single-metric dict so the rest of the
    pipeline can treat scalars and dicts uniformly."""
    if isinstance(result, dict):
        return {k: float(v) for k, v in result.items()}
    key = primary_metric or "value"
    return {key: float(result)}


def _run_family_batch(
    *,
    strategy_fns: tuple[_BacktestFn, ...],
    strategy_names: tuple[str, ...],
    datasets: list[Any],
    primary_metric: str,
    executor: Literal["serial", "thread"],
    max_workers: int | None,
    progress: bool,
    raise_on_failure: bool,
) -> tuple[list[list[dict[str, float]]], list[str]]:
    """Run every strategy on every dataset. Returns
    (per_path[strategy_results], failures)."""
    n = len(datasets)
    len(strategy_fns)
    results: list[list[dict[str, float]]] = [[] for _ in range(n)]
    failures: list[str] = []

    # Stream progress in ~5% increments so users see motion on heavy runs
    # rather than a silent hang. Falls back to "every path" for tiny N
    # where 5% rounds down to 0.
    progress_step = max(1, n // 20)

    def _emit_progress(done: int) -> None:
        if not progress:
            return
        if done == n or done % progress_step == 0:
            print(
                f"  [evaluate_family] {done}/{n} synthetic paths complete",
                file=sys.stderr,
            )

    def _run_one(path_idx: int) -> tuple[int, list[dict[str, float]], list[str]]:
        path_results: list[dict[str, float]] = []
        path_failures: list[str] = []
        df = datasets[path_idx]
        for s_idx, fn in enumerate(strategy_fns):
            try:
                out = fn(df)
                path_results.append(_normalise_result(out, primary_metric))
            except Exception as exc:
                if raise_on_failure:
                    raise
                path_failures.append(
                    f"path {path_idx}, strategy {strategy_names[s_idx]}: {exc!r}"
                )
                path_results.append({primary_metric: float("nan")})
        return path_idx, path_results, path_failures

    if executor == "serial":
        for i in range(n):
            _, path_results, path_failures = _run_one(i)
            results[i] = path_results
            failures.extend(path_failures)
            _emit_progress(i + 1)
        return results, failures

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, i): i for i in range(n)}
        for done_count, fut in enumerate(as_completed(futures), start=1):
            i, path_results, path_failures = fut.result()
            results[i] = path_results
            failures.extend(path_failures)
            _emit_progress(done_count)
    return results, failures
