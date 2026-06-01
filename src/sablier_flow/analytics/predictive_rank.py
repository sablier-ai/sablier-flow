"""Predictive-rank calibration score for the forward-generation use case.

A synthetic-data generator that passes distributional-fidelity
checks can still mis-rank the strategies a customer cares about.
This module ships the analytic that lets a customer measure that
directly: given per-strategy metric vectors computed on (a) real
OOS data and (b) synth-forward paths, it reports the Spearman rank
correlation ρ between the two vectors + a bootstrap 95% CI +
a verdict band.

Two-axis quality definition for synthetic financial paths:
distributional fidelity AND predictive-rank validity. ρ near +1
means strategy ranking on synth-forward paths is a meaningful
proxy for ranking on the eventual realized window; ρ near 0 means
the ranking is random; ρ < 0 means the generator actively misranks
(the most dangerous failure mode — a customer who trusts it would
deploy the strategies that lose money on the eventual real window).

Pure analytic — does NOT run strategies and does NOT generate
paths. The user owns execution (same boundary as ``sf.robustness``);
we take the metric vectors and return the rank correlation +
bootstrap CI + verdict band that tells them how much to trust
forward-forecast rankings on their universe.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = ["PredictiveRankReport", "predictive_rank_score"]


_PredictiveRankVerdict = Literal[
    "well_calibrated",
    "weakly_calibrated",
    "uncalibrated",
    "inverted",
]


@dataclass(frozen=True)
class PredictiveRankReport:
    """Per-model calibration of strategy ranking on synthetic forward
    paths vs realized OOS data. Built by :func:`predictive_rank_score`.

    The headline statistic is the Spearman rank correlation ρ between
    two per-strategy metric vectors:

      - ``real_results``  — what each strategy did on real OOS data
      - ``synth_results`` — what each strategy did on forward-
        generated synthetic paths (averaged across paths)

    A ρ near +1 means the customer's model preserves the strategy
    ranking — picking the best strategy on synth is approximately
    the same as picking the best strategy on the eventual realized
    deployment. ρ near 0 means the ranking is random. ρ < 0 means
    the model actively misranks (the most dangerous failure mode —
    a customer who trusts a ρ < 0 generator would deploy the
    strategies that lose money on the realized window).
    """

    spearman_rho: float
    """Spearman rank correlation between ``real_results`` and
    ``synth_results``, computed on the intersection of strategy names."""

    p_value: float
    """Two-sided ``scipy.stats.spearmanr`` p-value under the null of
    zero rank correlation. Read alongside the bootstrap CI — for
    small N (< 20 strategies) the CI is the more honest signal."""

    ci_95: tuple[float, float]
    """Bootstrap percentile 95% CI on ρ, resampling the strategy
    variants with replacement ``n_bootstrap`` times (default 10000)."""

    n_strategies: int
    """Number of strategies used in the rank correlation (intersection
    of keys across ``real_results`` and ``synth_results``)."""

    mean_abs_metric_gap: float
    """Mean absolute difference between real and synth on the primary
    metric across strategies. The rank correlation may be high (the
    model picks the right strategy) while the magnitude is biased (the
    Sharpe number is off by ~0.3). This field surfaces that magnitude
    bias so the customer doesn't read the rank as a point estimate."""

    primary_metric: str
    """The metric used. If inputs were dicts, picked via
    ``primary_metric=`` kwarg (default 'sharpe' if present, else first
    key). If inputs were scalars, defaults to 'value'."""

    real_values: dict[str, float]
    """The per-strategy real-OOS metric values fed in (after
    intersecting + extracting the primary metric). Retained for
    UI rendering."""

    synth_values: dict[str, float]
    """Same shape as ``real_values`` but on synth-forward."""

    n_bootstrap: int
    """How many bootstrap resamples produced ``ci_95``."""

    notes: list[str]
    """Non-fatal warnings: low n_strategies, near-zero variance in one
    vector, missing-key intersections, etc. Read these before treating
    the verdict as load-bearing."""

    @property
    def verdict(self) -> _PredictiveRankVerdict:
        """Bucketed verdict on ``spearman_rho`` + CI lower bound:

          - ``well_calibrated``    — ρ ≥ 0.60 AND CI lower bound > 0
          - ``weakly_calibrated``  — ρ ∈ [0.30, 0.60) OR CI brackets zero
          - ``uncalibrated``       — ρ ∈ [-0.30, 0.30) — the rank is essentially random
          - ``inverted``           — ρ < -0.30 (model actively misranks; do NOT deploy
                                     on its synth-forward ranking)

        The CI gate on ``well_calibrated`` is important: ρ = 0.75 on a
        15-strategy family with CI [-0.10, +0.95] does not warrant the
        same trust as ρ = 0.75 with CI [+0.55, +0.92].
        """
        rho = self.spearman_rho
        lo, _hi = self.ci_95
        if rho < -0.30:
            return "inverted"
        if rho < 0.30:
            return "uncalibrated"
        if rho < 0.60:
            return "weakly_calibrated"
        # rho >= 0.60 — check the CI doesn't crater
        if lo <= 0.0:
            return "weakly_calibrated"
        return "well_calibrated"

    @property
    def acceptable(self) -> bool:
        """True if the verdict is ``well_calibrated`` or
        ``weakly_calibrated``. UIs can gate "trust the forward forecast"
        deploy paths on this — uncalibrated / inverted models should not
        drive deployment decisions until the strategy family or model
        universe is fixed."""
        return self.verdict in ("well_calibrated", "weakly_calibrated")

    def summary(self) -> str:
        """One plain-English sentence translating the verdict + the CI
        + the magnitude bias for a non-PhD reader. Reads cleanly in a
        Slack message, a PR description, or a fund CI log."""
        rho = self.spearman_rho
        lo, hi = self.ci_95
        n = self.n_strategies
        gap = self.mean_abs_metric_gap
        metric = self.primary_metric

        if self.verdict == "well_calibrated":
            lead = (
                f"Well calibrated: Spearman ρ = {rho:+.2f} (95% CI "
                f"[{lo:+.2f}, {hi:+.2f}]) across {n} strategies. "
                f"Your strategy ranking on synth-forward paths is a "
                f"meaningful proxy for ranking on the eventual realized "
                f"deployment window."
            )
        elif self.verdict == "weakly_calibrated":
            lead = (
                f"Weakly calibrated: Spearman ρ = {rho:+.2f} (95% CI "
                f"[{lo:+.2f}, {hi:+.2f}]) across {n} strategies. "
                f"The synth ranking signal is present but not unambiguous "
                f"— treat as a tiebreaker, not a deploy gate."
            )
        elif self.verdict == "uncalibrated":
            lead = (
                f"Uncalibrated: Spearman ρ = {rho:+.2f} (95% CI "
                f"[{lo:+.2f}, {hi:+.2f}]) across {n} strategies. "
                f"The synth-forward ranking has no detectable relationship "
                f"to the real ranking on this universe. Do NOT use synth "
                f"Sharpe to pick strategies for deployment."
            )
        else:  # inverted
            lead = (
                f"INVERTED: Spearman ρ = {rho:+.2f} (95% CI "
                f"[{lo:+.2f}, {hi:+.2f}]) across {n} strategies. "
                f"Your model picks the WORST real-market strategies first "
                f"— do NOT deploy on this model's synth-forward "
                f"ranking; investigate (regime shift, broken features, "
                f"overfit/memorization on training data)."
            )

        magnitude = (
            f" Magnitude bias (mean |{metric}_real - {metric}_synth|) = "
            f"{gap:.2f} — the rank can be right while the absolute number "
            f"is biased, so do not read synth medians as point forecasts."
        )

        if self.notes:
            magnitude += f" Notes: {'; '.join(self.notes)}."
        return lead + magnitude

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable summary for telemetry / dashboards."""
        return {
            "spearman_rho": self.spearman_rho,
            "p_value": self.p_value,
            "ci_95": list(self.ci_95),
            "verdict": self.verdict,
            "n_strategies": self.n_strategies,
            "mean_abs_metric_gap": self.mean_abs_metric_gap,
            "primary_metric": self.primary_metric,
            "n_bootstrap": self.n_bootstrap,
            "notes": list(self.notes),
        }


def _extract_primary(
    results: Mapping[str, float | Mapping[str, float]],
    primary_metric: str | None,
) -> tuple[dict[str, float], str]:
    """Coerce ``{name: scalar}`` or ``{name: {metric: scalar}}`` to a
    flat ``{name: scalar}`` on a single chosen metric. Returns
    ``(flat_dict, resolved_metric, form)`` where ``form`` is the literal
    string ``'scalar'`` or ``'dict'`` describing what shape the input
    used. The caller compares the form across real / synth sides to
    catch the silent-mislabel bug (real=dict[sharpe], synth=scalar →
    scalar gets labeled 'sharpe' with no way to verify)."""
    out: dict[str, float] = {}
    resolved = primary_metric
    saw_scalar = False
    saw_dict = False
    for name, value in results.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            saw_scalar = True
            out[str(name)] = float(value)
            if resolved is None:
                resolved = "value"
        elif isinstance(value, Mapping):
            saw_dict = True
            v_dict = {str(k): float(v) for k, v in value.items()}
            if resolved is None:
                resolved = "sharpe" if "sharpe" in v_dict else next(iter(v_dict))
            if resolved not in v_dict:
                raise ValueError(
                    f"strategy {name!r} has no metric {resolved!r}; "
                    f"available keys: {list(v_dict)}"
                )
            out[str(name)] = v_dict[resolved]
        else:
            raise TypeError(
                f"strategy {name!r} value is {type(value).__name__}; "
                "expected a scalar or a dict of metrics"
            )
    if resolved is None:
        raise ValueError("results is empty — need at least 1 strategy")
    if saw_scalar and saw_dict:
        raise ValueError(
            "results mixes scalar and dict values across strategies — "
            "every entry must be the same shape. Pass either all scalars "
            "or all dicts with the same metric key."
        )
    form: Literal["scalar", "dict"] = "dict" if saw_dict else "scalar"
    return out, resolved, form


def predictive_rank_score(
    real_results: Mapping[str, float | Mapping[str, float]],
    synth_results: Mapping[str, float | Mapping[str, float]],
    *,
    primary_metric: str | None = None,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> PredictiveRankReport:
    """Calibrate strategy ranking on synthetic forward paths against
    realized OOS data.

    This is the model-specific predictive-validity check that tells a
    customer how much to trust forward-forecast Sharpe rankings on
    THEIR universe. The customer runs their own strategy family on
    (a) real OOS data and (b) synth-forward paths from
    ``sf.generate(model_id, horizon=N, anchor_data=real.iloc[-200:])``
    and passes the per-strategy metric vectors in.

    Pure analytic — does NOT run strategies and does NOT generate
    paths. The user owns execution; we take the metric vectors and
    return the rank correlation + bootstrap CI + verdict band.

    Parameters
    ----------
    real_results
        ``{strategy_name: metric}`` evaluated on real OOS data. Each
        value is either a scalar (the primary metric) or a dict of
        ``{metric_name: value}``. The strategy names must match
        ``synth_results`` — only the intersection of keys is used.
    synth_results
        Same shape as ``real_results`` but on synthetic forward paths.
        For each strategy, the typical pattern is::

            synth_results[name] = np.mean(
                [my_backtest(df)['sharpe'] for df in forward_paths.as_dataframes()]
            )

        i.e., average the per-path Sharpe across the synth-forward
        paths so each strategy gets a single number.
    primary_metric
        Which metric to compute the rank correlation on. If the values
        are scalars, this defaults to ``'value'``. If they are dicts,
        this defaults to ``'sharpe'`` (if present) or the first key.
    n_bootstrap
        Bootstrap resamples for the 95% CI on ρ. Default 10000;
        lower values for cheap sanity checks.
    seed
        Bootstrap seed. Different seeds shift the CI bounds by O(1/√n_bootstrap).

    Returns
    -------
    PredictiveRankReport
        Frozen dataclass — see the class docstring for the fields. The
        headline is ``report.verdict`` and ``report.summary()``.

    Raises
    ------
    ValueError
        If the inputs have fewer than 4 common strategies (rank
        correlation is degenerate below that), if the primary_metric
        is missing from any dict, or if either vector has zero
        variance (every strategy got the same score).

    Examples
    --------
    Sharpe-on-real vs Sharpe-on-synth-forward, scalar form. Note the
    per-strategy ``fn`` — calling ``my_backtest`` on every name with no
    per-strategy parameters gives every strategy the same Sharpe and
    triggers the zero-variance ValueError documented above::

        real_sharpes  = {name: fn(real_oos)['sharpe'] for name, fn in strategies.items()}
        synth_sharpes = {
            name: float(np.mean([fn(df)['sharpe'] for df in forward_paths.as_dataframes()]))
            for name, fn in strategies.items()
        }
        score = sf.predictive_rank_score(real_sharpes, synth_sharpes)
        print(score.summary())

    Multi-metric form — pick which metric to rank on::

        real_full  = {name: fn(real_oos)                       for name, fn in strategies.items()}
        synth_full = {
            name: {k: float(np.mean([fn(df)[k] for df in forward_paths.as_dataframes()]))
                   for k in ('sharpe', 'sortino')}
            for name, fn in strategies.items()
        }
        score = sf.predictive_rank_score(real_full, synth_full, primary_metric='sortino')
    """
    try:
        from scipy.stats import spearmanr
    except ImportError as exc:  # pragma: no cover — scipy is a hard dep
        raise ImportError(
            "scipy is required for predictive_rank_score. "
            "Install with: pip install sablier-flow[validation]"
        ) from exc

    if not real_results:
        raise ValueError("real_results is empty")
    if not synth_results:
        raise ValueError("synth_results is empty")

    real_flat, resolved_metric, real_form = _extract_primary(
        real_results, primary_metric
    )
    synth_flat, _, synth_form = _extract_primary(synth_results, resolved_metric)
    # 1.1.0 — cross-side form check. The pre-1.1 guard inside
    # _extract_primary fired whenever a scalar was seen with a non-default
    # primary_metric kwarg, which incorrectly rejected the legitimate
    # both-sides-scalar-with-explicit-label pattern (e.g.
    # predictive_rank_score({n: x}, {n: y}, primary_metric='sharpe')).
    # The real footgun is mixing forms ACROSS sides — real=dict[sharpe],
    # synth=scalar — which silently labels the scalar as the dict metric.
    # We catch that explicitly here.
    if real_form != synth_form:
        raise ValueError(
            f"real_results is in {real_form!r} form (e.g. "
            f"{{name: {'metric_dict' if real_form == 'dict' else 'scalar'}}}); "
            f"synth_results is in {synth_form!r} form. Pass both sides in "
            f"matching form — mixing dict-form and scalar-form silently "
            f"labels the rank correlation with the dict-side's metric name "
            f"while the scalar side could represent anything."
        )

    # Intersection of strategy names — defensive, the user may have
    # dropped a strategy from one side without realising.
    common = sorted(set(real_flat) & set(synth_flat))
    notes: list[str] = []

    dropped_from_real = sorted(set(real_flat) - set(synth_flat))
    if dropped_from_real:
        notes.append(
            f"{len(dropped_from_real)} strategies in real_results have "
            f"no matching synth_results entry — ignored: "
            f"{dropped_from_real[:5]}"
            + ("..." if len(dropped_from_real) > 5 else "")
        )
    dropped_from_synth = sorted(set(synth_flat) - set(real_flat))
    if dropped_from_synth:
        notes.append(
            f"{len(dropped_from_synth)} strategies in synth_results have "
            f"no matching real_results entry — ignored: "
            f"{dropped_from_synth[:5]}"
            + ("..." if len(dropped_from_synth) > 5 else "")
        )

    if len(common) < 4:
        raise ValueError(
            f"need at least 4 common strategies for a meaningful rank "
            f"correlation; got {len(common)} after intersecting "
            f"({len(real_flat)} real × {len(synth_flat)} synth). Add "
            f"more variants to the family — Sharpe rank correlation on "
            f"3 points is not interpretable."
        )

    if len(common) < 10:
        notes.append(
            f"only {len(common)} strategies — bootstrap CI on Spearman ρ "
            f"is high-variance below 10 variants; aim for ≥ 24 variants "
            f"for a tight CI."
        )

    real_vec = np.array([real_flat[s] for s in common], dtype=np.float64)
    synth_vec = np.array([synth_flat[s] for s in common], dtype=np.float64)

    finite = np.isfinite(real_vec) & np.isfinite(synth_vec)
    if finite.sum() < 4:
        raise ValueError(
            f"after dropping NaN/Inf entries only {int(finite.sum())} "
            f"strategies remain; cannot compute a meaningful rank "
            f"correlation. Inspect your backtest outputs for failing "
            f"variants and rerun."
        )
    if finite.sum() < len(common):
        notes.append(
            f"{len(common) - int(finite.sum())} strategies had NaN/Inf "
            f"in real or synth and were dropped from the rank correlation."
        )
    common = [s for s, ok in zip(common, finite, strict=True) if ok]
    real_vec = real_vec[finite]
    synth_vec = synth_vec[finite]

    if np.std(real_vec) == 0 or np.std(synth_vec) == 0:
        raise ValueError(
            "one of (real_results, synth_results) has zero variance "
            "across strategies — every strategy got the same score. "
            "Rank correlation is undefined; nothing to calibrate."
        )

    rho_full, p_value = spearmanr(real_vec, synth_vec)
    rho_full = float(rho_full)
    p_value = float(p_value)

    # Bootstrap CI: resample the strategy index with replacement
    # n_bootstrap times.
    rng = np.random.default_rng(seed)
    n = len(common)
    boot_rhos = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(real_vec[idx], synth_vec[idx])
        boot_rhos[i] = r if np.isfinite(r) else 0.0
    ci_lo, ci_hi = (
        float(np.percentile(boot_rhos, 2.5)),
        float(np.percentile(boot_rhos, 97.5)),
    )

    metric_gap = float(np.mean(np.abs(real_vec - synth_vec)))

    return PredictiveRankReport(
        spearman_rho=rho_full,
        p_value=p_value,
        ci_95=(ci_lo, ci_hi),
        n_strategies=len(common),
        mean_abs_metric_gap=metric_gap,
        primary_metric=resolved_metric,
        real_values={s: float(v) for s, v in zip(common, real_vec, strict=True)},
        synth_values={s: float(v) for s, v in zip(common, synth_vec, strict=True)},
        n_bootstrap=int(n_bootstrap),
        notes=notes,
    )
