"""Robustness scoring — the customer-facing overfit-detection report.

Given a backtest result on real data + N backtest results on synthetic
alternative-history paths, this module produces a :class:`RobustnessReport`
quantifying whether the strategy is overfit to the specific historical
sequence or robust across alternative draws from the same data-generating
process.

The headline number is **overfit_score** ∈ [0, 1]:
    0.50  real backtest sits at the median of the synthetic distribution.
          The strategy is consistent with the model's view of the DGP.
    >0.85 real backtest exceeds the synthetic distribution's 85th percentile.
          The strategy is likely overfit to historical idiosyncrasies.
    >0.95 real backtest sits at the top 5%. Probably very overfit.

The argument that justifies this number is in
:doc:`/concepts/in-sample-is-correct` — synthetic samples come from a
generator that learned the joint distribution; if a strategy only works
on the specific realization the data took, it must be exploiting noise
that's by-construction not in the learned distribution.

Public API:
    :func:`robustness` — compute report from real result + list of synth results.
    :class:`RobustnessReport` — return type.

The input format is intentionally flexible: each "backtest result" is
either a Python scalar (typically Sharpe ratio) or a dict of named
metrics. The function infers and reports across whatever keys are
present.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from sablier_flow.analytics.deflated_sharpe import DeflatedSharpeReport

__all__ = [
    "RobustnessReport",
    "robustness",
]


# ============================================================================
# Output type
# ============================================================================


_RobustnessVerdict = Literal[
    "robust",
    "borderline",
    "overfit",
    "highly_overfit",
    "degenerate_synth",
    "insufficient_data",
]


@dataclass(frozen=True)
class RobustnessReport:
    """The customer-facing verdict on a strategy.

    Built from one real-backtest result + N synthetic-backtest results.
    Designed for direct rendering in a UI or notebook.
    """

    overfit_score: float
    """Fraction of synthetic backtests that scored lower than the real
    backtest, on the primary metric. 0.50 = consistent with the DGP;
    >0.85 = real backtest sits at the top 15% of the synthetic
    distribution = overfit signal."""

    verdict: _RobustnessVerdict
    """Bucketed interpretation:
        robust         overfit_score in [0, 0.70)
        borderline     [0.70, 0.85)
        overfit        [0.85, 0.95)
        highly_overfit [0.95, 1.0]
    """

    primary_metric: str
    """The metric used to compute overfit_score. Defaults to 'sharpe' if
    present; otherwise the first numeric metric found."""

    real_value: float
    """The strategy's real-backtest value on ``primary_metric``."""

    synthetic_mean: float
    synthetic_median: float
    synthetic_std: float
    synthetic_min: float
    synthetic_max: float
    synthetic_p5: float
    synthetic_p25: float
    synthetic_p75: float
    synthetic_p95: float
    """Distribution statistics of the synthetic backtests on the primary
    metric. Use these to build a fan chart or histogram in the UI."""

    synthetic_ci_95: tuple[float, float]
    """95% confidence interval for the synthetic distribution
    (5th and 95th percentiles)."""

    n_synthetic: int
    """How many synthetic backtests went into the report."""

    per_metric: dict[str, dict[str, float]] = field(default_factory=dict)
    """If the inputs were dicts, this contains the same distribution stats
    keyed by metric name. Lets a UI render multiple panels (Sharpe,
    drawdown, etc.)."""

    notes: list[str] = field(default_factory=list)
    """Non-fatal warnings — e.g. "only 50 synthetic paths; report is
    high-variance"."""

    synthetic_values: tuple[float, ...] = ()
    """The raw per-path values on the primary metric. Retained so
    downstream calls like :meth:`deflated_sharpe` can recompute
    distribution-shape statistics without re-running the backtests."""

    higher_is_better: bool = True
    """Direction of the primary metric. Sharpe / return / win-rate are
    higher-is-better; drawdown / shortfall are lower-is-better. Drives
    the sign convention in ``deflated_sharpe``."""

    @property
    def acceptable(self) -> bool | None:
        """True if the verdict is ``robust`` or ``borderline``. UIs can use
        this to gate "deploy to live" workflows.

        Returns ``None`` when the verdict is ``insufficient_data`` —
        there's literally no signal to bucket on, so neither "ship" nor
        "block" is defensible; the customer needs to collect more
        synthetic backtests first.
        """
        if self.verdict == "insufficient_data":
            return None
        return self.verdict in ("robust", "borderline")

    def summary(self) -> str:
        """One plain-English sentence translating the verdict + DSR for a
        non-PhD reader.

        Lead with the action the customer should take; trail with the
        number they need to defend the decision. Reads cleanly from
        Slack / a PR description / a fund's CI output. The DSR numbers
        are still available through ``deflated_sharpe()`` for the quant
        who wants them.
        """
        metric = self.primary_metric
        real = self.real_value
        lo, hi = self.synthetic_p5, self.synthetic_p95

        # Edge verdicts short-circuit the full narrative — the percentile
        # mass is not meaningful enough to dress up with a CI sentence.
        if self.verdict == "insufficient_data":
            return (
                f"Insufficient data: only {self.n_synthetic} synthetic backtest"
                f"{'s' if self.n_synthetic != 1 else ''} provided on {metric}; "
                "robustness analysis requires >=10 (>=30 recommended). "
                "Generate more alt-history paths and re-run."
            )
        if self.verdict == "degenerate_synth":
            return (
                f"Degenerate synthetic distribution: all {self.n_synthetic} "
                f"alt-history backtests returned (effectively) identical "
                f"{metric} values. The verdict is not meaningful — most "
                "likely the generator collapsed or the strategy is "
                "constant across draws. Re-fit the generator or vary the "
                "strategy before interpreting overfit_score."
            )

        # ``robust`` means "no evidence of overfit" — that's orthogonal to
        # whether the strategy outperformed. A money-loser can be robust
        # (just bad, not overfit); we shouldn't say it's "in line" when
        # ``real`` sat outside the 5-95 CI. Lead with where the value
        # actually lands; only call it "in line" when it really is.
        if lo <= real <= hi:
            robust_msg = (
                f"Robust: {metric} of {real:+.3f} is in line with the "
                f"alt-history distribution (95% CI [{lo:+.3f}, {hi:+.3f}]). "
                "No overfit signal."
            )
        elif real < lo:
            robust_msg = (
                f"Robust: {metric} of {real:+.3f} falls below the 5th "
                f"percentile of the alt-history distribution "
                f"(CI [{lo:+.3f}, {hi:+.3f}]), in the direction of "
                "underperformance — no overfit signal "
                "(robust = not overfit, NOT a profitability verdict)."
            )
        else:  # real > hi but verdict still robust (rare — usually flips to overfit)
            robust_msg = (
                f"Robust: {metric} of {real:+.3f} sits above the 95th "
                f"percentile of the alt-history distribution "
                f"(CI [{lo:+.3f}, {hi:+.3f}]), but the overfit_score "
                f"({self.overfit_score:.0%}) and DSR pass the robustness "
                "thresholds."
            )

        # Top-tail caveat (0.7.2+) — when the verdict is `overfit` /
        # `highly_overfit` on a SINGLE strategy the label can be
        # misleading: a fixed parameterless strategy (e.g. buy-and-hold
        # in a rally) lands in the top tail not because it was
        # over-tuned but because the realization happened to favour it.
        # The percentile rank cannot distinguish "selected from a
        # search" from "lucky window" on its own data — only
        # ``sf.evaluate_family`` (CSCV-PBO) can. Surface this in-line so
        # the customer doesn't read the label as causal evidence.
        causality_caveat = (
            " This label assumes the strategy was selected from a "
            "search (one of many tested). If you ran a single fixed "
            "strategy with no parameter tuning, treat this as "
            "'real outperformed alt-histories' (skill OR luck), not "
            "evidence of overfit — for multi-strategy overfit "
            "detection use sf.evaluate_family (CSCV-PBO)."
        )

        prefix = {
            "robust": robust_msg,
            "borderline": (
                f"Borderline: {metric} of {real:+.3f} sits at the {self.overfit_score:.0%} "
                f"percentile of the alt-history distribution (CI [{lo:+.3f}, {hi:+.3f}]). "
                "Defensible but not unambiguous — increase n_paths on the "
                "next generate() to sharpen the percentile, and use "
                "sf.evaluate_family (CSCV-PBO) if multiple strategy "
                "variants are on the table."
            ),
            "overfit": (
                f"Overfit: {metric} of {real:+.3f} exceeded {self.overfit_score:.0%} of "
                f"alt-histories (CI [{lo:+.3f}, {hi:+.3f}]). "
                "Strategy likely exploits realization-specific noise."
                + causality_caveat
            ),
            "highly_overfit": (
                f"Highly overfit: {metric} of {real:+.3f} exceeded {self.overfit_score:.0%} of "
                f"alt-histories (CI [{lo:+.3f}, {hi:+.3f}]). "
                "Do not deploy without re-validating on out-of-sample data."
                + causality_caveat
            ),
        }.get(self.verdict, f"Verdict {self.verdict!r}: real={real:+.3f}, CI [{lo:+.3f}, {hi:+.3f}].")

        # If DSR is computable, append the realistic-null threshold so the
        # customer has a concrete "you'd need SR X to clear this" anchor.
        if self.higher_is_better and self.synthetic_values:
            try:
                dsr = self.deflated_sharpe(n_trials=1)
                prefix += (
                    f" Under the realistic null, you'd need {metric} ≥ "
                    f"{dsr.threshold_sr_realistic:+.3f} to clear the 95% "
                    "significance bar."
                )
            except Exception:  # never let summary() crash on edge inputs
                pass
        return prefix

    def to_html(self, path: str | None = None, *, title: str = "Robustness Report") -> str:
        """Render as a self-contained HTML document — the shareable artifact.

        No external dependencies; embeds an inline-SVG distribution bar
        showing the position of the real backtest within the synthetic
        distribution. If ``path`` is given, writes it and returns the
        path string; otherwise returns the HTML as a string.

        The output is single-file (no remote assets, no JavaScript) so it
        renders identically inside email previews, GitHub READMEs, and
        confluence/notion pages.
        """
        html = _render_report_html(self, title=title)
        if path is None:
            return html
        from pathlib import Path as _Path
        _Path(path).write_text(html, encoding="utf-8")
        return str(path)

    def _repr_html_(self) -> str:
        """Jupyter rich-display hook — returning the same HTML the
        notebook renderer would show. Removes the need to call
        ``.to_html()`` manually when inspecting in a cell."""
        return _render_report_html(self, title="Robustness Report")

    def deflated_sharpe(
        self,
        *,
        strategy_returns: Sequence[float] | np.ndarray | None = None,
        n_trials: int = 1,
    ) -> DeflatedSharpeReport:
        """Compute the Deflated Sharpe Ratio under the Sablier realistic
        null alongside the analytical Bailey-López-de-Prado IID-Gaussian
        null.

        The realistic null is regime-aware where the analytical null is
        not. The fold-conditional realistic DSR varies substantially
        across market regimes; the analytical correction does not.

        Only meaningful for higher-is-better metrics (Sharpe, return); for
        drawdown-like metrics the DSR concept does not directly apply.

        Parameters
        ----------
        strategy_returns
            Per-period returns of the strategy on the real backtest. If
            provided, enables the Bailey-LdP skew/kurtosis correction in
            the analytical DSR. If omitted, falls back to no correction
            (γ₃ = 0, γ₄ = 3, T = 252 daily observations).
        n_trials
            Number of strategies searched. 1 for a single backtest; M for
            an M-strategy family. The analytical ``E[max]`` term grows
            with N; for the family case, ``synthetic_values`` should
            already contain the per-path best-of-N.

        Returns
        -------
        DeflatedSharpeReport
        """
        if not self.higher_is_better:
            raise ValueError(
                "deflated_sharpe is only defined for higher-is-better metrics "
                f"(got higher_is_better=False on metric {self.primary_metric!r})"
            )
        if not self.synthetic_values:
            raise ValueError(
                "this RobustnessReport was built without retaining synthetic_values "
                "— rebuild with sablier_flow.robustness(...) on this SDK version"
            )
        from sablier_flow.analytics.deflated_sharpe import (
            deflated_sharpe as _dsr,
        )

        return _dsr(
            observed_sr=self.real_value,
            synthetic_sharpes=self.synthetic_values,
            strategy_returns=strategy_returns,
            n_trials=n_trials,
        )


# ============================================================================
# Public entry point
# ============================================================================


def robustness(
    real_result: float | dict[str, float],
    synthetic_results: Sequence[float | dict[str, float]],
    *,
    primary_metric: str | None = None,
    higher_is_better: bool = True,
) -> RobustnessReport:
    """Build a robustness report from one real backtest and N synthetic backtests.

    Single-strategy caveat
    ----------------------
    The verdict (``robust`` / ``borderline`` / ``overfit`` /
    ``highly_overfit``) ranks the real result against the synthetic
    distribution — that's a **directional** measurement, not a
    **causal** one. A single fixed strategy with no parameters to tune
    (e.g. buy-and-hold in a rally) can land in the top tail simply
    because the realization favoured it. The verdict alone cannot
    distinguish "selected from a search" from "lucky window" on a
    single strategy; treat the overfit label as "real outperformed
    alt-histories" until you've ruled out luck with a family-level
    test. :func:`evaluate_family` (CSCV-PBO) is the right tool when
    multiple strategies are on the table — its PBO statistic is
    luck-safe by construction. ``summary()`` and the ``notes`` field
    surface this caveat in-line for overfit verdicts.

    Parameters
    ----------
    real_result
        The strategy's backtest on real data. Either a scalar (typically
        Sharpe ratio) or a dict of named metrics. If a dict, ``primary_metric``
        selects which key drives the verdict.
    synthetic_results
        Backtests on synthetic alternative-history paths. Must match the
        type of ``real_result`` (all scalars or all dicts).
    primary_metric
        If results are dicts, which metric to bucket. Defaults to
        ``'sharpe'`` if present, otherwise the first numeric key.
    higher_is_better
        Whether higher values are better (default True — Sharpe, returns).
        Set False for drawdown-like metrics where lower is better;
        ``overfit_score`` then measures how far in the *worse* direction
        the real result is relative to synthetic.

    Returns
    -------
    RobustnessReport
        A frozen dataclass. Key fields and methods customers actually use:

        - ``verdict`` — one of ``'robust'``, ``'borderline'``, ``'overfit'``,
          ``'highly_overfit'``. Drives the deploy / don't-deploy decision.
        - ``overfit_score`` — fraction of synthetic backtests the real one
          beat (0.50 = consistent with the DGP; ≥0.85 = overfit signal).
        - ``real_value``, ``synthetic_median``, ``synthetic_ci_95`` — the
          headline numbers for a chart.
        - ``synthetic_values`` — the raw per-path metric values, in case
          you want to render your own histogram.
        - ``per_metric`` — same distribution stats for every metric in
          the input dicts (populated only when inputs were dicts).
        - ``acceptable`` (property) — ``True`` if verdict is
          ``robust`` or ``borderline``; gate live-deploy on this.
        - ``summary()`` — one English sentence including the DSR
          threshold, ready to paste into Slack or a PR.
        - ``deflated_sharpe(n_trials=N)`` — Bailey-LdP DSR under both
          analytical (IID-Gaussian) and realistic (Sablier) nulls;
          pass ``n_trials`` if you grid-searched the strategy.
        - ``to_html(path=...)`` — single-file self-contained HTML report
          for sharing with PMs / stakeholders.

    Raises
    ------
    ValueError
        If inputs are empty, mixed scalar/dict, or missing the
        ``primary_metric``.

    Examples
    --------
    Minimal scalar form (Sharpe-only)::

        verdict = robustness(1.64, synth_sharpes)
        print(verdict.verdict, verdict.overfit_score)

    Dict form (pick the bucketing metric explicitly)::

        verdict = robustness(
            {"sharpe": 1.64, "calmar": 0.9, "max_dd": -0.18},
            synth_dicts,
            primary_metric="sharpe",
        )
    """
    synthetic_results = list(synthetic_results)
    if not synthetic_results:
        raise ValueError("synthetic_results is empty — need at least 1 backtest")

    notes: list[str] = []
    if len(synthetic_results) < 100:
        notes.append(
            f"only {len(synthetic_results)} synthetic backtests; report is "
            "high-variance. Aim for >= 500 paths for stable estimates."
        )

    # ------------------------------------------------------------------
    # Normalize inputs to dict form for uniform handling
    # ------------------------------------------------------------------
    real_is_scalar = isinstance(real_result, (int, float, np.floating, np.integer))
    synth_are_scalars = all(
        isinstance(r, (int, float, np.floating, np.integer)) for r in synthetic_results
    )

    if real_is_scalar and not synth_are_scalars:
        raise ValueError("real_result is a scalar but some synthetic_results are dicts")
    if not real_is_scalar and synth_are_scalars:
        raise ValueError("real_result is a dict but synthetic_results are scalars")

    if real_is_scalar:
        # Wrap scalars into a single-metric dict for uniform handling
        real_scalar = cast("float", real_result)
        synth_scalars = cast("list[float]", synthetic_results)
        real_dict: dict[str, float] = {"value": float(real_scalar)}
        synth_dicts: list[dict[str, float]] = [{"value": float(r)} for r in synth_scalars]
        if primary_metric is None:
            primary_metric = "value"
    else:
        real_d = cast("dict[str, float]", real_result)
        synth_ds = cast("list[dict[str, float]]", synthetic_results)
        real_dict = {k: float(v) for k, v in real_d.items()}
        synth_dicts = [{k: float(v) for k, v in r.items()} for r in synth_ds]
        # Pick primary metric
        if primary_metric is None:
            primary_metric = "sharpe" if "sharpe" in real_dict else next(iter(real_dict))
        if primary_metric not in real_dict:
            raise ValueError(
                f"primary_metric={primary_metric!r} not in real_result; "
                f"available keys: {list(real_dict)}"
            )

    # ------------------------------------------------------------------
    # Compute per-metric distribution stats
    # ------------------------------------------------------------------
    per_metric: dict[str, dict[str, float]] = {}
    # Intersection of keys across real + all synthetic dicts.
    # set.intersection(*sets) wants positional args; build the list first.
    synth_key_sets: list[set[str]] = [set(d) for d in synth_dicts]
    common_keys: set[str] = set(real_dict).intersection(*synth_key_sets)
    for key in common_keys:
        synth_vals = np.array([d[key] for d in synth_dicts], dtype=np.float64)
        synth_vals = synth_vals[np.isfinite(synth_vals)]
        if len(synth_vals) == 0:
            continue
        per_metric[key] = _distribution_stats(synth_vals)

    # ------------------------------------------------------------------
    # Compute overfit_score on the primary metric
    # ------------------------------------------------------------------
    primary_synth = np.array(
        [d[primary_metric] for d in synth_dicts if primary_metric in d],
        dtype=np.float64,
    )
    # Drop NaN/Inf synthetic values defensively. If the drop changes the
    # count, warn — silent shrinkage of the alt-history distribution
    # would distort the verdict without the customer realising.
    primary_synth_raw_len = len(primary_synth)
    primary_synth = primary_synth[np.isfinite(primary_synth)]
    n_dropped = primary_synth_raw_len - len(primary_synth)
    if n_dropped > 0:
        notes.append(
            f"dropped {n_dropped} non-finite (NaN/Inf) synthetic "
            f"{primary_metric!r} value{'s' if n_dropped != 1 else ''} before "
            "computing overfit_score."
        )
    real_value = real_dict[primary_metric]

    # Fix #1 — real_value NaN/Inf is fatal: there's no number to bucket.
    if not np.isfinite(real_value):
        raise ValueError(
            f"real_result[{primary_metric!r}] is NaN/Inf — cannot compute "
            "robustness (no finite real backtest value to rank against the "
            "synthetic distribution)."
        )

    if len(primary_synth) == 0:
        raise ValueError(f"all synthetic values for {primary_metric!r} are NaN/Inf")

    # Fraction of synthetic backtests where the strategy did WORSE than reality.
    # If higher_is_better: count synth < real. If lower_is_better: count synth > real.
    if higher_is_better:
        overfit_score = float(np.mean(primary_synth < real_value))
    else:
        overfit_score = float(np.mean(primary_synth > real_value))

    # ------------------------------------------------------------------
    # Bucket the verdict
    # ------------------------------------------------------------------
    stats = per_metric.get(primary_metric, _distribution_stats(primary_synth))

    # Fix #3 — n_synthetic floor: <10 alt-history backtests is too thin
    # to bucket meaningfully. Anchor the verdict to insufficient_data
    # rather than producing a confident-looking score off 3 points.
    n_finite = len(primary_synth)
    if n_finite < 10:
        verdict: _RobustnessVerdict = "insufficient_data"
        notes.append(
            f"only {n_finite} synthetic value"
            f"{'s' if n_finite != 1 else ''} provided; robustness analysis "
            "requires >=10 (>=30 recommended). Treat overfit_score as "
            "indicative only and rerun with more alt-history paths."
        )
    # Fix #2 — degenerate constant input: when the synthetic distribution
    # collapsed to a single value (std == 0 or p5 == p95), the percentile
    # rank is a step function and the verdict isn't meaningful. Common
    # causes: generator mode-collapse, or a strategy whose output
    # doesn't depend on the path.
    elif stats["std"] == 0.0 or stats["p5"] == stats["p95"]:
        verdict = "degenerate_synth"
        notes.append(
            "synthetic distribution is degenerate — all alt-history "
            f"backtests returned identical {primary_metric!r} values "
            "(std == 0 / p5 == p95); verdict is not meaningful. Re-fit "
            "the generator (mode collapse?) or vary the strategy across "
            "draws before interpreting overfit_score."
        )
    elif overfit_score >= 0.95:
        verdict = "highly_overfit"
    elif overfit_score >= 0.85:
        verdict = "overfit"
    elif overfit_score >= 0.70:
        verdict = "borderline"
    else:
        verdict = "robust"

    # Single-strategy luck-vs-overfit caveat (0.7.2+) — the percentile
    # rank cannot distinguish "selected from a search" from "single
    # fixed strategy in a favourable window" on its own. Surface this
    # in-line whenever the verdict is in overfit territory so the
    # customer can't mistake a directional descriptor for a causal
    # diagnosis. ``evaluate_family`` (CSCV-PBO) is the right tool when
    # multiple strategies are on the table.
    if verdict in ("overfit", "highly_overfit"):
        notes.append(
            "Single-strategy overfit verdicts assume the strategy was "
            "selected from a search. If you tested ONE fixed strategy "
            "with no parameter tuning, this label conflates skill, "
            "luck, and overfit — read it as 'real outperformed the "
            "alt-history distribution', not 'real is overfit'. Use "
            "sf.evaluate_family for true overfit detection on a "
            "strategy family (CSCV-PBO is luck-safe)."
        )

    # Fix #4 — borderline verdict needs an actionable next step. Mirror
    # the overfit-branch wording: surface evaluate_family (CSCV-PBO) for
    # multi-strategy disambiguation, and the increase-n_paths tactic
    # for tightening the percentile estimate on the next pass.
    if verdict == "borderline":
        notes.append(
            "Borderline verdict — the real result sits in the 70-85% "
            "percentile band of the alt-history distribution. To "
            "disambiguate skill / luck / overfit, (a) increase n_paths "
            "on the next generate() call to tighten the percentile "
            "estimate, and (b) run sf.evaluate_family (CSCV-PBO) if "
            "multiple strategy variants are on the table — PBO is "
            "luck-safe by construction."
        )

    return RobustnessReport(
        overfit_score=overfit_score,
        verdict=verdict,
        primary_metric=primary_metric,
        real_value=float(real_value),
        synthetic_mean=stats["mean"],
        synthetic_median=stats["median"],
        synthetic_std=stats["std"],
        synthetic_min=stats["min"],
        synthetic_max=stats["max"],
        synthetic_p5=stats["p5"],
        synthetic_p25=stats["p25"],
        synthetic_p75=stats["p75"],
        synthetic_p95=stats["p95"],
        synthetic_ci_95=(stats["p5"], stats["p95"]),
        n_synthetic=len(primary_synth),
        per_metric=per_metric,
        notes=notes,
        synthetic_values=tuple(float(v) for v in primary_synth),
        higher_is_better=higher_is_better,
    )


# ============================================================================
# Internal helpers
# ============================================================================


def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    """Build the standard distribution stats dict from a 1-D array."""
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


# ============================================================================
# Pretty-print helper for notebooks
# ============================================================================


def _format_report_markdown(report: RobustnessReport) -> str:
    """Render a RobustnessReport as a markdown summary block — useful in
    notebooks and CLI output. Not part of the public API yet (will move
    to adapters/ in a later commit)."""
    verdict_emoji = {
        "robust": "✅",
        "borderline": "⚠️",
        "overfit": "🟠",
        "highly_overfit": "🔴",
    }
    emoji = verdict_emoji.get(report.verdict, "")
    return f"""\
{emoji} **{report.verdict.upper().replace('_', ' ')}** — overfit score: {report.overfit_score:.2%}

Primary metric: `{report.primary_metric}`

  Real backtest:       {report.real_value:+.4f}
  Synthetic median:    {report.synthetic_median:+.4f}
  Synthetic 95% CI:    [{report.synthetic_p5:+.4f}, {report.synthetic_p95:+.4f}]
  N synthetic paths:   {report.n_synthetic}
"""


# ============================================================================
# HTML rendering — the shareable artifact (no external deps, single file)
# ============================================================================


_VERDICT_COLOR = {
    "robust": ("#16a34a", "ROBUST"),                # green
    "borderline": ("#eab308", "BORDERLINE"),         # amber
    "overfit": ("#f97316", "OVERFIT"),               # orange
    "highly_overfit": ("#dc2626", "HIGHLY OVERFIT"),  # red
}


def _esc(s: str) -> str:
    """Minimal HTML escape — verdicts and metric names only, no user free-form text."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_distribution_svg(report: RobustnessReport, width: int = 640, height: int = 110) -> str:
    """Inline SVG showing the synthetic distribution range with the real
    value marked. Uses the 5/25/50/75/95 percentiles as boxplot-like ticks.
    """
    pad = 30
    inner_w = width - 2 * pad
    y_mid = height // 2

    lo = report.synthetic_min
    hi = report.synthetic_max
    if not (hi > lo):
        hi = lo + 1e-9  # avoid div-by-zero on degenerate distributions

    real_clamped = min(max(report.real_value, lo), hi)

    def x(v: float) -> float:
        return pad + (v - lo) / (hi - lo) * inner_w

    # 5–95 band
    band_x = x(report.synthetic_p5)
    band_w = max(1.0, x(report.synthetic_p95) - band_x)
    # 25–75 box
    box_x = x(report.synthetic_p25)
    box_w = max(1.0, x(report.synthetic_p75) - box_x)
    # Median tick
    med_x = x(report.synthetic_median)
    # Real value marker
    real_x = x(real_clamped)

    color, _ = _VERDICT_COLOR.get(report.verdict, ("#444", "?"))

    return f"""\
<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Synthetic distribution with real backtest marker">
  <rect x="{pad}" y="{y_mid - 1}" width="{inner_w}" height="2" fill="#cbd5e1"/>
  <rect x="{band_x:.1f}" y="{y_mid - 14}" width="{band_w:.1f}" height="28"
        fill="#dbeafe" rx="3"/>
  <rect x="{box_x:.1f}" y="{y_mid - 18}" width="{box_w:.1f}" height="36"
        fill="#93c5fd" rx="3"/>
  <line x1="{med_x:.1f}" y1="{y_mid - 20}" x2="{med_x:.1f}" y2="{y_mid + 20}"
        stroke="#1e3a8a" stroke-width="2"/>
  <line x1="{real_x:.1f}" y1="{y_mid - 28}" x2="{real_x:.1f}" y2="{y_mid + 28}"
        stroke="{color}" stroke-width="3"/>
  <circle cx="{real_x:.1f}" cy="{y_mid}" r="6" fill="{color}" stroke="white" stroke-width="2"/>
  <text x="{pad}" y="{height - 4}" font-size="11" fill="#64748b">min {report.synthetic_min:.3g}</text>
  <text x="{width - pad}" y="{height - 4}" font-size="11" fill="#64748b"
        text-anchor="end">max {report.synthetic_max:.3g}</text>
  <text x="{real_x:.1f}" y="14" font-size="11" fill="{color}" text-anchor="middle"
        font-weight="bold">real: {report.real_value:.3g}</text>
</svg>"""


def _render_report_html(report: RobustnessReport, *, title: str) -> str:
    """Self-contained HTML document — no external CSS, no JS, no remote assets."""
    color, label = _VERDICT_COLOR.get(report.verdict, ("#444", report.verdict.upper()))
    svg = _render_distribution_svg(report)

    metric_rows = ""
    for key, stats in sorted(report.per_metric.items()):
        marker = " (primary)" if key == report.primary_metric else ""
        metric_rows += (
            f"<tr><td>{_esc(key)}{marker}</td>"
            f"<td>{stats['median']:.4f}</td>"
            f"<td>{stats['p5']:.4f}</td>"
            f"<td>{stats['p95']:.4f}</td>"
            f"<td>{stats['std']:.4f}</td></tr>"
        )

    # ----- DSR panel -----
    # Best-effort: only renders when the report retained synthetic_values
    # and the metric is higher-is-better. Older payloads or drawdown-style
    # metrics silently skip it.
    dsr_panel = ""
    if report.higher_is_better and report.synthetic_values:
        try:
            dsr = report.deflated_sharpe(n_trials=1)
            dsr_panel = (
                "<div class='panel'><h2>Deflated Sharpe (Bailey-LdP)</h2>"
                "<table><thead><tr><th>Null distribution</th><th>DSR</th>"
                "<th>E[max SR]</th><th>SR threshold (DSR=0.95)</th>"
                "</tr></thead><tbody>"
                f"<tr><td>Sablier realistic (regime-aware)</td>"
                f"<td>{dsr.realistic:.3f}</td>"
                f"<td>{dsr.expected_max_sr_realistic:+.4f}</td>"
                f"<td>{dsr.threshold_sr_realistic:+.4f}</td></tr>"
                f"<tr><td>Bailey-LdP analytical IID-Gaussian</td>"
                f"<td>{dsr.analytical:.3f}</td>"
                f"<td>{dsr.expected_max_sr_analytical:+.4f}</td>"
                f"<td>{dsr.threshold_sr_analytical:+.4f}</td></tr>"
                "</tbody></table>"
                "<div style='font-size:12px;color:#64748b;margin-top:8px;'>"
                "Realistic null is the empirical distribution of synthetic-best "
                "Sharpes; the analytical null is the closed-form Bailey-LdP IID "
                "expression."
                "</div></div>"
            )
        except Exception:
            # Compute failure shouldn't break the HTML output — skip the panel.
            dsr_panel = ""

    notes_html = ""
    if report.notes:
        notes_html = (
            "<div class='notes'><strong>Notes</strong><ul>"
            + "".join(f"<li>{_esc(n)}</li>" for n in report.notes)
            + "</ul></div>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(title)} — sablier-flow</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 32px; max-width: 760px; color: #0f172a; background: #f8fafc;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px; font-weight: 600; }}
  .subtitle {{ color: #64748b; font-size: 13px; margin-bottom: 24px; }}
  .verdict-card {{
    background: white; border-radius: 8px; padding: 24px; margin-bottom: 16px;
    border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }}
  .verdict-badge {{
    display: inline-block; color: white; background: {color};
    padding: 4px 12px; border-radius: 4px; font-weight: 600;
    font-size: 12px; letter-spacing: 0.5px;
  }}
  .score {{ font-size: 42px; font-weight: 700; margin: 12px 0 4px; }}
  .score-label {{ color: #64748b; font-size: 13px; margin-bottom: 18px; }}
  .panel {{
    background: white; border-radius: 8px; padding: 20px; margin-bottom: 16px;
    border: 1px solid #e2e8f0;
  }}
  .panel h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.6px;
              color: #475569; margin: 0 0 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #f1f5f9; }}
  th {{ color: #64748b; font-weight: 600; font-size: 12px;
        text-transform: uppercase; letter-spacing: 0.4px; }}
  td:not(:first-child) {{ text-align: right; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px 24px; }}
  .stat-grid div {{ display: flex; justify-content: space-between;
                    padding: 6px 0; border-bottom: 1px solid #f1f5f9; }}
  .stat-grid .label {{ color: #64748b; }}
  .notes {{ background: #fef3c7; padding: 12px 16px; border-radius: 6px;
            font-size: 13px; margin-top: 16px; }}
  .notes ul {{ margin: 4px 0 0 16px; padding: 0; }}
  footer {{ margin-top: 24px; color: #94a3b8; font-size: 12px; }}
  footer a {{ color: #64748b; }}
</style>
</head>
<body>
  <h1>{_esc(title)}</h1>
  <div class="subtitle">
    Primary metric: <strong>{_esc(report.primary_metric)}</strong> &middot;
    {report.n_synthetic} synthetic paths
  </div>

  <div class="verdict-card">
    <span class="verdict-badge">{label}</span>
    <div class="score">{report.overfit_score:.2%}</div>
    <div class="score-label">
      Fraction of synthetic backtests the real strategy exceeded on
      <code>{_esc(report.primary_metric)}</code>.
      0.50 = consistent with the data-generating process; &gt;0.85 = overfit signal.
    </div>
    {svg}
  </div>

  <div class="panel">
    <h2>Distribution on primary metric</h2>
    <div class="stat-grid">
      <div><span class="label">Real backtest</span>
           <strong>{report.real_value:+.4f}</strong></div>
      <div><span class="label">Synthetic median</span>
           <strong>{report.synthetic_median:+.4f}</strong></div>
      <div><span class="label">Synthetic mean</span>
           <strong>{report.synthetic_mean:+.4f}</strong></div>
      <div><span class="label">Synthetic std</span>
           <strong>{report.synthetic_std:.4f}</strong></div>
      <div><span class="label">95% CI</span>
           <strong>[{report.synthetic_p5:+.4f}, {report.synthetic_p95:+.4f}]</strong></div>
      <div><span class="label">Min / max</span>
           <strong>{report.synthetic_min:+.4f} / {report.synthetic_max:+.4f}</strong></div>
    </div>
  </div>

  {(
    "<div class='panel'><h2>All metrics</h2>"
    "<table><thead><tr><th>Metric</th><th>Median</th><th>5%</th><th>95%</th><th>Std</th>"
    "</tr></thead><tbody>"
    + metric_rows
    + "</tbody></table></div>"
  ) if metric_rows else ""}

  {dsr_panel}

  {notes_html}

  <footer>
    Generated by <a href="https://sablier.ai">sablier-flow</a>.
    Methodology: synthetic samples drawn from a generator trained on the
    same window as the real backtest; an overfit strategy exploits
    realization-specific noise that's by construction not in the learned
    distribution.
  </footer>
</body>
</html>
"""
