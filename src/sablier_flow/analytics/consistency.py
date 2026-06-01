"""Live drift monitoring — has reality drifted from the baseline
synthetic distribution since deployment?

The flow:

  1. Pre-deployment: customer runs ``robustness(...)`` or ``@augment``
     on the strategy they're about to deploy, gets a
     :class:`RobustnessReport` that retains the synthetic distribution
     of the chosen metric.
  2. Customer persists the report (pickle / JSON / DB row — the report
     is a frozen dataclass).
  3. Post-deployment: every N weeks of live trading, customer feeds
     the strategy's realised metric back into
     :func:`consistency_check`, passing the original baseline.
  4. The returned :class:`ConsistencyReport` says whether the realised
     value is still inside the modeled distribution, and if not, by
     how much.

This is the highest-LTV signal for a fund post-deployment: it converts
the model from a one-shot "ship-or-not" gate into a continuous
"recalibrate-now?" alarm. Cheap to compute (no GPU, no network) and
auditable.

The semantics map onto a one-sided control chart:

  - **consistent** — realised value is within the baseline's 95% CI
  - **drifting** — outside the 95% CI but inside the [min, max] envelope
  - **out_of_distribution** — outside the baseline's [min, max] envelope
    entirely — the regime has clearly shifted and the model is stale

We do not advance a single recipe for "what to do next". The customer
either retrains, recalibrates parameters, or pulls the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

if TYPE_CHECKING:
    from sablier_flow.analytics.robustness import RobustnessReport

__all__ = ["ConsistencyReport", "ConsistencyVerdict", "consistency_check"]


ConsistencyVerdict = Literal["consistent", "drifting", "out_of_distribution"]


@dataclass(frozen=True)
class ConsistencyReport:
    """Result of comparing a realised backtest value against a previously
    generated synthetic distribution.

    Designed for direct logging into a monitoring system: the
    ``verdict`` field is the alarm level and ``drift_score`` is the
    magnitude.
    """

    realized_value: float
    """The realized value the customer fed in."""

    baseline_median: float
    """Median of the baseline synthetic distribution."""

    baseline_p5: float
    baseline_p95: float
    """5th and 95th percentile of the baseline distribution — the CI
    edges that define the ``drifting`` boundary."""

    baseline_min: float
    baseline_max: float
    """Minimum and maximum of the baseline distribution — beyond these,
    the realised value is ``out_of_distribution``."""

    verdict: ConsistencyVerdict
    """``"consistent"`` / ``"drifting"`` / ``"out_of_distribution"``.
    Use this as the alarm level in monitoring dashboards."""

    drift_score: float
    """Standardised distance of the realised value from the baseline
    median, scaled by half the 5-95 CI width::

        drift_score = (realized - median) / (0.5 * (p95 - p5))

    For higher-is-better metrics: positive = realised better than
    baseline median; negative = worse. |drift_score| < 1 corresponds
    roughly to inside the 90% CI; |drift_score| > ~1.5 is
    ``out_of_distribution`` territory."""

    empirical_cdf: float
    """The empirical CDF position of the realised value in the baseline
    distribution. 0.5 = at median; 0.0 = below every baseline draw;
    1.0 = above every baseline draw. For lower-is-better metrics the
    customer should read this with ``higher_is_better=False``
    semantics — see ``consistency_check``'s parameter."""

    higher_is_better: bool
    """Direction of the underlying metric (carried forward from the
    baseline). Affects only the textual ``notes``, not the numerical
    fields."""

    n_baseline_paths: int
    """How many synthetic paths formed the baseline distribution."""

    notes: list[str] = field(default_factory=list)
    """Plain-English commentary on the verdict — designed to read
    cleanly in a Slack alert or PR description."""

    @property
    def acceptable(self) -> bool:
        """True when the realised value is ``'consistent'`` with the
        baseline distribution. ``'drifting'`` and ``'out_of_distribution'``
        both flip to False — monitoring callers can gate alerts on
        ``not report.acceptable`` without special-casing the verdict
        labels."""
        return self.verdict == "consistent"

    def to_dict(self) -> dict:
        return {
            "realized_value": self.realized_value,
            "baseline_median": self.baseline_median,
            "baseline_p5": self.baseline_p5,
            "baseline_p95": self.baseline_p95,
            "baseline_min": self.baseline_min,
            "baseline_max": self.baseline_max,
            "verdict": self.verdict,
            "drift_score": self.drift_score,
            "empirical_cdf": self.empirical_cdf,
            "higher_is_better": self.higher_is_better,
            "n_baseline_paths": self.n_baseline_paths,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        """One plain-English sentence — joined notes, prefixed with the
        verdict label."""
        label = {
            "consistent": "✓ Consistent",
            "drifting": "⚠ Drifting",
            "out_of_distribution": "✗ Out-of-distribution",
        }.get(self.verdict, self.verdict)
        body = " ".join(self.notes) if self.notes else (
            f"realized={self.realized_value:+.3f} vs baseline "
            f"median {self.baseline_median:+.3f}."
        )
        return f"{label}: {body}"


def consistency_check(
    realized_value: float,
    baseline: RobustnessReport | Sequence[float] | np.ndarray,
    *,
    higher_is_better: bool | None = None,
    significance_level: float = 0.95,
) -> ConsistencyReport:
    """Compare a realised metric value against a baseline synthetic
    distribution and surface whether reality has drifted.

    Parameters
    ----------
    realized_value
        The customer's now-realised backtest value on the same metric
        the baseline was built on. For example: their live strategy's
        rolling 12-month annualised Sharpe.
    baseline
        Either a :class:`~sablier_flow.RobustnessReport` (the
        pre-deployment report; the synthetic distribution is read
        from ``synthetic_values``) or a raw sequence of synthetic
        values. The raw form lets a customer hand-pick which slice
        of synthetic results to use as the baseline (e.g. the
        ``synthetic_max_values`` from a :class:`FamilyReport`).
    higher_is_better
        Direction of the metric. If ``baseline`` is a RobustnessReport
        this **must match** the baseline's stored direction — passing
        a contradictory value raises ``ValueError``. If ``baseline`` is
        a raw sequence, defaults to ``True``.
    significance_level
        Quantile that defines the ``consistent`` boundary. Default
        0.95 means we use the 5th and 95th percentiles of the baseline
        distribution as the inside-CI edges.

    Returns
    -------
    ConsistencyReport
    """
    if significance_level <= 0.5 or significance_level >= 1.0:
        raise ValueError(
            f"significance_level must be in (0.5, 1.0), got {significance_level}"
        )

    n_baseline_paths: int
    if hasattr(baseline, "synthetic_values"):
        # RobustnessReport — pull the retained values + the direction.
        # mypy can't narrow the union purely from hasattr; cast via Any.
        baseline_any: Any = baseline
        synth_values: np.ndarray = np.asarray(
            baseline_any.synthetic_values, dtype=np.float64
        )
        baseline_higher_is_better = bool(baseline_any.higher_is_better)
        if higher_is_better is None:
            higher_is_better = baseline_higher_is_better
        elif bool(higher_is_better) != baseline_higher_is_better:
            # Silently accepting a contradictory override would flip the
            # customer-facing narrative ("worse than baseline" → "better
            # than baseline") with no audit trail. Refuse loudly.
            raise ValueError(
                f"higher_is_better={higher_is_better} contradicts the "
                f"baseline's stored direction "
                f"(baseline.higher_is_better={baseline_higher_is_better}). "
                "Migration: drop the override and let the baseline drive "
                "direction, or rebuild the baseline with the desired "
                "direction before calling consistency_check."
            )
        n_baseline_paths = int(baseline_any.n_synthetic)
    else:
        synth_values = np.asarray(baseline, dtype=np.float64)
        if higher_is_better is None:
            higher_is_better = True
        n_baseline_paths = int(synth_values.size)

    synth_values = synth_values[np.isfinite(synth_values)]
    if synth_values.size == 0:
        raise ValueError(
            "baseline has no finite synthetic values — cannot compute drift"
        )

    # Use 5/95 for the canonical 95% CI (matches RobustnessReport).
    # For other significance levels, take symmetric two-sided quantiles.
    if abs(significance_level - 0.95) < 1e-9:
        lower_pct, upper_pct = 0.05, 0.95
    else:
        tail = (1.0 - significance_level) / 2
        lower_pct, upper_pct = tail, 1.0 - tail
    baseline_p5 = float(np.quantile(synth_values, lower_pct))
    baseline_p95 = float(np.quantile(synth_values, upper_pct))

    baseline_median = float(np.median(synth_values))
    baseline_min = float(np.min(synth_values))
    baseline_max = float(np.max(synth_values))

    # Empirical CDF — the fraction of baseline values below the realised.
    empirical_cdf = float(np.mean(synth_values < realized_value))

    # Verdict — geometric: where does the realised value sit relative to
    # the [p5, p95] CI and [min, max] envelope?
    if baseline_p5 <= realized_value <= baseline_p95:
        verdict: ConsistencyVerdict = "consistent"
    elif baseline_min <= realized_value <= baseline_max:
        verdict = "drifting"
    else:
        verdict = "out_of_distribution"

    ci_half_width = max(0.5 * (baseline_p95 - baseline_p5), 1e-12)
    drift_score = float((realized_value - baseline_median) / ci_half_width)

    notes: list[str] = []
    direction = "above" if realized_value >= baseline_median else "below"
    if verdict == "consistent":
        notes.append(
            f"realized={realized_value:+.3f} sits inside the 95% CI "
            f"[{baseline_p5:+.3f}, {baseline_p95:+.3f}] of the baseline "
            f"synthetic distribution."
        )
    elif verdict == "drifting":
        worse = (
            (higher_is_better and realized_value < baseline_p5)
            or (not higher_is_better and realized_value > baseline_p95)
        )
        worse_word = "worse than" if worse else "better than"
        notes.append(
            f"realized={realized_value:+.3f} is {direction} the baseline 95% CI "
            f"[{baseline_p5:+.3f}, {baseline_p95:+.3f}] but still inside the "
            f"observed envelope [{baseline_min:+.3f}, {baseline_max:+.3f}]. "
            f"Strategy is performing {worse_word} the baseline median "
            f"({baseline_median:+.3f})."
        )
        if worse:
            notes.append(
                "If this persists across the next monitoring window, "
                "consider re-training the flow model on a more recent "
                "history slice."
            )
    else:  # out_of_distribution
        notes.append(
            f"realized={realized_value:+.3f} is outside the baseline "
            f"envelope [{baseline_min:+.3f}, {baseline_max:+.3f}] — the "
            f"regime has shifted since baseline."
        )
        notes.append(
            "Retrain the flow model on a window that includes the "
            "post-shift regime before relying on the verdict again."
        )

    return ConsistencyReport(
        realized_value=float(realized_value),
        baseline_median=baseline_median,
        baseline_p5=baseline_p5,
        baseline_p95=baseline_p95,
        baseline_min=baseline_min,
        baseline_max=baseline_max,
        verdict=verdict,
        drift_score=drift_score,
        empirical_cdf=empirical_cdf,
        higher_is_better=bool(higher_is_better),
        n_baseline_paths=n_baseline_paths,
        notes=notes,
    )
