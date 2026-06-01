"""Deflated Sharpe Ratio (DSR) under the Sablier realistic null
alongside the analytical Bailey-López-de-Prado IID-Gaussian null.

The analytical DSR formula comes from Bailey and López de Prado
(2014); the realistic-null version replaces ``E[max_n SR_n]`` under
IID-Gaussian with the empirical statistics from running the strategy
on synthetic alternative-history paths.

The realistic null is regime-aware: ``E[max_n SR_n]`` under the
realistic null shifts substantially across calm vs post-vol-spike
training contexts, while the analytical IID-Gaussian prediction
stays constant. The DSR derived from the realistic null inherits
this regime-conditioning; the analytical version does not.

Usage::

    from sablier_flow.analytics.deflated_sharpe import deflated_sharpe

    dsr = deflated_sharpe(
        observed_sr=1.5,
        synthetic_sharpes=[...],         # one Sharpe per synthetic alt-history
        strategy_returns=daily_returns,  # for skew/kurt correction; optional
        n_trials=1,                      # 1 for a single backtest; M for a family
    )
    print(dsr.realistic, dsr.analytical)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "DeflatedSharpeReport",
    "deflated_sharpe",
    "expected_max_sr_iid_gaussian",
]


_EULER_MASCHERONI = 0.5772156649015329

# Module-level "warned once" guard so callers in a tight loop don't get
# spammed. Resets on interpreter restart.
_NULL_SOURCE_WARNED = False


def expected_max_sr_iid_gaussian(n_trials: int) -> float:
    """Bailey-López-de-Prado (2014) approximation of ``E[max_n SR_n]`` under
    an IID-Gaussian null.

    Closed form::

        E[max_n] ≈ (1 - γ) · Φ⁻¹(1 - 1/N) + γ · Φ⁻¹(1 - 1/(N·e))

    where γ is the Euler-Mascheroni constant. The variance ``V[SR_n]``
    under the IID-Gaussian null is implicitly 1 in this expression; the
    full DSR scales by the Sharpe estimator variance separately.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_trials == 1:
        return 0.0
    from scipy.stats import norm
    return float(
        (1.0 - _EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
        + _EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    )


@dataclass(frozen=True)
class DeflatedSharpeReport:
    """DSR comparison between realistic and analytical nulls.

    Both nulls answer "what's the probability the observed Sharpe is
    real rather than the best-of-N from chance" — they differ in the
    distribution of best-of-N they assume.

    - **realistic**: empirical CDF of the observed Sharpe in the
      distribution of ``max_n SR_n`` under the Sablier synthetic
      alternative-history null. Regime-aware. This is the headline
      number Sablier puts forward.
    - **analytical**: closed-form Bailey-López-de-Prado (2014) DSR
      under an IID-Gaussian null with skew/kurtosis correction (if
      strategy returns provided). Regime-blind. Included for
      side-by-side comparison.

    Interpretation: higher = stronger evidence the strategy isn't noise.
    The standard significance threshold is DSR ≥ 0.95.
    """

    observed_sr: float
    """The observed Sharpe ratio (or whichever metric was the primary)."""

    n_trials: int
    """Number of strategies searched. 1 for a single backtest, M for an
    M-strategy family. The analytical null inflates ``E[max]`` with N;
    the realistic null uses the empirical best-of-N from synthetic paths."""

    realistic: float
    """DSR under the Sablier realistic null — empirical
    P(synthetic best-of-N Sharpe < observed). In [0, 1]."""

    analytical: float
    """DSR under the Bailey-LdP IID-Gaussian null — closed-form
    Φ((SR_obs - E[max])/√Var(SR)). In [0, 1]."""

    expected_max_sr_realistic: float
    """Mean of the synthetic best-of-N Sharpe distribution. Varies
    substantially across regime folds (e.g. calm vs post-vol-spike)
    even though the analytical IID-Gaussian reference is constant —
    the realistic null is the one that reflects the regime the
    customer is actually in."""

    expected_max_sr_analytical: float
    """Bailey-LdP closed-form E[max_n SR_n] under IID-Gaussian."""

    threshold_sr_realistic: float
    """The observed SR a customer would need to hit a realistic-null
    DSR of 0.95 — the empirical 95th percentile of the synthetic
    best-of-N distribution."""

    threshold_sr_analytical: float
    """The observed SR needed for analytical DSR = 0.95 — the
    closed-form Bailey-LdP threshold."""

    @property
    def verdict(self) -> str:
        """Bucketed verdict on the realistic-null DSR. Matches the
        :meth:`FamilyReport.summary` thresholds so the DSR field reads
        the same whether the customer pulls it off this report or off
        the family wrapper:

          - ``'significant'``      — ``realistic >= 0.95`` (clears the
            standard 95% bar under the regime-aware null)
          - ``'defensible'``       — ``0.50 <= realistic < 0.95`` (better
            than median best-of-N but not unambiguous)
          - ``'looks_like_noise'`` — ``realistic < 0.50`` (the real
            metric sits below the synthetic median; random selection on
            alt-histories routinely produces higher numbers)
        """
        if self.realistic >= 0.95:
            return "significant"
        if self.realistic >= 0.50:
            return "defensible"
        return "looks_like_noise"

    @property
    def acceptable(self) -> bool:
        """True when the realistic-null DSR clears the standard 0.95
        significance bar — the same gate fund / CI workflows use to
        accept a backtest. ``'defensible'`` results are intentionally
        not 'acceptable' here: the customer can still ship them with
        eyes open, but the deploy gate should default to strict."""
        return self.realistic >= 0.95

    def to_dict(self) -> dict[str, float | int]:
        return {
            "observed_sr": self.observed_sr,
            "n_trials": self.n_trials,
            "realistic": self.realistic,
            "analytical": self.analytical,
            "expected_max_sr_realistic": self.expected_max_sr_realistic,
            "expected_max_sr_analytical": self.expected_max_sr_analytical,
            "threshold_sr_realistic": self.threshold_sr_realistic,
            "threshold_sr_analytical": self.threshold_sr_analytical,
        }


def deflated_sharpe(
    *,
    observed_sr: float,
    synthetic_sharpes: Sequence[float] | np.ndarray,
    strategy_returns: Sequence[float] | np.ndarray | None = None,
    n_trials: int = 1,
    significance_level: float = 0.95,
) -> DeflatedSharpeReport:
    """Compute DSR under the Sablier realistic null and the Bailey-LdP
    analytical null.

    Parameters
    ----------
    observed_sr
        The Sharpe ratio observed on real data. For a single backtest,
        the strategy's Sharpe; for an M-strategy family, the best of M
        (``max_m SR_m``).
    synthetic_sharpes
        The distribution of synthetic-null Sharpes. For a single
        backtest, one Sharpe per synthetic alternative history. For a
        strategy family, one ``max_m SR_m`` per synthetic alternative
        history. Length N.
    strategy_returns
        The per-period returns of the strategy on the real backtest, if
        available. Used for the skew/kurtosis correction in the
        analytical DSR (Bailey-LdP eq. 5). If omitted, the analytical
        DSR drops the higher-moment correction (γ₃ = 0, γ₄ = 3).
    n_trials
        Number of strategies searched. 1 for a single backtest, M for
        an M-strategy family. Drives both ``E[max]`` calculations.
    significance_level
        Quantile for the threshold-SR fields. Default 0.95 — the SR a
        customer would need to clear DSR = 0.95.

    Returns
    -------
    DeflatedSharpeReport
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    synth = np.asarray(synthetic_sharpes, dtype=np.float64)
    synth = synth[np.isfinite(synth)]
    if synth.size == 0:
        raise ValueError("synthetic_sharpes is empty (or all NaN/inf)")

    # ----- Realistic null --------------------------------------------------
    # Empirical CDF of observed SR in the synthetic best-of-N distribution.
    dsr_realistic = float(np.mean(synth <= observed_sr))
    e_max_realistic = float(np.mean(synth))
    threshold_realistic = float(np.quantile(synth, significance_level))

    # ----- Analytical Bailey-LdP null --------------------------------------
    from scipy.stats import norm

    e_max_analytical = expected_max_sr_iid_gaussian(n_trials)

    # ----- Null-source consistency check -----------------------------------
    # If the realistic null's mean diverges from the analytical E[max_n
    # SR_n] by more than 3 stdev (of synthetic_sharpes / sqrt(N)), the
    # caller has almost certainly passed a distribution that's inconsistent
    # with n_trials — e.g. N(0,1) samples while claiming n_trials=1000. The
    # realistic and analytical DSRs will disagree dramatically and the
    # customer can't tell which to trust.
    global _NULL_SOURCE_WARNED
    if synth.size >= 2 and not _NULL_SOURCE_WARNED:
        synth_std = float(np.std(synth, ddof=1))
        sem = synth_std / np.sqrt(synth.size) if synth_std > 0 else 0.0
        if sem > 0 and abs(e_max_realistic - e_max_analytical) > 3.0 * sem:
            warnings.warn(
                "deflated_sharpe: synthetic_sharpes appears inconsistent "
                f"with n_trials={n_trials}; realistic and analytical nulls "
                "will disagree (realistic E[max]="
                f"{e_max_realistic:.3f} vs analytical E[max]="
                f"{e_max_analytical:.3f}). Pass n_trials matching the "
                f"actual selection process or pass synthetic_sharpes "
                f"matching N={n_trials}.",
                UserWarning,
                stacklevel=2,
            )
            _NULL_SOURCE_WARNED = True

    if strategy_returns is not None:
        from scipy.stats import kurtosis, skew
        rets = np.asarray(strategy_returns, dtype=np.float64)
        rets = rets[np.isfinite(rets)]
        if rets.size < 3:
            gamma3, gamma4 = 0.0, 3.0
            t_obs = max(rets.size, 252)
        else:
            gamma3 = float(skew(rets))
            gamma4 = float(kurtosis(rets, fisher=False))  # raw kurtosis (Normal = 3)
            t_obs = rets.size
    else:
        # No returns provided — drop the higher-moment correction.
        gamma3, gamma4 = 0.0, 3.0
        t_obs = 252  # assume one year of daily observations as a sensible default

    # Bailey-LdP (2014) variance of the Sharpe estimator:
    #   Var(SR) ≈ (1 - γ₃·SR + ((γ₄-1)/4)·SR²) / (T-1)
    var_factor = 1.0 - gamma3 * observed_sr + ((gamma4 - 1.0) / 4.0) * observed_sr * observed_sr
    if var_factor <= 0 or t_obs < 2:
        # Higher-moment correction can drive var_factor negative on
        # extreme inputs; fall back to no-correction variance.
        var_factor = 1.0
    var_sr = var_factor / max(t_obs - 1, 1)
    sd_sr = float(np.sqrt(var_sr))
    if sd_sr <= 0:
        dsr_analytical = float("nan")
        threshold_analytical = float("nan")
    else:
        dsr_analytical = float(norm.cdf((observed_sr - e_max_analytical) / sd_sr))
        threshold_analytical = float(
            e_max_analytical + sd_sr * norm.ppf(significance_level)
        )

    return DeflatedSharpeReport(
        observed_sr=float(observed_sr),
        n_trials=int(n_trials),
        realistic=dsr_realistic,
        analytical=dsr_analytical,
        expected_max_sr_realistic=e_max_realistic,
        expected_max_sr_analytical=e_max_analytical,
        threshold_sr_realistic=threshold_realistic,
        threshold_sr_analytical=threshold_analytical,
    )
