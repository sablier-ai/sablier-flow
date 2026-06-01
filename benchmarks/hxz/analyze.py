"""Analyze the HXZ study output — produce the Gate 2 verdict.

Reads the JSON produced by `run_study.py` and computes:
  - Spearman correlation between S_OOS_synth and S_OOS_real
  - Median absolute Sharpe error
  - Overfit-catch rate (for strongly-overfit anomalies)
  - Memorization risk distribution

Prints a verdict + writes a publication-ready summary table.

Pass criteria (Gate 2):
  Spearman corr >= 0.75
  Median absolute Sharpe error <= 0.15
  Overfit-catch rate (when S_IS - S_OOS_real > 0.5, S_IS - S_OOS_synth > 0.3) >= 80%
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-not-found]

PASS_CRITERIA = {
    "spearman_min": 0.75,
    "median_abs_error_max": 0.15,
    "overfit_catch_min": 0.80,
    "strong_overfit_threshold": 0.5,   # |S_IS - S_OOS_real|
    "synth_overfit_threshold": 0.3,    # |S_IS - S_OOS_synth| we expect for catch
}


def analyze(results_path: Path) -> dict:
    """Load the study output, compute the verdict."""
    data = json.loads(results_path.read_text())
    df = pd.DataFrame(data["results"])
    if df.empty:
        raise SystemExit("No results to analyze.")

    # ----- Predictive correlation -----
    spearman_r, spearman_p = stats.spearmanr(df["S_OOS_synth_mean"], df["S_OOS_real"])

    # ----- Calibration: median absolute error -----
    abs_err = (df["S_OOS_synth_mean"] - df["S_OOS_real"]).abs()
    median_abs_err = float(abs_err.median())

    # ----- Overfit-catch rate -----
    is_decay = df["S_IS"] - df["S_OOS_real"]
    synth_decay = df["S_IS"] - df["S_OOS_synth_mean"]
    strongly_overfit = is_decay > PASS_CRITERIA["strong_overfit_threshold"]
    caught = (synth_decay > PASS_CRITERIA["synth_overfit_threshold"]) & strongly_overfit
    overfit_catch_rate = (
        float(caught.sum() / max(strongly_overfit.sum(), 1)) if strongly_overfit.any() else float("nan")
    )

    # ----- Memorization distribution -----
    mem_counts = Counter(df["memorization_risk"].tolist())

    # ----- Verdict -----
    passed = (
        spearman_r >= PASS_CRITERIA["spearman_min"]
        and median_abs_err <= PASS_CRITERIA["median_abs_error_max"]
        and (np.isnan(overfit_catch_rate) or overfit_catch_rate >= PASS_CRITERIA["overfit_catch_min"])
    )

    verdict = {
        "passed_gate_2": bool(passed),
        "spearman_corr": float(spearman_r),
        "spearman_p_value": float(spearman_p),
        "median_abs_sharpe_error": float(median_abs_err),
        "overfit_catch_rate": float(overfit_catch_rate),
        "n_anomalies": len(df),
        "n_strongly_overfit": int(strongly_overfit.sum()),
        "memorization_distribution": dict(mem_counts),
        "pass_criteria": PASS_CRITERIA,
    }
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HXZ study results + Gate 2 verdict")
    parser.add_argument("results", type=Path, help="Path to JSON produced by run_study.py")
    parser.add_argument("--summary", type=Path, default=Path("gate_2_verdict.json"))
    args = parser.parse_args()

    verdict = analyze(args.results)

    print("=" * 60)
    print(f"  GATE 2 VERDICT: {'✅ PASS' if verdict['passed_gate_2'] else '❌ FAIL'}")
    print("=" * 60)
    print(f"  Spearman correlation        : {verdict['spearman_corr']:.3f}  (need ≥ {PASS_CRITERIA['spearman_min']})")
    print(f"  Median absolute Sharpe err  : {verdict['median_abs_sharpe_error']:.3f}  (need ≤ {PASS_CRITERIA['median_abs_error_max']})")
    print(f"  Overfit-catch rate          : {verdict['overfit_catch_rate']:.2%}  (need ≥ {PASS_CRITERIA['overfit_catch_min']:.0%})")
    print(f"  N anomalies                 : {verdict['n_anomalies']}")
    print(f"  N strongly-overfit          : {verdict['n_strongly_overfit']}")
    print(f"  Memorization distribution   : {verdict['memorization_distribution']}")

    args.summary.write_text(json.dumps(verdict, indent=2))
    print(f"\nFull verdict written to {args.summary}")


if __name__ == "__main__":
    main()
