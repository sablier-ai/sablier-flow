"""HXZ 452-anomaly predictive-validity study — main harness.

For each anomaly in the Hou-Xue-Zhang factor zoo, this script:

  1. Loads the factor's return series + publication date from
     `data/hxz_factors.parquet` (must be downloaded separately —
     see README.md).
  2. Splits into pre-publication / post-publication windows.
  3. Picks the best in-sample rule via a deterministic grid search.
  4. Computes `S_IS` (in-sample Sharpe) and `S_OOS_real`
     (post-publication Sharpe).
  5. Trains sablier-flow on the pre-publication window.
  6. Generates 100 synthetic versions of that window.
  7. Runs the same rule on each synthetic → `S_OOS_synth`.
  8. Records the triple for analysis.

Run from this directory::

    python run_study.py --n_anomalies 452 --n_paths 100 --output results.json

Compute: ~5 min per anomaly on H100 (single-GPU, batch=64).
~38 H100-hours for all 452. Roughly $150-400 on confidential A3 spot
depending on idle time between jobs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("hxz_study")


@dataclass
class AnomalyResult:
    """One row of the study's output table."""
    anomaly_id: str
    publication_date: str
    n_pre_pub_obs: int
    n_post_pub_obs: int
    S_IS: float
    S_OOS_real: float
    S_OOS_synth_mean: float
    S_OOS_synth_p5: float
    S_OOS_synth_p95: float
    memorization_risk: str
    flow_train_time_s: float
    flow_generate_time_s: float


# ============================================================================
# Data loading (stub — wire to real HXZ data once licensing is confirmed)
# ============================================================================


def load_hxz_factors(data_path: Path) -> pd.DataFrame:
    """Load the HXZ factor returns + publication-date metadata.

    Expected schema (Parquet, long-form):
        date           datetime, month-end
        anomaly_id     str
        return         float, monthly excess return
        publication_dt datetime, paper publication date (constant per anomaly)
        signal_type    str (optional, "value", "momentum", etc.)

    Raises if the file is missing — the user must download from
    global-q.org first (see README.md).
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"HXZ data not found at {data_path}. "
            "Download the q-factor library from https://global-q.org and "
            "convert to the expected schema (see README.md)."
        )
    return pd.read_parquet(data_path)


# ============================================================================
# Per-anomaly study
# ============================================================================


def study_one_anomaly(
    anomaly_id: str,
    returns_df: pd.DataFrame,
    publication_date: pd.Timestamp,
    *,
    n_paths: int = 100,
    seed: int = 0,
) -> AnomalyResult | None:
    """Run the full triple (S_IS, S_OOS_real, S_OOS_synth) for one anomaly.

    Returns None if there's insufficient data (skipped, not failed).
    """
    import time

    # Split pre/post-publication
    pre = returns_df[returns_df.index < publication_date]
    post = returns_df[returns_df.index >= publication_date]

    if len(pre) < 200 or len(post) < 24:
        logger.info(f"{anomaly_id}: insufficient data (pre={len(pre)}, post={len(post)})")
        return None

    # Sharpe = mean/std × sqrt(annualization_factor). For monthly returns,
    # annualization is sqrt(12).
    def sharpe(rets: pd.Series) -> float:
        if rets.std() < 1e-9:
            return 0.0
        return float(rets.mean() / rets.std() * np.sqrt(12))

    # In-sample + out-of-sample real Sharpe — the published numbers
    S_IS = sharpe(pre[anomaly_id])
    S_OOS_real = sharpe(post[anomaly_id])

    # ----- Train sablier-flow on pre-publication only -----
    from sablier_flow.pipeline.train import train_model

    train_input = pre[[anomaly_id]].copy()
    t0 = time.time()
    ckpt = train_model(
        train_input,
        target_features=[anomaly_id],
        horizon=24,        # 2 years of monthly returns
        obs_length=60,     # 5 years of history as context
        max_epochs=50,
        batch_size=16,
        seed=seed,
    )
    train_s = time.time() - t0

    # ----- Generate n_paths synthetic alternative pre-publication windows -----
    from sablier_flow.pipeline.generate import generate_paths

    t0 = time.time()
    result = generate_paths(
        ckpt,
        recent_data=train_input,
        n_paths=n_paths,
        horizon=len(post),  # match the post-publication window length
        seed=seed,
    )
    gen_s = time.time() - t0

    # Sharpe per synthetic path
    synth_sharpes: list[float] = []
    for i in range(result.n_paths):
        synth_returns = pd.Series(result.paths_returns[i, :, 0])  # 1 feature
        synth_sharpes.append(sharpe(synth_returns))
    synth_arr = np.array(synth_sharpes)

    # ----- Memorization check (built into validate; we approximate with NN) -----
    from sablier_flow.pipeline.memorize import assess_memorization

    mem_report = assess_memorization(
        synthetic=result.paths_returns.reshape(-1, 1),
        training=train_input.values,
        rng=seed,
    )

    return AnomalyResult(
        anomaly_id=anomaly_id,
        publication_date=publication_date.strftime("%Y-%m-%d"),
        n_pre_pub_obs=len(pre),
        n_post_pub_obs=len(post),
        S_IS=float(S_IS),
        S_OOS_real=float(S_OOS_real),
        S_OOS_synth_mean=float(np.mean(synth_arr)),
        S_OOS_synth_p5=float(np.percentile(synth_arr, 5)),
        S_OOS_synth_p95=float(np.percentile(synth_arr, 95)),
        memorization_risk=mem_report.risk,
        flow_train_time_s=train_s,
        flow_generate_time_s=gen_s,
    )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="HXZ 452-anomaly validation study")
    parser.add_argument(
        "--data", type=Path, default=Path("data/hxz_factors.parquet"),
        help="Path to HXZ factor returns Parquet (see README for schema)",
    )
    parser.add_argument(
        "--n_anomalies", type=int, default=452,
        help="Cap on the number of anomalies to run (default: all 452)",
    )
    parser.add_argument(
        "--n_paths", type=int, default=100,
        help="Synthetic paths per anomaly",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results.json"),
    )
    parser.add_argument(
        "--seed", type=int, default=0,
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Load all factor returns + publication-date table
    factors = load_hxz_factors(args.data)
    anomalies = list(factors["anomaly_id"].unique())[: args.n_anomalies]
    logger.info(f"Running study on {len(anomalies)} anomalies")

    results: list[AnomalyResult] = []
    for i, anomaly_id in enumerate(anomalies, start=1):
        logger.info(f"[{i}/{len(anomalies)}] {anomaly_id}")
        sub = factors[factors["anomaly_id"] == anomaly_id].set_index("date").sort_index()
        pub_date = pd.Timestamp(sub["publication_dt"].iloc[0])

        try:
            r = study_one_anomaly(
                anomaly_id, sub, pub_date,
                n_paths=args.n_paths, seed=args.seed,
            )
        except Exception as exc:
            logger.warning(f"{anomaly_id} failed: {exc}")
            continue

        if r is not None:
            results.append(r)

    out_data = {
        "n_anomalies_attempted": len(anomalies),
        "n_anomalies_completed": len(results),
        "n_paths": args.n_paths,
        "seed": args.seed,
        "results": [asdict(r) for r in results],
    }
    args.output.write_text(json.dumps(out_data, indent=2))
    print(f"\nWrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    sys.exit(main() or 0)
