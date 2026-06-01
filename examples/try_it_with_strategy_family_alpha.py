"""Strategy-family overfit audit against the LIVE alpha endpoint.

This is the example for the *family-of-strategies* case — the regime
where the regime-aware DSR + PBO contribution shines. We evaluate
11 MA-crossover variants (different lookback combinations) on real
SPY history + on 100 synthetic alternative histories, then ask:

  - What's the family DSR (deflated Sharpe of the family-best strategy)
    under the Sablier realistic null vs the analytical Bailey-LdP null?
  - What's the Probability of Backtest Overfitting (PBO via CSCV) on
    the real history alone?

Run::

    .venv/bin/python examples/try_it_with_strategy_family_alpha.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import backtrader as bt
import pandas as pd
import yfinance as yf

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import sablier_flow

OUT_DIR = _REPO_ROOT / "examples" / "alpha_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bootstrap_alpha_env() -> None:
    if os.environ.get("SABLIER_FLOW_ENDPOINT") and os.environ.get("SABLIER_FLOW_CERT"):
        return
    cert_path = OUT_DIR / "alpha-server.crt"
    url_path = OUT_DIR / "alpha-endpoint.url"
    print("[setup] fetching alpha endpoint + pinned cert from GCS...")
    subprocess.run(
        ["gcloud", "storage", "cp",
         "gs://sablier-flow-demo-results/alpha-server.crt", str(cert_path)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["gcloud", "storage", "cp",
         "gs://sablier-flow-demo-results/alpha-endpoint.url", str(url_path)],
        check=True, capture_output=True,
    )
    os.environ.setdefault("SABLIER_FLOW_API_KEY", "sk-dev")
    os.environ.setdefault("SABLIER_FLOW_ENDPOINT", url_path.read_text().strip())
    os.environ.setdefault("SABLIER_FLOW_CERT", str(cert_path))
    os.environ.setdefault("SABLIER_FLOW_ATTESTATION_MODE", "fake-for-dev")
    os.environ.setdefault(
        "SABLIER_FLOW_PINNED_IMAGE_DIGEST", "sha256:" + "0" * 64
    )


# -----------------------------------------------------------------------------
# Single backtrader strategy — used by every member of the family
# -----------------------------------------------------------------------------


class MovingAverageCrossover(bt.Strategy):
    params = (("fast", 10), ("slow", 30))

    def __init__(self) -> None:
        d = self.datas[0]
        self.f = bt.indicators.SimpleMovingAverage(d.close, period=self.p.fast)
        self.s = bt.indicators.SimpleMovingAverage(d.close, period=self.p.slow)
        self.x = bt.indicators.CrossOver(self.f, self.s)

    def next(self) -> None:
        if self.x > 0 and not self.position:
            self.order_target_percent(target=0.99)
        elif self.x < 0 and self.position:
            self.order_target_percent(target=0.0)


def _to_ohlc(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": 1_000_000, "openinterest": 0},
        index=close.index,
    )


def _run_one(prices: pd.DataFrame, *, fast: int, slow: int) -> dict[str, float]:
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=_to_ohlc(prices["SPY"]), name="SPY"))
    cerebro.addstrategy(MovingAverageCrossover, fast=fast, slow=slow)
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0005)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        timeframe=bt.TimeFrame.Days, riskfreerate=0.0, annualize=True)
    strat = cerebro.run()[0]
    final = float(cerebro.broker.getvalue())
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio") or 0.0
    return {"sharpe": float(sharpe), "total_return": float(final / 100_000.0 - 1.0)}


# -----------------------------------------------------------------------------
# The family — same engine, multiple parameter combinations
# -----------------------------------------------------------------------------
# Late-binding fix is the lambda f=fast, s=slow=... trick.
_PARAM_GRID: list[tuple[int, int]] = [
    (5, 20), (10, 30), (15, 45), (20, 60),
    (25, 75), (30, 90), (40, 120), (50, 150),
    (5, 50), (10, 100), (20, 200),
]

STRATEGIES: dict[str, object] = {
    f"ma_{f}_{s}": (lambda f=f, s=s: lambda df: _run_one(df, fast=f, slow=s))()
    for f, s in _PARAM_GRID
}


def main() -> None:
    bootstrap_alpha_env()
    print("=" * 72)
    print("sablier-flow + backtrader — STRATEGY-FAMILY overfit audit (live alpha)")
    print(f"sablier-flow {sablier_flow.__version__} | backtrader {bt.__version__}")
    print(f"endpoint: {os.environ['SABLIER_FLOW_ENDPOINT']}")
    print(f"family: {len(STRATEGIES)} MA-crossover variants")
    print("=" * 72)

    tickers = ["SPY", "QQQ", "IWM", "TLT"]
    print("\n[1/2] Downloading universe (yfinance)...")
    df = yf.download(tickers, start="2010-01-01", end="2024-01-01",
                     progress=False, auto_adjust=False)["Adj Close"].dropna().sort_index()
    train_df = df.loc[df.index < "2023-01-01"]
    print(f"     {len(train_df)} business days x {df.shape[1]} tickers (train window)")

    print("\n[2/2] Evaluating family on real + 50 synthetic alt-histories...")
    print("       (synth gen + family backtest + PBO via CSCV, ~5-8 min total)")
    # n_paths=50 + S=8 keeps the example runnable in <10min on a laptop.
    # Production audits should use 100+ paths and S>=16 for tighter
    # estimates.
    report = sablier_flow.evaluate_family(
        STRATEGIES,
        train_df,
        n_paths=50,
        horizon=252,
        seed=42,
        features=tickers,
        primary_metric="sharpe",
        higher_is_better=True,
        pbo_cscv_splits=8,
        progress=True,
    )

    print()
    print("=" * 72)
    print("  STRATEGY-FAMILY OVERFIT AUDIT")
    print("=" * 72)
    # Plain-English headline (item 1)
    print(f"  {report.summary()}")
    print()
    dsr = report.deflated_sharpe
    print(f"  Family size:                              {len(STRATEGIES)} variants")
    print(f"  Best in-sample strategy:                  {report.real_argmax_strategy}")
    print(f"  Best in-sample Sharpe (real):             {report.real_max_value:+.3f}")
    print()
    print("  Realistic null (Sablier):")
    print(f"    E[max_n SR_n] across synth alt-paths:   {dsr.expected_max_sr_realistic:+.3f}")
    print(f"    SR threshold for DSR=0.95:              {dsr.threshold_sr_realistic:+.3f}")
    print(f"    DSR (real best vs realistic null):      {dsr.realistic:.3f}")
    print()
    print("  Analytical Bailey-LdP IID-Gaussian null:")
    print(f"    E[max_n SR_n] closed-form (N={dsr.n_trials}):       {dsr.expected_max_sr_analytical:+.3f}")
    print(f"    SR threshold for DSR=0.95:              {dsr.threshold_sr_analytical:+.3f}")
    print(f"    DSR (real best vs analytical null):     {dsr.analytical:.3f}")
    print()
    print(f"  Probability of Backtest Overfitting (CSCV, S={report.pbo_cscv_splits}):")
    print(f"    PBO:                                    {report.pbo:.3f}")
    print(f"    Partitions evaluated:                   {report.pbo_n_partitions}")
    print("    (0 = best in-sample also wins OOS; 0.5 = no signal; 1 = always loses)")
    print()
    if report.failures:
        print(f"  {len(report.failures)} synthetic backtests failed (dropped). "
              "See report.failures.")
    for n in report.notes:
        print(f"   * {n}")

    # Per-strategy real Sharpe table
    print("\n  Per-strategy real Sharpe (sorted by overfit score, item 4):")
    print(f"    {'strategy':<14s}  {'real_sharpe':>11s}  {'synth_median':>12s}  {'overfit_score':>13s}")
    pairs = sorted(
        report.strategy_names,
        key=lambda nm: report.per_strategy_overfit_score.get(nm, 0.0),
        reverse=True,
    )
    for name in pairs:
        real_sr = report.per_strategy_real_metric[name]
        synth_med = report.per_strategy_synthetic_median[name]
        of = report.per_strategy_overfit_score[name]
        print(f"    {name:<14s}  {real_sr:>+11.3f}  {synth_med:>+12.3f}  {of:>13.0%}")

    # Strategy attribution: which variants are pulling the family overfit
    # signal (item 4)?
    top = report.most_overfit_variants(top=3)
    if top:
        print("\n  Top variants pulling the family overfit signal:")
        for name, score in top:
            print(f"    {name}  overfit_score={score:.0%}")

    print()
    if dsr.realistic >= 0.95:
        print("  VERDICT (realistic null): best strategy clears DSR=0.95 ✓")
    elif dsr.realistic >= 0.50:
        print("  VERDICT (realistic null): best strategy beats most synth alt-histories")
    else:
        print("  VERDICT (realistic null): best strategy looks like noise vs alt-histories ✗")

    if dsr.analytical >= 0.95 and dsr.realistic < 0.95:
        print("  NOTE: analytical null says clear, realistic null says not — this is")
        print("        exactly the regime-aware gap the realistic DSR is designed to expose.")


if __name__ == "__main__":
    main()
