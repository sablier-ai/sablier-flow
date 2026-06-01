"""Customer integration: backtrader + @sablier_flow.augment against the
LIVE alpha endpoint.

This is the file we hand a fund evaluator. It is intentionally short:
the whole sablier-flow integration is the ``@augment`` decorator and
two env vars. Everything else is the customer's existing backtrader
code, unchanged.

    .venv/bin/python examples/try_it_with_backtrader_alpha.py

Prerequisite — one-time alpha config (also documented in
docs/alpha-onboarding.md):

    gcloud storage cp gs://sablier-flow-demo-results/alpha-endpoint.url ./endpoint.url
    gcloud storage cp gs://sablier-flow-demo-results/alpha-server.crt   ./alpha.crt
    export SABLIER_FLOW_API_KEY=sk-dev
    export SABLIER_FLOW_ENDPOINT=$(cat ./endpoint.url)
    export SABLIER_FLOW_CERT=$(realpath ./alpha.crt)
    export SABLIER_FLOW_ATTESTATION_MODE=fake-for-dev
    export SABLIER_FLOW_PINNED_IMAGE_DIGEST=sha256:$(printf '0%.0s' {1..64})

This script auto-fetches the alpha config from GCS on first run so a
new evaluator can just `python examples/try_it_with_backtrader_alpha.py`
and see the verdict.
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


# -----------------------------------------------------------------------------
# Section 1 — alpha config bootstrap (one-time)
# -----------------------------------------------------------------------------

def bootstrap_alpha_env() -> None:
    """Populate SABLIER_FLOW_* env vars from GCS if they aren't already set."""
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
# Section 2 — the customer's existing backtrader strategy
# -----------------------------------------------------------------------------

class MovingAverageCrossover(bt.Strategy):
    """Long when 10d MA > 30d MA, flat otherwise. Single-asset SPY."""

    params = (("fast", 10), ("slow", 30))

    def __init__(self) -> None:
        d = self.datas[0]
        self.sma_fast = bt.indicators.SimpleMovingAverage(d.close, period=self.p.fast)
        self.sma_slow = bt.indicators.SimpleMovingAverage(d.close, period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self) -> None:
        if self.crossover > 0 and not self.position:
            self.order_target_percent(target=0.99)
        elif self.crossover < 0 and self.position:
            self.order_target_percent(target=0.0)


def _to_ohlc(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": 1_000_000, "openinterest": 0},
        index=close.index,
    )


# -----------------------------------------------------------------------------
# Section 3 — THE 8-LINE INTEGRATION
# -----------------------------------------------------------------------------
# Customer wraps their existing backtest function with @augment. That's it.
# `@augment` decorator handles:
#   1. running the wrapped function on the real input
#   2. calling alternative_versions against sablier-flow alpha
#   3. running the wrapped function on each synthetic alternative
#   4. computing the robustness verdict
#   5. exposing an HTML report
#
# The wrapped function is run verbatim — no inspection, no monkey-patching.
# The customer's backtrader code is untouched.

@sablier_flow.augment(
    n_paths=100,
    horizon=252,
    seed=42,
    features=["SPY", "QQQ", "IWM", "TLT"],
    primary_metric="sharpe",
    progress=True,
)
def my_backtest(prices: pd.DataFrame) -> dict[str, float]:
    """Customer's existing single-asset MA crossover backtest on SPY.

    Note: sablier-flow generates synthetic versions of the **whole**
    universe (SPY/QQQ/IWM/TLT) — the customer's backtest is free to
    use any subset of the columns. Here we only use the SPY column.
    """
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=_to_ohlc(prices["SPY"]), name="SPY"))
    cerebro.addstrategy(MovingAverageCrossover)
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0005)
    # NOTE: backtrader's default SharpeRatio analyzer uses timeframe=Years
    # and returns None on windows shorter than ~2 years. Set timeframe=Days
    # with annualize=True for the meaningful daily-resolution Sharpe.
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        timeframe=bt.TimeFrame.Days, riskfreerate=0.0, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    strat = cerebro.run()[0]
    final = float(cerebro.broker.getvalue())
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio") or 0.0
    dd = strat.analyzers.dd.get_analysis().max.drawdown
    return {
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(dd),
        "total_return": float(final / 100_000.0 - 1.0),
    }


# -----------------------------------------------------------------------------
# Section 4 — the workflow
# -----------------------------------------------------------------------------

def main() -> None:
    bootstrap_alpha_env()
    print("=" * 70)
    print("sablier-flow + backtrader — overfit detection on the LIVE alpha")
    print(f"sablier-flow {sablier_flow.__version__} | backtrader {bt.__version__}")
    print(f"endpoint: {os.environ['SABLIER_FLOW_ENDPOINT']}")
    print("=" * 70)

    tickers = ["SPY", "QQQ", "IWM", "TLT"]
    print("\n[1/2] Downloading universe (yfinance)...")
    df = yf.download(tickers, start="2010-01-01", end="2024-01-01",
                     progress=False, auto_adjust=False)["Adj Close"].dropna().sort_index()
    train_df = df.loc[df.index < "2023-01-01"]
    print(f"     {len(train_df)} business days x {df.shape[1]} tickers (train window)")

    # ----- The whole sablier-flow integration is the single call below.
    print("\n[2/2] Calling decorated backtest — generates synth + runs each + computes verdict...")
    report = my_backtest(train_df)

    # ----- Plain-English headline (item 1) ------------------------------
    print()
    print("=" * 70)
    print("  PLAIN-ENGLISH VERDICT")
    print("=" * 70)
    print(f"  {report.summary()}")

    # ----- Generation-diagnostic warnings (items 2+5) -------------------
    # If memorization is high or structural validation failed, the verdict
    # is on shaky synthetic data. Flag it.
    diag = report.generation_diagnostics
    if diag:
        print()
        print("  Synthetic-data trust signals (from the TEE response):")
        print(f"    memorization_risk:    {diag.get('memorization_risk', '?')}  "
              f"(NN-ratio {diag.get('memorization_nn_distance_ratio', 0.0):.2f})")
        print(f"    validation_overall:   {diag.get('validation_overall', '?')}")

    print()
    print("=" * 70)
    rb = report.robustness
    print(f"  VERDICT: {rb.verdict.upper().replace('_', ' ')}")
    print(f"  Overfit score: {rb.overfit_score:.0%}")
    print("=" * 70)
    print(f"  Real backtest Sharpe:        {report.original['sharpe']:+.3f}")
    print(f"  Synthetic median Sharpe:     {rb.synthetic_median:+.3f}")
    print(f"  Synthetic 95% CI:            [{rb.synthetic_p5:+.3f}, {rb.synthetic_p95:+.3f}]")
    print(f"  N synthetic paths:           {report.n_paths}")
    if report.failures:
        print(f"  Synthetic failures (dropped): {len(report.failures)}")
    for note in rb.notes:
        print(f"   * {note}")

    # ----- DSR side-by-side -------------------------
    # Realistic null vs analytical Bailey-LdP null. The realistic null
    # uses the empirical distribution of synthetic Sharpes — regime-aware.
    # The analytical null uses the IID-Gaussian closed-form — regime-blind.
    dsr = rb.deflated_sharpe(n_trials=1)
    print()
    print("  Deflated Sharpe Ratio:")
    print(f"    realistic null  (Sablier):       {dsr.realistic:.3f}")
    print(f"    analytical null (Bailey-LdP IID):  {dsr.analytical:.3f}")
    print(f"    E[max SR_n] realistic:             {dsr.expected_max_sr_realistic:+.3f}")
    print(f"    E[max SR_n] analytical (N=1):      {dsr.expected_max_sr_analytical:+.3f}")
    print("    SR threshold for DSR=0.95:")
    print(f"      under realistic null:            {dsr.threshold_sr_realistic:+.3f}")
    print(f"      under analytical null:           {dsr.threshold_sr_analytical:+.3f}")

    # ----- Live-drift example (item 3) ---------------------------------
    # Demonstrate the post-deployment monitoring API: pretend the
    # strategy has been live for a quarter and a now-realised Sharpe of
    # +0.20 has come in. consistency_check tells us whether that's
    # consistent with the pre-deployment synthetic distribution.
    realized_sharpe = 0.20
    drift = sablier_flow.consistency_check(realized_sharpe, baseline=rb)
    print()
    print("  Live-drift example (post-deployment monitoring):")
    print(f"    {drift.summary()}")
    print(f"    drift_score = {drift.drift_score:+.2f}  "
          f"(realised in CDF position {drift.empirical_cdf:.0%})")

    html_path = OUT_DIR / "backtrader_alpha_report.html"
    report.to_html(str(html_path), title="SPY MA Crossover — alpha audit")
    print(f"\n  HTML report saved: {html_path}")


if __name__ == "__main__":
    main()
