"""sablier-flow + backtrader — production-engine integration walkthrough.

Same shape as try_it_yourself.py, but runs the strategy through
backtrader's Cerebro engine (the most common Python backtest framework
in quant-fund production). Demonstrates the as_backtrader_feeds
adapter end-to-end.

    .venv/bin/python examples/try_it_with_backtrader.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import backtrader as bt
import numpy as np
import pandas as pd
import yfinance as yf

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import sablier_flow
from sablier_flow.adapters.backtrader import as_backtrader_feeds
from sablier_flow.client.client import Client
from sablier_flow.client.crypto import EnvelopeEncrypted
from sablier_flow.client.payload import (
    JobResultPayload,
    JobUploadPayload,
    encrypt_result,
)
from sablier_flow.client.transport import InMemoryTransport
from sablier_flow.types import GenerationResult
from server.tee.attestation import generate_attestation_quote
from server.tee.crypto import TEEKeyState

# -----------------------------------------------------------------------------
# Section 1 — the customer's existing backtrader strategy
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
        # order_target_percent — proper allocation, not the toy self.buy()
        if self.crossover > 0 and not self.position:
            self.order_target_percent(target=0.99)
        elif self.crossover < 0 and self.position:
            self.order_target_percent(target=0.0)


def run_backtest(feed: bt.feeds.PandasData, *, starting_cash: float = 100_000.0) -> dict:
    """Run one backtest in Cerebro, return Sharpe + DD + final equity."""
    cerebro = bt.Cerebro()
    cerebro.adddata(feed)
    cerebro.addstrategy(MovingAverageCrossover)
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.setcommission(commission=0.0005)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        riskfreerate=0.0, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    strats = cerebro.run()
    s = strats[0]
    final_value = float(cerebro.broker.getvalue())
    sharpe = s.analyzers.sharpe.get_analysis().get("sharperatio") or 0.0
    dd = s.analyzers.dd.get_analysis().max.drawdown
    return {
        "final_value": final_value,
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(dd),
        "total_return": float(final_value / starting_cash - 1.0),
    }


# -----------------------------------------------------------------------------
# Section 2 — block-bootstrap mock TEE (same as the other example)
# -----------------------------------------------------------------------------


def _block_bootstrap_paths(
    df: pd.DataFrame,
    target_features: list[str],
    *,
    n_paths: int,
    horizon: int,
    block_size: int = 20,
    seed: int | None = None,
) -> GenerationResult:
    rng = np.random.default_rng(seed)
    rets = df[target_features].pct_change().dropna().values
    T, n_features = rets.shape
    last_prices = df[target_features].iloc[-1].values.astype(np.float32)

    paths_returns = np.empty((n_paths, horizon, n_features), dtype=np.float32)
    paths_prices = np.empty((n_paths, horizon, n_features), dtype=np.float32)

    for p in range(n_paths):
        synthetic_rets = np.empty((horizon, n_features), dtype=np.float32)
        i = 0
        while i < horizon:
            start = rng.integers(0, T - block_size)
            chunk = rets[start:start + block_size]
            n_copy = min(block_size, horizon - i)
            synthetic_rets[i:i + n_copy] = chunk[:n_copy]
            i += n_copy
        paths_returns[p] = synthetic_rets
        paths_prices[p] = last_prices * np.cumprod(1.0 + synthetic_rets, axis=0)

    return GenerationResult(
        paths_returns=paths_returns,
        paths_prices=paths_prices,
        feature_names=list(target_features),
        last_prices=last_prices,
        horizon=horizon,
        n_paths=n_paths,
        seed=seed,
        sdk_version="block-bootstrap-mock",
    )


def _make_fake_tee(real_data, target_features):  # type: ignore[no-untyped-def]
    tee_keys = TEEKeyState()

    def run_job(job):  # type: ignore[no-untyped-def]
        env = EnvelopeEncrypted.from_bytes(job.input_ciphertext)
        upload = JobUploadPayload.from_bytes(tee_keys.decrypt(env))
        result = _block_bootstrap_paths(
            real_data, target_features,
            n_paths=int(upload.params.get("n_paths", 100)),
            horizon=int(upload.params.get("horizon") or 252),
            seed=upload.params.get("seed"),
        )
        return encrypt_result(
            JobResultPayload.from_generation_result(result).to_bytes(),
            upload.result_key,
        )

    def quote_gen(pubkey):  # type: ignore[no-untyped-def]
        return generate_attestation_quote(pubkey, image_digest="sha256:" + "d" * 64)

    return InMemoryTransport(run_job=run_job, quote_generator=quote_gen, tee_keys=tee_keys)


# -----------------------------------------------------------------------------
# Section 3 — the workflow
# -----------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("sablier-flow + backtrader — overfit detection demo")
    print(f"sablier-flow {sablier_flow.__version__} | backtrader {bt.__version__}")
    print("=" * 70)

    target_features = ["SPY", "QQQ", "IWM", "TLT"]
    test_window = 504  # ~2 years of trading days — enough for many trades

    print("\n[1/5] Downloading universe...")
    real_data = yf.download(target_features, start="2015-01-01", end="2024-01-01",
                            progress=False, auto_adjust=False)["Adj Close"]
    real_data = real_data.dropna().sort_index()
    print(f"     {len(real_data)} business days x {real_data.shape[1]} tickers")

    # ----- Run the customer's backtrader strategy on the real tail -----
    print(f"\n[2/5] Running backtrader strategy on the last {test_window} days...")
    real_tail = real_data.tail(test_window)
    real_close = real_tail["SPY"]
    # backtrader needs full OHLCV columns — synthesize from close like
    # sablier_flow.adapters.backtrader does.
    real_ohlc = pd.DataFrame(
        {
            "open": real_close,
            "high": real_close,
            "low": real_close,
            "close": real_close,
            "volume": 1_000_000,
            "openinterest": 0,
        },
        index=real_tail.index,
    )
    real_feed = bt.feeds.PandasData(dataname=real_ohlc, name="SPY")
    real_result = run_backtest(real_feed)
    print("     Real backtest:")
    print(f"       Sharpe:       {real_result['sharpe']:+.3f}")
    print(f"       Total return: {real_result['total_return']:+.2%}")
    print(f"       Max drawdown: -{real_result['max_drawdown_pct']:.2f}%")

    # ----- Connect to sablier-flow Client -----
    print("\n[3/5] Calling sablier-flow Client...")
    fake_tee = _make_fake_tee(real_data, target_features)
    client = Client(
        api_key="demo-key",
        endpoint="http://demo",
        pinned_image_digest="sha256:" + "d" * 64,
        attestation_mode="fake-for-dev",
        transport=fake_tee,
        poll_interval_s=0.0,
    )

    result = client.alternative_versions(
        real_data.iloc[:-test_window],
        n_paths=100,
        horizon=test_window,
        features=target_features,
        seed=42,
    )
    print(f"     Got {result.n_paths} synthetic histories, "
          f"shape ({result.horizon}, {len(result.feature_names)})")

    # ----- Convert to backtrader feeds + run -----
    print("\n[4/5] Converting to backtrader feeds + running strategy on each...")
    test_index = pd.bdate_range(real_tail.index[0], periods=test_window)
    feeds = as_backtrader_feeds(result, ticker_column="SPY", index=test_index)
    print(f"     {len(feeds)} backtrader PandasData feeds")

    synth_results = []
    for i, feed in enumerate(feeds):
        synth_results.append(run_backtest(feed))
        if (i + 1) % 25 == 0:
            print(f"     ... ran {i + 1}/{len(feeds)}")

    synth_sharpes = [r["sharpe"] for r in synth_results]
    print("     Synthetic Sharpe distribution:")
    print(f"       median:  {np.median(synth_sharpes):+.3f}")
    print(f"       5%-95%:  [{np.percentile(synth_sharpes, 5):+.3f}, "
          f"{np.percentile(synth_sharpes, 95):+.3f}]")

    # ----- Verdict -----
    print("\n[5/5] Computing robustness report...")
    report = sablier_flow.robustness(real_result["sharpe"], synth_sharpes)

    print()
    print("=" * 70)
    print(f"  VERDICT: {report.verdict.upper().replace('_', ' ')}")
    print(f"  Backtest Overfitting Score: {report.overfit_score:.0%}")
    print("=" * 70)
    print(f"  Real Sharpe (backtrader):  {report.real_value:+.3f}")
    print(f"  Synthetic median:          {report.synthetic_median:+.3f}")
    print(f"  Synthetic 95% CI:          [{report.synthetic_p5:+.3f}, {report.synthetic_p95:+.3f}]")
    print(f"  Real total return:         {real_result['total_return']:+.2%}")
    print(f"  Real max drawdown:         -{real_result['max_drawdown_pct']:.2f}%")
    for note in report.notes:
        print(f"   * {note}")


if __name__ == "__main__":
    main()
