"""sablier-flow — first-time customer experience.

Run this file to feel the full customer workflow end-to-end against the
production sablier-flow service:

    .venv/bin/python examples/try_it_yourself.py

You'll see:
  1. A real backtest (MA crossover on SPY) run on a small synthetic
     price panel.
  2. sablier_flow training a flow model on the panel (sf.fit).
  3. sablier_flow validating the fitted model on a held-out OOS slice
     (sf.validate) — that's where the structural-validation suite +
     memorization-risk score live now (a separate billed call, NOT
     bundled with the generation result).
  4. sablier_flow generating ~50 synthetic alternative histories of the
     backtest window (sf.generate).
  5. The same backtest run on every synthetic history.
  6. sf.robustness comparing the real result to the synthetic
     distribution and printing a verdict + overfit score.

What's REAL here (all of it):
  - sablier_flow.fit / sablier_flow.generate / sablier_flow.validate
    talking to the production hosted service over HTTPS via the
    default HttpxTransport. The api_key comes from
    ~/.sablier/credentials (run ``sablier_flow.login()`` if missing).
  - sablier_flow.robustness — pure client-side scoring.

A tiny synthetic panel (~3 features, ~260 bars) keeps the run cheap.
Swap in your own DataFrame to use it on real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sablier_flow as sf


# -----------------------------------------------------------------------------
# Section 1 — the customer's existing backtest, unchanged
# -----------------------------------------------------------------------------


def my_backtest(prices: pd.DataFrame, ticker: str = "ASSET_A", *,
                fast: int = 10, slow: int = 30) -> dict:
    """Standard moving-average crossover backtest on close prices.

    Returns annualized Sharpe + max drawdown + total return. This is the
    kind of function any quant has lying around — sablier-flow doesn't
    require you to rewrite it.
    """
    px = prices[ticker]
    fast_ma = px.rolling(fast).mean()
    slow_ma = px.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(int)
    position = signal.shift(1).fillna(0).astype(int)
    rets = px.pct_change().fillna(0.0)
    strat_rets = position * rets
    if strat_rets.std() < 1e-12:
        sharpe = 0.0
    else:
        sharpe = float(strat_rets.mean() / strat_rets.std() * np.sqrt(252))
    equity = (1.0 + strat_rets).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min())
    final_return = float(equity.iloc[-1] - 1.0)
    return {"sharpe": sharpe, "max_drawdown": max_dd, "total_return": final_return}


# -----------------------------------------------------------------------------
# Section 2 — build a small synthetic price panel
# -----------------------------------------------------------------------------


def make_synthetic_panel(*, n_bars: int = 400, seed: int = 7) -> pd.DataFrame:
    """Generate a tiny multi-asset daily price panel.

    3 correlated GBM-ish series across ~1 trading year. Small enough to
    keep the fit + generate + validate round-trips cheap (~handful of
    credits) and avoid pulling external data.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-01", periods=n_bars)
    n_assets = 3
    # Correlated daily returns: shared market factor + idiosyncratic noise.
    market = rng.normal(0.0004, 0.012, size=n_bars)
    idio = rng.normal(0.0, 0.008, size=(n_bars, n_assets))
    betas = np.array([1.0, 0.8, 1.2])
    rets = market[:, None] * betas[None, :] + idio
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    df = pd.DataFrame(
        prices,
        index=index,
        columns=["ASSET_A", "ASSET_B", "ASSET_C"],
    )
    return df


# -----------------------------------------------------------------------------
# Section 3 — the actual customer workflow
# -----------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print(f"sablier-flow {sf.__version__} — overfit detection demo")
    print("=" * 70)

    # ----- 1. Load data (the customer's existing dataset) ---------------
    print("[1/6] Building a tiny synthetic 3-asset price panel...")
    real_data = make_synthetic_panel()
    features = list(real_data.columns)
    data_types = {c: "price" for c in features}
    print(f"     {len(real_data)} bars x {len(features)} features "
          f"(from {real_data.index[0].date()} to {real_data.index[-1].date()})")

    # ----- 2. Run the customer's backtest on the real tail --------------
    test_window = 60
    print(f"\n[2/6] Running customer backtest on the last {test_window} days of real history...")
    real_window = real_data.tail(test_window)
    real_result = my_backtest(real_window)
    print("     Real-history backtest:")
    print(f"       Sharpe:        {real_result['sharpe']:+.3f}")
    print(f"       Max drawdown:  {real_result['max_drawdown']:.2%}")
    print(f"       Total return:  {real_result['total_return']:+.2%}")

    # ----- 3. Fit a flow model on the training portion -----------------
    print("\n[3/6] sf.fit(...) — training a flow model on prod...")
    train_df = real_data.iloc[:-test_window]
    fit = sf.fit(
        train_df,
        features=features,
        data_types=data_types,
        horizon=test_window,
        train_split=0.8,
        embargo_days=5,
        seed=42,
    )
    print(f"     model_id = {fit.model_id}")
    print(f"     training_loss = {fit.training_loss:.4f}  ({fit.loss_source})")

    # ----- 4. Validate the fitted model (separate call) -----------------
    # Structural validation + memorization risk now live in their own
    # ValidationReport returned by sf.validate(). They are no longer
    # bundled into GenerationResult.
    print("\n[4/6] sf.validate(...) — structural + memorization check...")
    rep = sf.validate(fit.model_id, data_types=data_types)
    print(f"     overall:                       {rep.overall}")
    print(f"     memorization_risk:             {rep.memorization_risk}")
    print(f"     memorization_nn_distance_ratio:"
          f" {rep.memorization_nn_distance_ratio}")
    print(f"     holdout (true OOS?):           {rep.holdout}")
    if rep.caveats:
        print("     Caveats from honest-threshold pass:")
        for c in rep.caveats[:5]:
            print(f"       * {c}")
    if rep.overall == "fail" or rep.memorization_risk == "high":
        print("     Warning: validation flagged a problem — interpret the verdict")
        print("              below with caution.")

    # ----- 5. Generate synthetic alternative histories -----------------
    n_paths = 50
    print(f"\n[5/6] sf.generate(..., n_paths={n_paths}, like=window)...")
    gen = sf.generate(
        fit.model_id,
        n_paths=n_paths,
        like=real_window,
        data_types=data_types,
        seed=42,
    )
    print(f"     Got {gen.n_paths} synthetic paths of shape "
          f"({gen.horizon}, {len(gen.feature_names)}).")
    if gen.memorization_risk is not None:
        print(f"     Bundled memorization_risk on the result: {gen.memorization_risk}")

    # ----- 6. Run the SAME backtest on each synthetic history ---------
    print("\n[6/6] Running customer backtest on every synthetic path...")
    synth_dfs = gen.as_dataframes()
    synth_results = [my_backtest(df) for df in synth_dfs]
    synth_sharpes = [r["sharpe"] for r in synth_results]
    print("     Synthetic Sharpe distribution:")
    print(f"       median:    {np.median(synth_sharpes):+.3f}")
    print(f"       5%-95%:    [{np.percentile(synth_sharpes, 5):+.3f}, "
          f"{np.percentile(synth_sharpes, 95):+.3f}]")
    print(f"       min:       {min(synth_sharpes):+.3f}")
    print(f"       max:       {max(synth_sharpes):+.3f}")

    # ----- 7. Verdict --------------------------------------------------
    report = sf.robustness(real_result["sharpe"], synth_sharpes)
    print()
    print("=" * 70)
    print(f"  VERDICT: {report.verdict.upper().replace('_', ' ')}")
    print(f"  Backtest Overfitting Score: {report.overfit_score:.0%}")
    print("=" * 70)
    print(f"  Your real Sharpe:        {report.real_value:+.3f}")
    print(f"  Synthetic median:        {report.synthetic_median:+.3f}")
    print(f"  Synthetic 95% CI:        [{report.synthetic_p5:+.3f}, "
          f"{report.synthetic_p95:+.3f}]")
    print(f"  N synthetic paths:       {report.n_synthetic}")
    for note in report.notes:
        print(f"   * {note}")

    # ----- 8. Optional: write JSON ------------------------------------
    out_dir = Path(__file__).parent / "demo_output"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "real": real_result,
                "synthetic_sharpes": synth_sharpes,
                "verdict": report.verdict,
                "overfit_score": report.overfit_score,
                "synthetic_median": report.synthetic_median,
                "synthetic_p5": report.synthetic_p5,
                "synthetic_p95": report.synthetic_p95,
                "validation_overall": rep.overall,
                "memorization_risk": rep.memorization_risk,
            },
            indent=2,
        )
    )
    print(f"\nJSON:   {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
