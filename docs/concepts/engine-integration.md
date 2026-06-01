# Engine integration — your backtester, unchanged

The premise: customers already have a backtester. They've invested in it for years. They will not switch engines, port their strategies, or learn a new DSL just to get synthetic data.

So `sablier-flow` doesn't have a "Sablier strategy language." It's a data layer, not an engine. You feed it your DataFrame, it gives you back N synthetic DataFrames with the same schema. Your existing engine consumes those exactly the same way it consumes the real data.

## The universal contract

```python
synthetic = client.alternative_versions(real_data, n_paths=1000)
synth_dfs = sablier_flow.adapters.dataframe.as_dataframes(synthetic, index=test_dates)
# synth_dfs is list[pd.DataFrame], each with the same columns + index as real_data.

real_pnl   = my_backtest(real_data)
synth_pnls = [my_backtest(df) for df in synth_dfs]
report     = sablier_flow.robustness(real_pnl, synth_pnls)
```

The last three lines are the entire customer integration. Everything below is glue for specific engines that don't natively consume pandas DataFrames.

## Supported engines

### Raw pandas / numpy

Zero adapter needed. Synthetic paths come out as DataFrames. Use them.

### [backtrader](https://github.com/mementum/backtrader)

```python
from sablier_flow.adapters.backtrader import as_backtrader_feeds
feeds = as_backtrader_feeds(synthetic, ticker_column="SPY", index=test_dates)
for feed in feeds:
    cerebro = bt.Cerebro()
    cerebro.adddata(feed)
    cerebro.addstrategy(MyStrategy)
    cerebro.run()
```

OHLCV is synthesized from close prices (open=high=low=close, vol=placeholder). If your strategy depends on intraday range, you'll want to write your own adapter that ingests minute-level synthetic paths instead.

### [vectorbt](https://github.com/polakowo/vectorbt)

```python
from sablier_flow.adapters.vectorbt import as_vectorbt_panel
panel = as_vectorbt_panel(synthetic, ticker_column="SPY", index=test_dates)
# panel is a wide DataFrame (T × n_paths)
pf = vbt.Portfolio.from_signals(panel, entries, exits, freq="D")
```

Especially useful for parameter sweeps × synthetic paths — vectorbt broadcasts naturally across the panel's column dimension.

### LEAN / QuantConnect

LEAN consumes a per-symbol CSV directory tree. The `lean` adapter writes one such tree per synthetic path:

```python
from sablier_flow.adapters.lean import write_lean_csv_universe
write_lean_csv_universe(synthetic, "lean-data/", index=test_dates)
# Produces:
#   lean-data/path_0000/equity/usa/daily/SPY.csv
#   lean-data/path_0001/equity/usa/daily/SPY.csv
#   ...
```

In LEAN, run your algorithm N times pointing `data-folder` at each `path_NNNN` directory. Aggregate the per-path metrics yourself.

### In-house / proprietary engines

This is the most common case at funds. You have a C++ or Java or KDB engine that's been around for ten years. Everyone has one.

The pattern: write a *one-time* shim that takes a `synthetic.paths_prices` NumPy array and emits whatever format your engine expects (FBP, CSV, custom binary, KDB tickerplant feed). The shim is ~50 lines of code, written once.

```python
import numpy as np

result = client.alternative_versions(real_data, n_paths=1000)
# result.paths_prices is (n_paths, horizon, n_features) np.float32
arr = result.paths_prices

# Write a custom binary your engine reads
for i in range(result.n_paths):
    with open(f"synthetic_{i:04d}.bin", "wb") as f:
        f.write(arr[i].astype(np.float64).tobytes())
```

Or push directly into your KDB ticker:

```python
import pykx
for i in range(result.n_paths):
    pykx.q("`:synth_path", f"insert", {
        "ts": result_dates,
        "path_id": i,
        "spy": arr[i, :, 0],
        "qqq": arr[i, :, 1],
    })
```

## What about engines I haven't heard of?

If the engine reads DataFrames or NumPy arrays or CSVs (which is all of them), you can integrate in under an hour. The list above is just the engines we've personally validated; the architecture is engine-agnostic by construction.

If you have an exotic engine and you'd like us to add a first-class adapter, file an issue with a description of the input format the engine expects.

## What about live trading?

`sablier-flow` produces *synthetic backtest data*. It's not for live execution. The output looks like historical data; you'd run your existing backtest on it. Don't try to use it as a real-time feed.

For *what-if* scenarios on live state ("what if VIX spikes to 60 tomorrow?"), the constraints API in v1.1 covers that. v1.0 is just unconstrained alternative histories.
