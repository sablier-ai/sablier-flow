# Engine integration — your backtester, unchanged

The premise: customers already have a backtester. They've invested in it for years. They will not switch engines, port their strategies, or learn a new DSL just to get synthetic data.

So `sablier-flow` doesn't have a "Sablier strategy language." It's a data layer, not an engine. You feed it your DataFrame, it gives you back N synthetic DataFrames with the same schema. Your existing engine consumes those exactly the same way it consumes the real data.

## The universal contract

```python
import sablier_flow as sf

sf.login()
fit       = sf.fit(real_data, features=list(real_data.columns),
                   data_types=real_data.attrs["data_types"], horizon=252)
synthetic = sf.generate(fit.model_id, n_paths=1000, like=real_data.iloc[-252:])
synth_dfs = synthetic.as_dataframes()
# synth_dfs is list[pd.DataFrame], each with the same columns + index as the
# `like=` window you passed in. Your existing backtest function runs unchanged.

real_pnl   = my_backtest(real_data)
synth_pnls = [my_backtest(df) for df in synth_dfs]
report     = sf.robustness(real_pnl, synth_pnls, primary_metric="sharpe")
```

The last three lines are the entire customer integration. Everything below is glue for specific engines that don't natively consume pandas DataFrames.

## Supported engines

### Raw pandas / numpy

Zero adapter needed. Synthetic paths come out as DataFrames. Use them.

### [backtrader](https://github.com/mementum/backtrader)

```python
from sablier_flow.adapters import as_backtrader_feeds   # requires [adapters-backtrader] extra
# ticker_column must be one of synthetic.feature_names; pick the asset
# your backtrader strategy trades on.
feeds = as_backtrader_feeds(synthetic, ticker_column=synthetic.feature_names[0])
for feed in feeds:
    cerebro = bt.Cerebro()
    cerebro.adddata(feed)
    cerebro.addstrategy(MyStrategy)
    cerebro.run()
```

OHLCV is synthesized from close prices (open=high=low=close, vol=placeholder). If your strategy depends on intraday range, you'll want to write your own adapter that ingests minute-level synthetic paths instead.

### [vectorbt](https://github.com/polakowo/vectorbt)

```python
from sablier_flow.adapters import as_vectorbt_panel    # requires [adapters-vectorbt] extra
# ticker_column must be one of synthetic.feature_names.
panel = as_vectorbt_panel(synthetic, ticker_column=synthetic.feature_names[0])
# panel is a wide DataFrame (T × n_paths)
pf = vbt.Portfolio.from_signals(panel, entries, exits, freq="D")
```

Especially useful for parameter sweeps × synthetic paths — vectorbt broadcasts naturally across the panel's column dimension.

### LEAN / QuantConnect

LEAN consumes a per-symbol CSV directory tree. The `lean` adapter writes one such tree per synthetic path:

```python
from sablier_flow.adapters import write_lean_csv_universe   # ships in core, no extra needed
write_lean_csv_universe(synthetic, "lean-data/")
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
import sablier_flow as sf

fit    = sf.fit(real_data, features=list(real_data.columns),
                data_types=real_data.attrs["data_types"], horizon=252)
result = sf.generate(fit.model_id, n_paths=1000, like=real_data.iloc[-252:])
# result.paths_prices is (n_paths, horizon, n_features) np.float32
arr = result.paths_prices

# Write a custom binary your engine reads
for i in range(arr.shape[0]):
    with open(f"synthetic_{i:04d}.bin", "wb") as f:
        f.write(arr[i].astype(np.float64).tobytes())
```

Or push directly into your kdb+ session via PyKX. Convert each path into a `kx.Table` and upsert into a wide synth-paths table; the index column from your `like=` window becomes the timestamp column:

```python
import pykx as kx
import numpy as np

ts = real_data.iloc[-252:].index             # the index your `like=` window used
feature_names = result.feature_names         # column order matches arr's last axis

# One-time: declare the destination table (empty schema; types inferred on first upsert).
kx.q('synth_path:([] path_id:`long$(); ts:`timestamp$(); ' +
     ';'.join(f'{f.lower()}:`float$()' for f in feature_names) + ')')

# Upsert each path as a kx.Table. PyKX's q['name'] handle is the
# idiomatic way to mutate a named server-side table.
for i in range(arr.shape[0]):
    row_dict = {
        'path_id': np.full(arr.shape[1], i, dtype=np.int64),
        'ts': ts,
    }
    for j, fname in enumerate(feature_names):
        row_dict[fname.lower()] = arr[i, :, j].astype(np.float64)
    kx.q['synth_path'].upsert(kx.Table(data=row_dict))
```

## What about engines I haven't heard of?

If the engine reads DataFrames or NumPy arrays or CSVs (which is all of them), you can integrate in under an hour. The list above is just the engines we've personally validated; the architecture is engine-agnostic by construction.

If you have an exotic engine and you'd like us to add a first-class adapter, file an issue with a description of the input format the engine expects.

## What about live trading?

`sablier-flow` produces *synthetic backtest data*. It's not for live execution. The output looks like historical data; you'd run your existing backtest on it. Don't try to use it as a real-time feed.

For *what-if* scenarios on live state ("what if VIX spikes to 60 tomorrow?"), the constraints API in v1.1 covers that. v1.0 is just unconstrained alternative histories.
