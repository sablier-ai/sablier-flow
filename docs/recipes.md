# Data-source recipes

`sablier-flow` is engine-agnostic and data-source-agnostic. It expects a `pd.DataFrame` with a `DatetimeIndex` and one numeric column per feature (a "feature" is whatever the customer wants — a ticker close, a yield, a vol surface point, a credit spread, etc.).

Below are the canonical snippets for the data layers real quant desks actually use. Drop the resulting DataFrame straight into `sablier_flow.fit` / `sablier_flow.generate` / `sablier_flow.robustness` (or `sablier_flow.evaluate_family`).

> If you just want to try the SDK with no data wiring, use the bundled demo dataset:
>
> ```python
> import sablier_flow
> df = sablier_flow.demo_data()   # SPY/QQQ/IWM/TLT + 3 macro features (VIX, TNX, DXY), 2010-2023, no network
> ```

---

## ArcticDB (Man Group / Bloomberg open source)

Most likely data layer in 2026 quant desks. Storage-backed DataFrames with point-in-time indexing.

```python
import arcticdb as adb
import sablier_flow as sf

ac = adb.Arctic("s3://my-fund-quant-bucket")
df = ac.get_library("equity_prices").read("us_largecap").data
# df: DatetimeIndex × ticker columns, exactly the shape sablier-flow expects.

sf.login()
fit = sf.fit(df, features=list(df.columns),
             data_types={c: "price" for c in df.columns},
             horizon=252)
paths = sf.generate(fit.model_id, n_paths=100)
synth_dfs = paths.as_dataframes()

real_result   = my_backtest(df)
synth_results = [my_backtest(p) for p in synth_dfs]
report = sf.robustness(real_result, synth_results, primary_metric="sharpe")
print(report.summary())
```

ArcticDB returns native pandas DataFrames — no conversion step. Same recipe works against ArcticDB's lmdb / s3 / gcs backends.

---

## kdb+/q (via PyKX)

Standard at quant firms with kdb-native pipelines.

```python
import pykx as kx
import sablier_flow as sf

# Open the customer's kdb+ session
kx.q("\\l /path/to/historical.q")
prices_kt = kx.q('select date, sym, close from prices where date within 2010.01.01 2023.12.31')

# Convert keyed table to wide DataFrame
df = prices_kt.pd().pivot(index="date", columns="sym", values="close")
df.index = pd.to_datetime(df.index)  # PyKX returns date as object dtype

sf.login()
fit = sf.fit(df, features=list(df.columns),
             data_types={c: "price" for c in df.columns},
             horizon=252)
paths = sf.generate(fit.model_id, n_paths=100)
synth_dfs = paths.as_dataframes()

real_result   = my_backtest(df)
synth_results = [my_backtest(p) for p in synth_dfs]
report = sf.robustness(real_result, synth_results, primary_metric="sharpe")
print(report.summary())
```

---

## Bloomberg BQuant (Terminal-side)

For analysts running on the Bloomberg Terminal — BQuant integrates ArcticDB so the recipe is identical to the ArcticDB one above. The only difference is the library namespace your fund's BQuant admin set up; ask them for the `ac.get_library(...)` name.

---

## Polars DataFrame (growing adoption in 2026 quant code)

`sablier-flow` expects pandas. Conversion at the boundary is one line:

```python
import polars as pl
import sablier_flow as sf

prices_pl = pl.read_parquet("us_universe.parquet")   # native Polars
df = prices_pl.to_pandas().set_index("date")         # convert at boundary

sf.login()
fit = sf.fit(df, features=list(df.columns),
             data_types={c: "price" for c in df.columns},
             horizon=252)
paths = sf.generate(fit.model_id, n_paths=100)
synth_dfs = paths.as_dataframes()

real_result   = my_backtest(df)
synth_results = [my_backtest(p) for p in synth_dfs]
report = sf.robustness(real_result, synth_results, primary_metric="sharpe")
print(report.summary())
```

Native Polars support is on the roadmap (see [docs/SDK.md](SDK.md) #versioning).

---

## Parquet / CSV / Feather (file-based)

The simplest path. Whatever your data engineering team writes out:

```python
import pandas as pd

df = pd.read_parquet("/data/us_largecap_close_2010_2024.parquet")
# or pd.read_csv("history.csv", index_col=0, parse_dates=True)
# or pd.read_feather("snapshot.feather").set_index("date")
```

---

## Custom internal warehouse (Snowflake / BigQuery / Postgres / Redshift)

```python
import pandas as pd
import sablier_flow as sf
from sqlalchemy import create_engine

eng = create_engine("snowflake://...@account/db/schema")
df = pd.read_sql_query(
    """
    SELECT date, ticker, adj_close
    FROM prices
    WHERE date BETWEEN '2010-01-01' AND '2024-01-01'
    """,
    eng,
    parse_dates=["date"],
)
df = df.pivot(index="date", columns="ticker", values="adj_close")

sf.login()
fit = sf.fit(df, features=list(df.columns),
             data_types={c: "price" for c in df.columns},
             horizon=252)
paths = sf.generate(fit.model_id, n_paths=100)
synth_dfs = paths.as_dataframes()

real_result   = my_backtest(df)
synth_results = [my_backtest(p) for p in synth_dfs]
report = sf.robustness(real_result, synth_results, primary_metric="sharpe")
print(report.summary())
```

Identical patterns for BigQuery (`google.cloud.bigquery`), Postgres (`psycopg`), Redshift (`redshift-connector`).

---

## yfinance (free public data, demos / academic / personal projects only)

Not a production source — Yahoo's API is unofficial and rate-limited — but fine for prototyping or replicating textbook examples.

```python
import yfinance as yf
import sablier_flow as sf

df = yf.download(
    ["SPY", "QQQ", "IWM", "TLT"], start="2010-01-01", end="2024-01-01",
    progress=False, auto_adjust=False,
)["Adj Close"].dropna().sort_index()

sf.login()
fit = sf.fit(df, features=list(df.columns),
             data_types={c: "price" for c in df.columns},
             horizon=252)
paths = sf.generate(fit.model_id, n_paths=100)
synth_dfs = paths.as_dataframes()

real_result   = my_backtest(df)
synth_results = [my_backtest(p) for p in synth_dfs]
report = sf.robustness(real_result, synth_results, primary_metric="sharpe")
print(report.summary())
```

For first-touch evaluation of the SDK without yfinance, use the bundled demo dataset:

```python
df = sablier_flow.demo_data()                        # ~99 KB bundled, no network
print(sablier_flow.available_demo_datasets())        # list other bundled options
```

---

## Across all recipes: the contract

| Property | What `sablier-flow` expects |
|---|---|
| Type | `pandas.DataFrame` |
| Index | `pandas.DatetimeIndex`, sorted ascending, no duplicates, no `NaT` |
| Columns | Numeric (`float32` or `float64`), one per feature |
| Naming | Column names are arbitrary strings; we use them as `feature_names` in the response |
| Missing data | Drop or forward-fill before calling — the SDK rejects NaN rows |
| Length | At least 252 rows (1y daily) for stable training; 1000+ recommended; 5000+ is the empirical sweet spot for daily equity panels (see `sf.demo_data("us_equities_macro_2010_2024")` — ~3500 bars across 7 features) |
| Frequency | Daily is the default. Other frequencies work but the `periods_per_year` arg on `quick_validate` should match (252 for daily, 12 for monthly, etc.). |

The SDK does not know what your features mean — it learns the joint distribution from the rows you give it. Equities, FX, futures, credit spreads, vol surfaces, yields, even non-financial time series (energy demand, weather, retail sales) all work the same way.
