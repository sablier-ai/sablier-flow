# Data sourcing — bring your own

`sablier-flow` does not integrate with market-data vendors. We don't bundle Refinitiv, we don't ship a Bloomberg connector, we don't license any data ourselves. That's deliberate, and it's central to the product.

## Why

Quant funds either:

1. **Pay for proprietary data** (Bloomberg, Refinitiv, FactSet, S&P Capital IQ, Compustat, CRSP, Axioma, ...) where the contract forbids redistribution. The data is their licensed asset, expensive to acquire, and contractually impossible to share with a third-party SaaS.
2. **Have built proprietary data** (alternative data: satellite imagery features, credit-card panels, app-usage telemetry, web scrapes). This is their *moat*. They will not feed it through a vendor's pipeline.
3. **Use open data poorly enough that buying our product wouldn't help.** Self-disqualifying.

Categories 1 and 2 are our customers. They will not let their data leave their infrastructure under any circumstance. They've built their entire operating model — data licensing, vendor contracts, security reviews, compliance — around that constraint.

This is exactly why the architecture is what it is. **You hold your data; we hold the compute.** The envelope-encryption + image-pinning wire protocol the SDK already speaks is designed to bind that pairing to a confidential VM with hardware memory encryption; the substrate that delivers that binding ships with v0.6 (see [Security posture](#security-posture-today)).

## The data contract — DataFrame in, DataFrame out

Customers provide a `pandas.DataFrame` with a `DatetimeIndex`. Each column is one feature. That's the entire integration surface.

```python
import pandas as pd
import sablier_flow as sf

# Your data, loaded however you load data today.
real_data = pd.read_parquet("my_universe_2010_2023.parquet")
# or pd.read_csv, or pyodbc → DataFrame, or your_data_lake_client.fetch(),
# or LEAN's data folder, or anything else.

# Columns: tickers, factors, signals — whatever your backtest takes.
# Index: a DatetimeIndex at whatever frequency you use (daily, weekly, monthly).
print(real_data.shape)       # (3000, 50) — 3000 rows × 50 features
print(real_data.dtypes)      # all numeric

# REQUIRED — per-column data-type annotation. sablier-flow uses this to
# pick the right transform per column (log-return for prices, z-score
# for rates / volatility / index levels, identity for already-stationary
# returns). Pure Parquet doesn't carry this — you build it once at the
# loader, by reading whatever column-semantic mapping your data team
# already has. Example for a price-only equity panel:
data_types = {col: 'price' for col in real_data.columns}
# Or for a mixed panel — annotate per column:
# data_types = {
#     'SPY': 'price', 'QQQ': 'price',
#     'US10Y': 'rate', 'VIX': 'volatility', 'DXY': 'index',
# }

sf.login()                                      # or set SABLIER_FLOW_API_KEY
fit  = sf.fit(real_data,
              features=list(real_data.columns),
              data_types=data_types,
              horizon=252)
gen  = sf.generate(fit.model_id, n_paths=1000, like=real_data.iloc[-252:])
synthetic = gen.as_dataframes()                 # list[pd.DataFrame], one per alt-history
```

> **Note on `df.attrs['data_types']`.** The bundled demo dataset
> (`sf.demo_data()`) ships the annotation pre-attached so you can pass
> `real_data.attrs['data_types']` straight through in examples. For
> your own data loaded from Parquet/CSV/SQL, `df.attrs` will be empty —
> build the dict at the loader as shown above and persist it however
> you persist column metadata at your firm.

That's it. No schema registration, no field-mapping config, no "please contact our data team."

## Where your DataFrame comes from

This is your problem, but here are the patterns we see:

### Pattern 1: Parquet/CSV files from your data lake

```python
import pandas as pd
df = pd.read_parquet("s3://my-bucket/universe/equity_panel_2023.parquet")
df = df.set_index("date").sort_index()
```

If you already have a Parquet/Feather/CSV workflow feeding your backtester, sablier-flow consumes the same files. Zero new infrastructure.

### Pattern 2: SQL query against your warehouse

```python
import pandas as pd
import sqlalchemy

engine = sqlalchemy.create_engine("postgresql://localhost/quant_warehouse")
df = pd.read_sql(
    "SELECT date, ticker, adj_close, volume FROM equity_prices "
    "WHERE date BETWEEN '2010-01-01' AND '2024-01-01'",
    engine,
).pivot(index="date", columns="ticker", values="adj_close")
```

Snowflake, BigQuery, Redshift, MotherDuck — all fine, all return DataFrames.

### Pattern 3: Vendor SDK → DataFrame

Most vendors ship Python clients that return DataFrames natively:

```python
# Bloomberg's blpapi via xbbg
from xbbg import blp
df = blp.bdh(["SPY US Equity", "QQQ US Equity"], "PX_LAST",
             "2015-01-01", "2024-01-01")

# Refinitiv Eikon
import refinitiv.data as rd
rd.open_session()
df = rd.get_history(universe=["SPY.N", "QQQ.O"], fields="TR.PriceClose",
                    start="2015-01-01", end="2024-01-01")

# Direct from your fund's pricing API
df = my_fund.api.get_prices(tickers, start, end)
```

Whatever you already use to feed your backtester — feed sablier-flow the same way.

### Pattern 4: LEAN's data folder

LEAN stores per-symbol daily/minute CSVs under `data/equity/usa/daily/<ticker>.zip`. Read them with:

```python
import pandas as pd
from pathlib import Path
from zipfile import ZipFile

def lean_close(data_root: Path, ticker: str) -> pd.Series:
    with ZipFile(data_root / "equity/usa/daily" / f"{ticker.lower()}.zip") as z:
        df = pd.read_csv(z.open(f"{ticker.lower()}.csv"), header=None,
                         names=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d %H:%M")
    return df.set_index("date")["close"] / 10_000.0  # LEAN stores 1/10000 ticks

df = pd.DataFrame({
    t: lean_close(Path("./lean-data"), t) for t in ["SPY", "QQQ", "IWM"]
}).dropna()
```

### Pattern 5: Your own proprietary alt-data

Whatever you've built — sentiment scores, credit-card panels, satellite-derived inventory features — already lives in some pipeline that produces DataFrames. Plug that into `sf.fit(your_alt_data_df, features=list(your_alt_data_df.columns), data_types={c: 'index' for c in your_alt_data_df.columns}, horizon=252)`. Most alt-data series are non-tradeable factor levels — use `data_types='index'` for those (`'rate'` for credit-card APRs, `'price'` only if the series is itself tradeable). The model trains on whatever joint structure you feed it.

## What sablier-flow does with your DataFrame

1. **Serialize the DataFrame to Parquet bytes** *on your machine*, before any network call.
2. **Generate a fresh AES-256-GCM symmetric key** locally, per job. Used once, never persisted.
3. **Encrypt the Parquet bytes + the params JSON** with that symmetric key.
4. **Wrap the symmetric key** in an X25519 envelope to the worker's ephemeral public key — extracted from an attestation quote the SDK verifies against the SDK-pinned image digest. Mismatched digest → SDK refuses to encrypt; your data never leaves the laptop.
5. **Ship the encrypted bundle** to the worker over TLS 1.3.
6. **The worker decrypts in RAM**, runs the FLOW model, encrypts the result with the same one-shot symmetric key, returns it.

## Security posture today

The protocol above runs on every request — the code is real, the digest pinning is real, the keys are real. What's **not yet** real is the hardware substrate that would make step 6 immune to a privileged GCP operator:

| Layer | Status |
|---|---|
| TLS 1.3 in transit, KMS-encrypted at rest in GCS, one-shot per-job symmetric keys | ✓ Today |
| Image-digest pinning verified before the encryption key is generated | ✓ Today (structure-only check; full root-key signature verification ships with the SEV-SNP rollout) |
| **AMD SEV-SNP** CPU memory encryption — encrypts RAM so even a privileged host OS / GCP operator can't see plaintext during training | 🚧 v0.6 (awaiting GCP H100-CC quota) |
| **NVIDIA H100 CC mode** — GPU memory encryption, same goal at the device level | 🚧 v0.6 |
| **NRAS attestation chain** — NVIDIA-signed attestation of the GPU state | 🚧 Roadmap |

What this means: today's deployment is meaningfully better than vanilla cloud SaaS (encrypted everywhere except in the worker's RAM during the ~minutes-long training job), but it does **not** yet defend against a privileged GCP insider inspecting that RAM. The SDK and the wire protocol the customer code touches stay identical when SEV-SNP + H100 CC ship — only the substrate underneath changes.

If your security review absolutely requires hardware memory encryption before you can ship data, hold until the SEV-SNP rollout. If TLS + KMS + ephemeral keys + image-digest pinning meets your bar today (most quant-tech-stack reviews do clear this), the current release is usable.

## Data quality is your problem

We deliberately reject bad data at the client edge, before it ships, so you don't burn credits on a job that was going to fail anyway:

- **Mixed dtypes** — every column must be numeric.
- **Mixed timezones** — index must be tz-naive or tz-aware (consistently across rows).
- **Duplicate index** — must be sorted strictly ascending; the client raises on duplicates.
- **Missing data** — NaN is tolerated. The server forward-fills, then back-fills, then zero-fills before training.
- **Survivorship bias** — that's *your* responsibility. We don't filter for it, don't detect it, and the FLOW model will faithfully learn whatever bias is in the input. Garbage in, biased garbage out.

The error messages point to the exact problem so the customer's data team can fix at the source.

## Frequency

`sf.fit` auto-detects the bar period from your `DatetimeIndex` via `pd.infer_freq` (with a median-bar-delta fallback for irregular indices) and classifies the data into one of `daily` / `weekly` / `monthly` / `quarterly`. Pass `frequency=` to override.

Intraday classification (minute / 5-min / 15-min bars) is **deferred to 1.1.0** when the cyclical minute-of-day / day-of-week embeddings ship. The bundled `sf.demo_data('us_equities_macro_5min_3mo')` is a preview-only sample so you can see the data shape today; running `sf.fit` on a 5-min DataFrame raises during the schema check. For now, aggregate to daily bars before fitting.

When intraday lights up: the model will emit a single price track per feature per path, not full OHLCV. Strategies that key off intra-bar high/low/volume will see flat OHLC on synthetic.

## Multiple asset classes in one model

Yes — feed equity prices alongside bond yields alongside FX. The model learns the joint distribution. Common patterns:

```python
df = pd.DataFrame({
    "SPY": equity_close["SPY"],
    "TLT": bond_close["TLT"],
    "EURUSD": fx["EURUSD"],
    "VIX": vix_close,
    "CL_F": crude_oil_futures,
})
```

This is how customers typically test macro strategies — they want to know how their strategy holds up across alternative joint scenarios of equity + rates + FX.

## What we *don't* do

| | |
|---|---|
| Provide data | ❌ |
| Vendor data through us | ❌ |
| Cache your data on disk | ❌ — Parquet ciphertext is held in GCS only for the duration of the job, then deleted; the SDK's optional local cache lives on *your* disk, not ours |
| Cross-customer sharing | ❌ — every job spawns its own ephemeral worker instance, scaled to zero between jobs; encrypted blobs are namespaced per `model_id` and rotated per-customer KMS keys |
| Sell aggregated insights | ❌ — we have no data-monetization business model and the engineering doesn't aggregate across customers |

This is the trust pitch. We are a compute layer, not a data layer.
