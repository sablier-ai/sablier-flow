# sablier-flow

> **Stop shipping overfit backtests.**
> Run your strategy on **N alternative versions of history** that share your data's statistical fingerprint. If the strategy only works on the one specific past that happened, that's a problem you can now measure.

---

## Canonical 5-line workflow

```python
import sablier_flow as sf
df = your_data  # multivariate time series, DatetimeIndex
fit  = sf.fit(df, features=df.columns.tolist(), data_types={c: 'price' for c in df.columns}, horizon=21)
gen  = sf.generate(fit.model_id, n_paths=100, like=df.iloc[-21:])
synth_results = [my_backtest(d) for d in gen.as_dataframes()]
verdict = sf.robustness(my_backtest(df), synth_results, primary_metric='sharpe')
```

`gen.as_dataframes()` returns `list[pd.DataFrame]`, one per synthetic alternative-history path; your existing `my_backtest(df) -> dict` runs on each unchanged.

## What you get in 30 lines

```python
import pandas as pd
import sablier_flow as sf

# 0. Auth — set SABLIER_FLOW_API_KEY env var, or pass api_key="sk_live_..." per call.
# 1. Load your data — any DataFrame with a DatetimeIndex + numeric columns.
real = pd.read_parquet("my_universe.parquet")
backtest_window = real.loc["2023-01-01":"2024-01-01"]    # the slice you'll evaluate

# 2. Fit (one-time, several minutes). 80/20 train/OOS split + 21-bar embargo by
#    default. The held-out OOS slice is kept encrypted next to the model so
#    sf.validate() picks it up automatically. `data_types=` is required on
#    every call — the bundled demo attaches the canonical map on
#    `df.attrs['data_types']`; for your own data, pass an explicit dict
#    like {'AAPL': 'price', '10Y': 'rate'}.
fit = sf.fit(real, features=list(real.columns),
             data_types=real.attrs['data_types'],
             horizon=252, train_split=0.8, embargo_days=21, seed=42)

# 3. Validate the model on the held-out OOS slice — full structural metric
#    suite (calibration, dependence, tails, dynamics, memorization).
report = sf.validate(fit.model_id, data_types=real.attrs['data_types'])
print(report.overall)                  # 'pass' | 'warn' | 'fail'
print(report.memorization_risk)        # 'low' | 'medium' | 'high'

# 4. Generate N synthetic alternative-history paths shaped like your backtest
#    window. ``like=`` derives length + dates + price anchor from the window.
paths = sf.generate(fit.model_id, n_paths=1000, like=backtest_window,
                    data_types=real.attrs['data_types'], seed=42)
synth_dfs = paths.as_dataframes()      # list[pd.DataFrame], one per path

# 5. Run *your existing* backtest on the real window AND on each synthetic.
real_result   = my_backtest(backtest_window)
synth_results = [my_backtest(df) for df in synth_dfs]

# 6. The smoking gun.
verdict = sf.robustness(real_result, synth_results, primary_metric="sharpe")
print(verdict.verdict)                 # 'robust' | 'borderline' | 'overfit' | 'highly_overfit'
print(verdict.overfit_score)           # 0.85 → real beat 85% of synthetic → overfit
print(verdict.summary())               # one-line English summary you can paste anywhere
verdict.to_html("audit.html")          # shareable single-file report
```

Two-step `fit` + `generate` (instead of one shot) means you fit once and generate as many windows as you want from the same `model_id` — cheap iteration on your strategy without paying to retrain.

See the [getting-started notebook on docs.sablier.ai](https://docs.sablier.ai/notebook/) for the step-by-step walkthrough — it embeds the same code as `examples/00_getting_started.ipynb` and lets you download the raw `.ipynb` to run locally.

---

## Sign up + install

1. **Create an account** at <https://sablier.ai> — email/password or "Sign in with Google" both work. New accounts get free credits to cover several full cycles of the getting-started notebook.
2. **Verify your email** by clicking the confirmation link Sablier sends (Google OAuth users skip this step).
3. **Install the SDK** in a fresh venv:

```bash
python -m venv .venv && source .venv/bin/activate

pip install sablier-flow                           # thin client (~30 MB, no GPU deps)
pip install 'sablier-flow[adapters-backtrader]'    # + backtrader integration
pip install 'sablier-flow[adapters-vectorbt]'      # + vectorbt integration
```

Pin to an exact version (e.g. `sablier-flow==1.0.0`) rather than a range when publishing a backtest audit so the analysis re-runs identically months later. Bump the pin explicitly when you want a newer build.

Transitive deps: `pandas`, `numpy`, `pyarrow`, `httpx`, `cryptography`, `pydantic` — installed automatically with `sablier-flow`. No vendor data libraries (`yfinance` etc.); bundled demo datasets ship inside the wheel.

4. **Authenticate**:

```python
import sablier_flow as sf
sf.login()                       # opens https://sablier.ai/auth/device, click Authorize, done
```

Or pass the API key explicitly (preferred for CI):

```bash
export SABLIER_FLOW_API_KEY=sk_live_<your-token>     # from Dashboard → Settings → API Keys
```

That's the whole setup. The default endpoint `https://flow.sablier.ai/v1` over standard TLS — no `gcloud`, no cert pinning, no extra steps. Credit usage is shown live on the dashboard; a full fit + validate + generate cycle on the bundled demo dataset uses a small fraction of the free starter balance.

---

## What ships

| Capability | API |
|---|---|
| Fit a flow model on your history (joint over all your features at daily / weekly / monthly / quarterly cadence; auto-detected from your `DatetimeIndex`. Intraday lands in 1.1.0.) | `sf.fit(df, features=[...], data_types={...}, horizon=..., train_split=..., embargo_days=...)` |
| Generate N synthetic alternative-history paths, anchored at any window | `sf.generate(model_id, n_paths=..., like=window, data_types={...})` |
| Run the full structural validation suite on the held-out OOS slice | `sf.validate(model_id, data_types={...})` — returns `ValidationReport` with `overall`, `memorization_risk`, and ~20 per-metric entries |
| Score a backtest's overfit (real vs synthetic distribution) | `sf.robustness(real_result, synth_results)` — returns `RobustnessReport` with `overfit_score`, `verdict`, and `synthetic_*` percentiles |
| Deflated Sharpe Ratio under two nulls (analytical Bailey-LdP + empirical synthetic-best-of-N) | `sf.deflated_sharpe(...)` or `verdict.deflated_sharpe(n_trials=N)` |
| Evaluate a *family* of strategy variants — CSCV PBO + family-best DSR | `sf.evaluate_family({"name": fn, ...}, real_data, n_paths=...)` |
| Live drift monitoring once a strategy is deployed | `sf.consistency_check(realized_metric, baseline=robustness_report)` |
| List / inspect / delete your fitted models | `sf.list_models()`, `sf.get_model(model_id)`, `sf.delete_model(model_id)` |
| Bundled demo datasets so you can try it with zero data setup | `sf.demo_data()` — daily SPY/QQQ/IWM/TLT + macro series; `sf.demo_data('us_equities_macro_5min_3mo')` — 5-min intraday |
| Engine adapters | `result.as_dataframes()` for pandas; `from sablier_flow.adapters import as_backtrader_feeds, as_vectorbt_panel, write_lean_csv_universe` |

Full API reference: [docs.sablier.ai](https://docs.sablier.ai).

---

## What happens under the hood

```
your laptop ──HTTPS──> Sablier API (Cloud Run) ──Cloud Tasks──> GPU worker (Cloud Run + L4)
     │                                                                    │
     │   1. POST /v1/jobs                                                  │
     │   ◄── 2. ephemeral X25519 pubkey + image digest                     │
     │                                                                     │
     │   3. envelope-encrypt your DataFrame (X25519 + AES-256-GCM)         │
     │   ──> PUT /v1/jobs/{id}/data ──────────────────────────────────────►│
     │                                                                     │
     │                                          4. decrypt in worker RAM,  │
     │                                             train the flow model,   │
     │                                             generate N paths,       │
     │                                             AES-GCM-encrypt back    │
     │                                                                     │
     │   5. GET /v1/jobs/{id}/result ◄────────────────────────────────────-│
     │   6. decrypt locally; result.paths_returns is yours                 │
     ▼
backtester (pandas / backtrader / vectorbt / LEAN / your own)
```

### Security posture today (alpha)

Honest picture of what the SDK actually guarantees right now:

| Layer | Status |
|---|---|
| **TLS 1.3 in transit** (client ↔ API ↔ worker) | ✓ |
| **One-shot AES-256-GCM symmetric key per job** (wrapped in X25519 envelope to the worker's ephemeral pubkey; never re-used, never persisted) | ✓ |
| **GCS at-rest encryption** with Cloud KMS-managed keys (checkpoints + OOS holdouts + result blobs) | ✓ |
| **Customer data isolation** — each job runs in its own Cloud Run instance, scaled to zero between jobs | ✓ |
| **Image-digest pinning** — the SDK ships a pinned digest of the worker image; mismatched server image is rejected before any data is sent | ✓ |
| **AMD SEV-SNP CPU memory encryption** — encrypted RAM, so even a privileged host OS or GCP operator cannot inspect plaintext during training | 🚧 Not yet — Cloud Run L4 is not a confidential VM. Plaintext customer data exists in worker RAM during the ~minutes-long training job. |
| **NVIDIA H100 CC mode** (GPU memory encryption) + **NRAS attestation chain** | 🚧 Awaiting H100 quota |
| **Cryptographic attestation** verified against AMD / NVIDIA root keys before the customer's encryption key is released | 🚧 Same gate — the SDK's `AttestationVerifier` exists and runs the protocol, but the digest pinned today corresponds to a regular Cloud Run image, not a measured-boot enclave |

**What this means concretely**: today the SDK delivers strong network-layer + storage-layer + key-lifecycle protection that's meaningfully better than most quant data SaaS offerings. It does **not** yet deliver memory-encryption-grade protection against a privileged GCP operator. A CISO evaluating us before SEV-SNP + H100 CC ship needs to see and accept that trade-off.

The full SEV-SNP + H100 CC + NRAS attestation deploys with **v0.6**, which lands when GCP releases our H100 confidential-compute quota. The wire protocol the SDK already speaks is the same one we'll use post-rollout, so customer code does not change.

---

## Try it in 20 minutes — no setup beyond `pip install`

```bash
pip install sablier-flow matplotlib
```

Open [`examples/00_getting_started.ipynb`](examples/00_getting_started.ipynb) and run the cells top-to-bottom. The notebook uses the bundled demo dataset (`sf.demo_data()` for daily SPY/QQQ/IWM/TLT + VIX/TNX/DXY macro series; `sf.demo_data('us_equities_macro_5min_3mo')` for 5-min intraday), so it works with zero data setup and zero external network beyond the SDK's hosted endpoint.

---

## Engine integrations

`sablier-flow` is a data layer, not a backtest engine. We integrate with whatever you already use:

| Engine | How |
|---|---|
| Raw pandas / numpy | `result.as_dataframes(index=...)` → `list[pd.DataFrame]`; `from sablier_flow.adapters import as_array` → `np.ndarray` |
| backtrader | `from sablier_flow.adapters import as_backtrader_feeds` → `list[bt.feeds.PandasData]` |
| vectorbt | `from sablier_flow.adapters import as_vectorbt_panel` → wide `pd.DataFrame` |
| LEAN / QuantConnect | `from sablier_flow.adapters import write_lean_csv_universe` → per-path CSV directories |
| In-house C++ / KDB / proprietary | `result.paths_prices` is plain `np.float32[n_paths, horizon, n_features]` — shim ≤ 50 lines |

---

## Why "in-sample is correct" — short version

The model is trained on the same history your backtest will run on. If that triggers your overfit alarm, the answer is in the memorization metric: every fit ships a `memorization_risk` flag computed from the ratio of median synthetic-to-training NN distance over median training-to-training NN distance. If the model is regurgitating samples, the synth lands on top of training points, the ratio collapses below 0.50, the SDK marks it `high`, and the overfit verdict on top of it is explicitly not to be trusted. When it's `low` or `medium`, the synthetic distribution lives in the data-generating process's *neighborhood*, not on top of the training points themselves — which is precisely what you want to stress-test a strategy against. See [`docs/concepts/in-sample-is-correct.md`](docs/concepts/in-sample-is-correct.md) for the long form.

---

## License

`sablier-flow` (this package) — **Apache-2.0**.

---

## Links

- [Getting-started notebook](examples/00_getting_started.ipynb)
- [Concepts docs](docs/concepts/)
- [sablier.ai](https://sablier.ai) — sign up + manage account
- [docs.sablier.ai](https://docs.sablier.ai) — SDK reference + notebook + recipes
- Bug reports / feedback: <team@sablier.it>
