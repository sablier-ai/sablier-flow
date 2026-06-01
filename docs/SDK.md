# sablier-flow SDK — Full Reference

> Canonical reference for the `sablier-flow` Python SDK. Type signatures and
> code examples are copy-paste runnable against the current PyPI release
> (`sablier_flow.__version__`). If a behavior or API you expect is not
> documented here, it does not exist in the SDK yet — don't hallucinate
> features.

---

## What sablier-flow does

The customer has a backtest function `f(prices) -> {"sharpe": ...}`. They run it on their real history and want to know whether the result is genuine signal or overfit to the specific realization their data took. `sablier-flow` answers that by:

1. Training a joint flow model on the customer's history on a remote GPU worker (see [Security posture](#security-posture-today-alpha) for the current and target deployment specifics).
2. Generating `N` synthetic *alternative versions* of the same history — different paths, same statistical fingerprint.
3. Running the customer's backtest on every synthetic alt-history.
4. Comparing the real result to the distribution of synthetic results.

If the real result sits at the extreme tail of the synthetic distribution, the strategy is exploiting realization-specific noise — **overfit**. If it sits in the bulk, the strategy is **robust**.

Two additional outputs surface for serious quants:
- **Deflated Sharpe Ratio (DSR)** under two nulls (empirical synthetic-best-of-N + analytical Bailey–LdP IID-Gaussian).
- **Probability of Backtest Overfitting (PBO)** via Combinatorially Symmetric Cross-Validation on the real history alone.

---

## Table of contents

- [Quick start](#quick-start-zero-data-setup)
- [Installation](#installation)
- [Authentication](#authentication) — `sf.login()` / env / `~/.sablier/credentials`
- [Security posture today (alpha)](#security-posture-today-alpha)
- [The workflow: `fit` → `generate` → `validate`](#the-workflow-fit--generate--validate)
  - [Schema contract](#schema-contract--what-real_data-must-look-like)
  - [Strict `features=` validation](#strict-features-validation-071)
  - [Async jobs (`fit_async` / `fetch_result` / `list_jobs` / `cancel_job`)](#async-jobs--fit_async--fetch_result--list_jobs--cancel_job)
- [Forward generation — deployment forecasting](#forward-generation--deployment-forecasting)
  - [Predictive validity (`sf.predictive_rank_score`)](#predictive-validity--sfpredictive_rank_score)
- [Strategy families and parameter sweeps](#strategy-families-and-parameter-sweeps)
- [Interpreting the output](#interpreting-the-output)
  - [Verdict](#verdict)
  - [Deflated Sharpe Ratio](#deflated-sharpe-ratio)
  - [Probability of Backtest Overfitting](#probability-of-backtest-overfitting)
  - [Memorization risk](#memorization-risk)
  - [Structural validation](#structural-validation)
- [Live drift monitoring](#live-drift-monitoring)
- [Demo datasets](#demo-datasets)
- [Engine adapters](#engine-adapters)
- [Model management](#model-management)
- [CLI reference](#cli-reference)
- [Full API reference](#full-api-reference)
- [Known limitations](#known-limitations)
- [Common errors](#common-errors)
- [Glossary](#glossary)
- [Versioning](#versioning)

---

## Quick start (zero data setup)

```bash
pip install sablier-flow
```

The canonical 5-line workflow — run this verbatim on any DataFrame
with a `DatetimeIndex` (use `df = sf.demo_data()` to see it end-to-end
with no setup):

```python
import sablier_flow as sf
df = your_data  # multivariate time series, DatetimeIndex
fit  = sf.fit(df, features=df.columns.tolist(), data_types={c: 'price' for c in df.columns}, horizon=21)
gen  = sf.generate(fit.model_id, n_paths=100, like=df.iloc[-21:])
synth_results = [my_backtest(d) for d in gen.as_dataframes()]
verdict = sf.robustness(my_backtest(df), synth_results, primary_metric='sharpe')
```

`gen.as_dataframes()` returns `list[pd.DataFrame]`, one per synthetic
alternative-history path, with the same columns / index shape as the
slice you passed to `like=`; your existing backtest function runs on
each `d` unchanged.

A longer, explicitly-authenticated variant that also runs
`sf.validate(...)` for the cheap OOS structural check:

```python
import sablier_flow as sf

sf.login()                                               # device flow: opens browser, you confirm → ~/.sablier/credentials

real = sf.demo_data()                                    # SPY/QQQ/IWM/TLT + macro, daily 2010-2023
backtest_window = real.loc["2023-01-01":"2024-01-01"]    # the slice you'll evaluate

# `data_types=` is required on every fit / generate / validate call.
# The demo ships the canonical map on `df.attrs['data_types']` so the
# simplest call passes it straight through.
fit     = sf.fit(real, features=list(real.columns), data_types=real.attrs["data_types"], horizon=252, seed=42)
report  = sf.validate(fit.model_id)                                          # zero-config OOS check
paths   = sf.generate(fit.model_id, n_paths=100, like=backtest_window, data_types=real.attrs["data_types"], seed=42)
synth   = paths.as_dataframes()

real_result   = my_backtest(backtest_window)
synth_results = [my_backtest(df) for df in synth]

verdict = sf.robustness(real_result, synth_results, primary_metric="sharpe")
print(verdict.summary())
verdict.to_html("audit.html")
```

That's the entire installation-to-verdict path against `https://flow.sablier.ai/v1` over standard TLS — no cert pinning, no GCS fetch. Swap `real` for your own DataFrame once you've seen it work.

`sf.login()` writes the minted API key to `~/.sablier/credentials` (mode `0600`). Subsequent processes — including different Python interpreters — pick the key up automatically via `sf.Client()`. For CI / non-interactive runs, set `SABLIER_FLOW_API_KEY=sk_live_...` in the environment instead.

---

## Installation

```bash
pip install sablier-flow                            # thin client only (~30 MB)
pip install 'sablier-flow[adapters-backtrader]'     # + backtrader integration
pip install 'sablier-flow[adapters-vectorbt]'       # + vectorbt integration
```

Python 3.10 or newer.

---

## Authentication

The SDK supports three credential sources, resolved in this order whenever
`api_key=` is omitted:

1. **Explicit kwarg** — `sf.Client(api_key="sk_live_...")` or `sf.fit(real, api_key="sk_live_...")`. Always wins.
2. **`SABLIER_FLOW_API_KEY` env var** — set this in CI / containers / AI coding assistants.
3. **`~/.sablier/credentials` file** — written by `sf.login()` for interactive use.

### First time? Sign up at <https://sablier.ai>

The SDK has no programmatic signup — like Stripe, Twilio, AWS, OpenAI, and every other infrastructure SDK, account creation lives on the web because that's where email verification, ToS acceptance, and billing onboarding logically belong. Five-step flow:

1. Go to <https://sablier.ai> → **Sign up**. Email/password or "Sign in with Google" both work.
2. **Verify your email** — Sablier sends a confirmation link at signup; click it before continuing. ("Sign in with Google" skips this step because Google has already verified the email.)
3. Open a terminal and run `python -c "import sablier_flow as sf; sf.login()"`.
4. Click **Authorize** on the browser tab the SDK pops open.
5. The SDK now works on this machine without any further setup. Subsequent sessions read the stored credential automatically.

If you skip step 2, the **Authorize** click in step 4 returns a "verify your email first" prompt — the dashboard surfaces a one-click resend so you can complete the loop without leaving the page.

### `sf.login()` — recommended for humans

```python
import sablier_flow as sf
sf.login()
```

prints a one-time code, opens `https://sablier.ai/auth/device` in your
browser, waits for you to sign in (or recognizes a session you already
have) and click **Authorize**. On success the SDK writes the minted API
key to `~/.sablier/credentials` and `sf.Client()` picks it up
automatically from then on — no key paste, no env var.

The flow is RFC 8628 Device Authorization Grant adapted to mint a
Sablier API key. The credentials file is mode `0600` (owner read/write
only); the secret is wiped from the server-side device row the moment
the SDK's polling loop retrieves it, so a leaked `device_code` can't
replay the handoff.

```python
sf.logout()         # drops the credentials entry locally
                    # (does NOT revoke the key on the server — use the dashboard for that)
```

Multiple profiles are supported when you switch between staging / prod
or per-team identities:

```python
sf.login(profile="staging")
client = sf.Client(profile="staging")     # picks up the staging entry
```

### Non-interactive use — CI, containers, AI coding assistants

Anything that can't open a browser — CI jobs, Docker containers, Claude
Code / Cursor running headless, automated scripts — should use the env
var path:

```bash
# Once, on the dashboard: Settings → API Keys → Create → copy sk_live_...
export SABLIER_FLOW_API_KEY=sk_live_<your-token>

# Then anywhere — no sf.login() needed, no credentials file required:
python -c "import sablier_flow as sf; print(sf.whoami())"
```

For AI coding assistants running on your laptop (Claude Code, Cursor,
Continue, etc.): if you've already done `sf.login()` once, those tools
read the same `~/.sablier/credentials` file — they inherit your
authentication with zero extra setup. For a clean separation, set
`SABLIER_FLOW_API_KEY` in the shell they launch from and they'll
prefer the env var.

### Environment fallbacks

The SDK also reads `SABLIER_FLOW_ENDPOINT`, `SABLIER_FLOW_CERT`,
`SABLIER_FLOW_PINNED_IMAGE_DIGEST`, and `SABLIER_FLOW_ATTESTATION_MODE`
when the corresponding constructor kwarg is omitted. Explicit kwargs
always win.

### Manual API key (for CI / web dashboard users)

1. Sign in / sign up at <https://sablier.ai>.
2. Settings → API Keys → New API Key.
3. Copy the `sk_live_...` value (shown once).
4. Either `export SABLIER_FLOW_API_KEY=sk_live_...` or pass explicitly:

```python
sf.fit(real, api_key="sk_live_...")
client = sf.Client(api_key="sk_live_...")
```

### Custom / staging deployments

```python
client = sf.Client(
    api_key="sk_live_...",
    endpoint="https://staging.example.com/v1",
    attestation_mode="fake-for-dev",        # skip signature math; still enforces image digest + hardware
    verify="/path/to/staging-ca.pem",       # pin a self-signed cert
)
```

---

## Security posture today (alpha)

The SDK and worker run a full envelope-encryption + image-digest-pinning protocol that's *designed* to bind the customer's encryption keys to a measured-boot confidential VM. The protocol code is in place and runs on every request. What's **not yet** in place is the underlying confidential-compute substrate.

| Layer | Status |
|---|---|
| **TLS 1.3 in transit** (client ↔ API ↔ worker) | ✓ |
| **One-shot AES-256-GCM symmetric key per job** (wrapped in X25519 envelope to the worker's ephemeral pubkey; never re-used, never persisted to disk) | ✓ |
| **GCS at-rest encryption** with Cloud KMS-managed keys (checkpoints + OOS holdouts + result blobs) | ✓ |
| **Customer data isolation** — each job runs in its own Cloud Run instance, scaled to zero between jobs | ✓ |
| **Image-digest pinning** — the SDK ships a pinned digest of the worker image; mismatched server image is rejected before any data is sent | ✓ |
| **AMD SEV-SNP CPU memory encryption** — even a privileged host OS or GCP operator cannot inspect plaintext during training | 🚧 Not yet — Cloud Run L4 is not a confidential VM. Plaintext customer data exists in worker RAM during the ~minutes-long training job. |
| **NVIDIA H100 CC mode** (GPU memory encryption) + **NRAS attestation chain** | 🚧 Awaiting H100 quota |
| **Cryptographic attestation** verified against AMD / NVIDIA root keys before the customer's encryption key is released to the worker | 🚧 Same gate — the SDK's `AttestationVerifier` runs the protocol on every request, but the digest pinned today corresponds to a regular Cloud Run image, not a measured-boot enclave. Production-grade attestation (signature math against pinned root keys) ships with the SEV-SNP rollout. |

**Bottom line**: today the SDK delivers strong network-layer + storage-layer + key-lifecycle protection. It does **not** yet deliver memory-encryption-grade protection against a privileged GCP operator inspecting worker RAM during training. The full SEV-SNP + H100 CC + NRAS attestation deploys with v0.6, which lands when GCP releases our H100 confidential-compute quota. The wire protocol the SDK already speaks is the same one we'll use post-rollout — customer code doesn't change.

---

## The workflow: `fit` → `generate` → `validate`

The SDK splits the lifecycle into three explicit calls so you train once and reuse the trained model across as many windows / strategies as you want.

```python
# 1. Train once (~minutes, scales with data size). The server splits 80/20 with
#    a 21-bar embargo by default and keeps the held-out OOS slice encrypted
#    alongside the model so sf.validate(model_id) picks it up automatically.
fit = sf.fit(
    real,
    features=list(real.columns),         # all columns are co-generated jointly
    data_types=real.attrs["data_types"], # per-column transform annotation
    horizon=252,                         # training-window length (bars, not days)
    train_split=0.8,                     # 80% train, 20% OOS held out for validate()
    embargo_days=21,                     # bar gap between train end + OOS start
    seed=42,
)
print(fit.model_id)                 # opaque handle; pass to generate / validate / get_model
print(fit.training_loss, fit.loss_source)
# loss_source ∈ {'validation', 'training_proxy'} — the latter means the inner
# val split was too small to form a single (obs_length + horizon) window, so
# the loss reported is the training-loss proxy; the real OOS check still
# happens via sf.validate(...) on the persisted holdout.

# 2. Validate the model on the held-out OOS slice (zero-config — no holdout
#    DataFrame argument needed). Returns a ValidationReport with `overall`,
#    `memorization_risk`, and ~20 per-metric entries.
report = sf.validate(fit.model_id)

# 3. Generate N synthetic paths shaped like any window you want. `like=df`
#    derives length + index + price anchor from the window — synth paths
#    overlay your real series directly.
paths = sf.generate(fit.model_id, n_paths=1000, like=backtest_window,
                    data_types=real.attrs["data_types"], seed=42)
```

**`data_types=`.** A required `dict[str, str]` mapping every column in `features=` to one of `{'price', 'return', 'rate', 'index', 'volatility'}`. The SDK picks the right transform per column (log-return for prices, z-score for rates / volatility / index levels, identity for already-stationary returns). Bundled demo DataFrames attach the canonical map on `df.attrs['data_types']` so you can pass it straight through. Missing the kwarg raises `TypeError` with the allowed-set message; an unknown value raises `ValueError`.

**Frequency auto-detection.** `sf.fit` auto-detects the bar period from the median Δt of `real.index` and classifies the data into one of `'daily'` / `'weekly'` / `'monthly'` / `'quarterly'`. Pass `frequency=` to override. Irregular indices raise (the SDK refuses to silently round-off your bars). Intraday classification is deferred to 1.1.0 — the 5-min preview demo is shipped so you can see the data shape and `data_types=` pattern, but intraday `fit` is rejected during schema validation.

### Strategies with a lookback / warmup period

A strategy that needs pre-backtest history to compute its signal (12-1 momentum, 6-month rolling stats, anything that reads bars older than the first bar of the backtest window) requires the same warmup on synthetic data. Otherwise the strategy's first-bar signal is computed against zeros / NaNs and the synthetic backtest doesn't compare apples-to-apples against the real one.

The fix is to extend `like=` to include the warmup period, not just the backtest window:

```python
LOOKBACK = 252                                    # 12-month momentum lookback
backtest_start = pd.Timestamp("2023-01-01")
backtest_end   = pd.Timestamp("2024-01-01")

# Pull a window that runs from (backtest_start - 12 months) through backtest_end.
warmup_start = backtest_start - pd.DateOffset(months=12)
extended_window = real.loc[warmup_start:backtest_end]   # ≈ 504 bars instead of 252

paths = sf.generate(fit.model_id, n_paths=1000, like=extended_window,
                    data_types=real.attrs["data_types"], seed=42)
synth_dfs = paths.as_dataframes()                 # each is 504 bars long

def my_backtest(prices: pd.DataFrame) -> dict:
    # `prices` is 504 bars: first 252 are warmup (signal computation),
    # last 252 are the actual backtest. The strategy already handles this
    # for real data — it'll handle it identically for synthetic.
    sig = prices["SPY"].pct_change(LOOKBACK)      # 12-1 momentum signal
    pos = sig.shift(1).gt(0).astype(int)
    rets = prices["SPY"].pct_change().fillna(0.0) * pos
    rets = rets.iloc[LOOKBACK:]                   # discard warmup before Sharpe
    return {"sharpe": float(rets.mean() / rets.std() * np.sqrt(252))}

real_result   = my_backtest(real.loc[warmup_start:backtest_end])
synth_results = [my_backtest(df) for df in synth_dfs]
```

The synthetic paths inherit the price anchor at `warmup_start` (so they continue from the real price level on that date) and run for `len(extended_window)` bars forward. Quality during the warmup period is identical to quality during the backtest window — the model treats every bar in the path the same way.

**Cost note**: extending `like=` doesn't cost extra credits per call; `generate` is billed per path × bars on the server side, so a 504-bar generation costs roughly twice a 252-bar one. Plan accordingly for long-lookback strategies.

**`features=`.** Single list — all listed columns are co-generated jointly. There is no target / conditioning distinction at the API level; everything is modeled together.

**`horizon=`.** The window length the model is trained against. The generator is horizon-agnostic, so `generate(model_id, horizon=M)` works for any `M` — quality is best near the trained value and degrades modestly as you stretch further past it. Default depends on data length; pass explicitly for intraday since the unit is bars (not days).

**`train_split=None`.** Pass `None` to skip the 80/20 split and train on the full DataFrame. Useful when you've already done a split externally or when running an offline calibration where OOS isn't relevant. In that mode, `sf.validate(model_id)` requires you to pass `holdout_data` explicitly.

### Schema contract — what `real_data` must look like

`sf.fit`, `sf.generate(anchor_data=...)`, and `sf.validate(holdout_data=...)` all take a `pd.DataFrame`. The constraints they enforce locally (before the network round-trip) are:

| field | requirement |
|---|---|
| `df.index` | `pd.DatetimeIndex`, monotonic increasing, no duplicates (tz-naive or tz-aware, both fine) |
| `df.columns` | numeric dtype on every column you list in `features=` — NaNs pass through to the model (it masks); a column whose post-ffill NaN fraction exceeds 0.7 is rejected with an error naming it |
| values | raw prices, returns, rates, index levels, or volatility — the per-column transform is selected from `data_types=` (log-return for prices, z-score for rates / vol / index levels, identity for returns) |
| `data_types=` (kwarg) | **required dict** mapping every column in `features=` to one of `{'price', 'return', 'rate', 'index', 'volatility'}`. Missing the kwarg raises `TypeError`; unknown values raise `ValueError`. Bundled demos attach the canonical map on `df.attrs['data_types']`. |
| frequency | auto-detected from the median Δt of `df.index`. Allowed: `'daily'`, `'weekly'`, `'monthly'`, `'quarterly'`. Irregular indices raise. Intraday classification is deferred to 1.1.0 (the 5-min demo is preview-only). |
| length | ≥ 200 rows on `fit` (the SDK rejects locally); shorter slices are allowed for `like=` / `anchor_data=` / `holdout_data=` |

### Strict `features=` validation

`sf.fit` requires `features=` to match `real_data.columns` exactly when set: every column in `features` must exist in the DataFrame **and** every numeric column in the DataFrame must be listed in `features`. Mismatches raise `ValueError` with a list of the offending names so you can fix them in the call site instead of running an expensive fit on the wrong universe.

Pass `features=None` to opt out and fit on every numeric column (no coverage check).

### Async jobs — `fit_async` / `fetch_result` / `list_jobs` / `cancel_job`

Every sync method has an async sibling that returns a `JobHandle` immediately after the encrypted upload completes. The handle carries the `job_id`, the kind (`'fit'` / `'generate'` / `'validate'`), and the one-shot AES key needed to decrypt the result.

```python
handle = sf.fit_async(real, features=list(real.columns),
                      data_types=real.attrs["data_types"], horizon=252)
print(handle.job_id, handle.kind)

# Do other work — or shut your laptop, restart your kernel, whatever.
# Later (in the same OR a different process):
result = sf.fetch_result(handle)        # FitResult, GenerationResult, or ValidationReport depending on handle.kind
```

Persisting across processes:

```python
import json

# Process 1: open the job, save the handle
handle = sf.fit_async(real, features=[...], data_types={...})
with open("job-handle.json", "w") as f:
    json.dump(handle.to_dict(), f)

# Process 2 (different machine, different day, different Python)
handle = sf.JobHandle.from_dict(json.load(open("job-handle.json")))
fit = sf.fetch_result(handle)
```

**The handle is a bearer secret** — anyone holding it can fetch the result. Treat it like an API key: don't paste it into chat, don't commit it.

```python
sf.list_jobs(status="running")       # see what's in flight
sf.list_jobs(status="completed")     # recent completed jobs
sf.cancel_job(handle)                # or pass a raw job_id string
```

Async + sync share every line of the wire and crypto path — the cache, the attestation check, the envelope encryption. The only thing async skips is the local-disk cache (a cache hit would skip the TEE round-trip and leave no `job_id` to return).

---

## Forward generation — deployment forecasting

Everything above frames the SDK around **backtest augmentation**: synthetic paths that parallel a realized backtest window. The same generator also serves the inverse problem — **deployment forecasting**: synthetic paths that project forward from your most recent bar to predict the distribution of strategy performance you should expect when you go live.

The mechanics are identical to alt-history generation; the only difference is where the anchor sits:

| Use case | Call | Anchor |
|---|---|---|
| Alt-history (overfit audit) | `sf.generate(model_id, like=backtest_window)` | `like.iloc[0]` — start of your past backtest window |
| Forward forecast (deployment) | `sf.generate(model_id, horizon=N, anchor_data=real.iloc[-200:])` | `anchor_data.iloc[-1]` — "today" (your last bar) |

### Recipe

```python
import sablier_flow as sf
import numpy as np
import pandas as pd

real = pd.read_parquet("my_universe.parquet")
# real.index[-1] is "today"

# Per-column data_types — required on every call. Build the dict explicitly
# for your own data (or attach it on `real.attrs['data_types']` ahead of time).
data_types = {col: "price" for col in real.columns}      # adjust per column as needed

# 1. Fit on full history. Default train_split=0.8 reserves an OOS slice
#    for the auto-validation in step 2.
fit = sf.fit(real, features=list(real.columns), data_types=data_types,
             horizon=60, seed=42)

# 2. Verify the model is structurally healthy before trusting its forecast.
report = sf.validate(fit.model_id)                                        # zero-config OOS check
assert report.overall != "fail" and report.memorization_risk != "high"

# 3. Generate N forward paths anchored at "today". The anchor_data tail
#    tells the server "condition on these bars, start the trajectory
#    from anchor_data.index[-1]".
forward_paths = sf.generate(
    fit.model_id,
    n_paths=1000,
    horizon=60,                       # bars to project forward
    anchor_data=real.iloc[-200:],     # last 200 bars (model's obs_length) = today's context
    data_types=data_types,            # required on every call
    seed=42,
)

# 4. Your backtest doesn't know or care that the data is forward-looking.
#    Same f(prices) -> dict that ran on the backtest window or alt-histories.
forward_dfs = forward_paths.as_dataframes()
forward_sharpes = np.array([my_backtest(df)["sharpe"] for df in forward_dfs])

# 5. Pure numpy — you own the distribution analytics. No new verb needed.
print(f"expected sharpe over next 60 bars: {np.median(forward_sharpes):+.2f}")
print(f"90% CI:                            [{np.percentile(forward_sharpes, 5):+.2f}, "
      f"{np.percentile(forward_sharpes, 95):+.2f}]")
```

The forward paths are 60 bars long (matching `horizon=`) and their index runs from the bar AFTER `real.index[-1]` forward. The first synthetic future bar is `forward_dfs[0].iloc[0]`.

### Two alternative anchoring choices

- **`anchor_data=real.iloc[-200:]` (recommended)** — keep the default 80/20 fit split so you get `sf.validate()` for free; pass the recent tail (must be at least the model's `obs_length`, ~200 daily bars) as the conditioning context. Forward generation starts from `real.index[-1]`.
- **`train_split=None` at fit time** — train on every bar including the most recent. The server-stored anchor naturally lands at `real.index[-1]`, so `sf.generate(model_id, horizon=60)` (no `anchor_data`) gives you the forward forecast directly. Tradeoff: `sf.validate()` requires an explicit `holdout_data` kwarg since the server has no auto-OOS slice.

### Predictive validity — `sf.predictive_rank_score`

A faithful generator should preserve the **ranking** of strategies between synthetic and real data — picking the highest-Sharpe variant on synth-forward paths should match the highest-Sharpe variant on the eventual realized OOS deployment. This is a two-axis quality definition: distributional fidelity alone is not sufficient.

The risk the distributional gate alone does not catch: a generator can pass standard distributional checks yet **invert** the strategy ranking, leading a practitioner to systematically pick the worst real-market variant. Predictive-rank validity is the second axis you need to check.

`sf.predictive_rank_score` re-runs that calibration on **the customer's own model + strategy family**:

```python
# User runs THEIR strategy family on both real OOS data and synth forward paths.
# Use the slice sf.fit held out at train time so the calibration runs on
# truly unseen data. A naive last-N-row slice would overlap the server's
# held-out OOS slice (last 20% minus embargo) and bias the rank
# correlation upward.
real_oos       = real.loc[fit.holdout_start_date:fit.holdout_end_date]   # OOS reference from FitResult
anchor_end_pos = real.index.get_indexer([real_oos.index[0]])[0]
ref_anchor     = real.iloc[anchor_end_pos - 200:anchor_end_pos]          # 200 bars before holdout start

forward_paths = sf.generate(fit.model_id, n_paths=200, horizon=len(real_oos),
                             anchor_data=ref_anchor,
                             data_types=real.attrs["data_types"])

real_sharpes  = {name: backtest_fn(real_oos)["sharpe"]
                 for name, backtest_fn in strategies.items()}
synth_sharpes = {
    name: float(np.mean([backtest_fn(df)["sharpe"]
                         for df in forward_paths.as_dataframes()]))
    for name, backtest_fn in strategies.items()
}

# Sablier does the analysis — pure numpy + scipy.stats, no path generation.
score = sf.predictive_rank_score(real_sharpes, synth_sharpes)
print(score.summary())
# "Well calibrated: Spearman ρ = +0.82 (95% CI [+0.61, +0.94]) across 24
#  strategies. Your strategy ranking on synth-forward paths is a meaningful
#  proxy for ranking on the eventual realized deployment window. Magnitude
#  bias (mean |sharpe_real - sharpe_synth|) = 0.36 — the rank can be right
#  while the absolute number is biased, so do not read synth medians as
#  point forecasts."

print(score.verdict, score.spearman_rho, score.ci_95)
# 'well_calibrated' +0.82 (+0.61, +0.94)
```

Verdict bands:

| `verdict` | Spearman ρ | Bootstrap CI lower bound | What to do |
|---|---|---|---|
| `well_calibrated` | ≥ 0.60 | > 0 | Trust the forward forecast's ranking; deploy on the highest-ranked synth-forward strategy |
| `weakly_calibrated` | [0.30, 0.60) OR CI crosses zero | — | Use as a tiebreaker only; don't gate deploy decisions on it |
| `uncalibrated` | [−0.30, 0.30) | — | Don't read synth ranking as predictive; the forecast is uncalibrated for this universe |
| `inverted` | < −0.30 | — | DO NOT deploy; your model picks the worst real-market strategies first. Investigate regime shift, broken features, or training-data leakage. |

**Magnitude vs rank.** The score reports rank correlation, not absolute Sharpe correctness. A well-calibrated model can still have a mean |ΔSharpe| of ~0.3 — the rank is preserved but the absolute number is biased. Read forward synth medians as ordinal forecasts (this strategy will beat that one), not point estimates (this strategy will land at Sharpe +0.5).

**Caveats.** The C7 protocol assumes the synthesizer was trained on a regime that includes the deployment regime. A genuinely novel regime (new asset class, structural break, post-event) can break predictive validity even on a model that scored `well_calibrated` historically. Re-run the score with fresh OOS data periodically; gate deployment on the most recent calibration.

### How to read a forward-Sharpe CI

When you have a synthetic forward Sharpe distribution and want to interpret its p5–p95 band as a deployment forecast, the SDK already gives you three pieces that compose into a defensible reading:

| Question | The right gate |
|---|---|
| Is the CI **shape** (width, tail thickness) consistent with what real OOS data looks like? | `sf.validate(model_id).overall == 'pass'` — the structural metric suite (`tail_quantiles`, `marginal_ks`, `volatility_clustering`, etc.) tests synth distribution against the held-out OOS slice. A passing `validate` is evidence the synth CI isn't pathologically narrow or wide. |
| Is the **ranking** across strategies preserved between synth and real? | `sf.predictive_rank_score(...).verdict == 'well_calibrated'` — gates the rank claim. |
| Is the **center** of the CI biased on the absolute Sharpe axis? | `PredictiveRankReport.mean_abs_metric_gap` — typical magnitude bias. Widen the synth p5–p95 by this amount in each direction for a conservative envelope. |

**The combined recipe**, given a synth forward Sharpe CI of `[1.2, 2.3]`:

```python
fit_report   = sf.validate(fit.model_id)
calibration  = sf.predictive_rank_score(real_oos_sharpes, synth_sharpes)

if fit_report.overall == 'pass' and calibration.verdict == 'well_calibrated':
    # Conservative envelope for capital sizing — widen by magnitude bias.
    bias = calibration.mean_abs_metric_gap
    capital_envelope = (1.2 - bias, 2.3 + bias)
    print(f"deployment envelope: ~{capital_envelope}")
else:
    # Don't size capital on the synth CI alone; rerun the structural
    # validation + recalibrate, or restrict to strategies whose rank is
    # robust within the failure mode you have.
    ...
```

**What's NOT validated.** Direct empirical coverage of the synth CI (i.e., "does 90% of realized outcomes actually fall in the p5–p95 band?") is not tested by either `validate` or `predictive_rank_score`. Validating that directly would require ≳100 independent held-out anchor dates from the same model — most customers don't have that history, and the structural metrics catch the relevant failure modes indirectly. The honest read: treat the synth CI as a model-grounded forecast envelope (not a calibrated 90% probability statement) and use the validation + ranking gates above to decide whether to trust it.

---

## Strategy families and parameter sweeps

When you have multiple variants of the same strategy and want to know which (if any) survive a multiple-testing correction:

```python
strategies = {
    f"ma_{f}_{s}": (lambda f, s: lambda df: my_backtest(df, fast=f, slow=s))(f, s)
    for f, s in [(5, 20), (10, 30), (20, 60), (30, 90)]
}

report = sf.evaluate_family(
    strategies,
    real,
    n_paths=100,
    pbo_cscv_splits=16,         # CSCV partitions for PBO computation (SDK floor; <16 under-detects overfit)
    seed=42,
)

print(report.summary())
print(report.real_argmax_strategy, report.real_max_value)
print(report.deflated_sharpe.realistic)
print(report.pbo)               # 0..1; lower is better — see PBO interpretation table below
```

Generates synthetic paths once (paying the fit cost a single time), runs every strategy on every path, then computes the family-best DSR + CSCV PBO. Cheaper than N separate `fit + generate + robustness` calls when N > 1.

PBO standalone (no synthetic data needed) is exposed as `sf.probability_of_backtest_overfitting(strategies, real_data, ...)`.

---

## Interpreting the output

### Verdict

`RobustnessReport.verdict` is bucketed from `overfit_score`:

| Bucket | `overfit_score` | Meaning |
|---|---|---|
| `robust` | `[0.00, 0.70)` | Real result is consistent with the synthetic distribution. No overfit signal. |
| `borderline` | `[0.70, 0.85)` | Real result is in the top quartile of synth; defensible but not unambiguous. |
| `overfit` | `[0.85, 0.95)` | Real result exceeds 85%+ of synthetic alt-histories. Likely exploits realization-specific noise. |
| `highly_overfit` | `[0.95, 1.00]` | Real result is in the top 5%. Do not deploy without out-of-sample re-validation. |

For higher-is-better metrics, `overfit_score = mean(synthetic < real)` — the fraction of alt-histories where the strategy did worse than reality. For lower-is-better metrics (drawdown), it's reversed.

**`robust` ≠ profitable.** The verdict measures overfit only — it's *orthogonal* to whether the strategy made money. A money-losing strategy can be `robust` (just bad, not overfit); a money-making strategy can be `overfit` (alpha is realization-specific noise). The `summary()` string makes this explicit when the real value sits outside the synth 5–95 CI — read the sign of the Sharpe separately from the verdict.

**Caveats on verdict stability.**

- Verdict labels were consistent across seeds in our N=3 spot checks (`overfit_score` swing < 0.05), but we don't yet claim guaranteed cross-seed stability. A strategy whose real Sharpe sits exactly at the 95th-percentile boundary could flip between `borderline` and `overfit` on a different seed; sit comfortably inside one bucket and the verdict is robust to seed in practice.
- `memorization_risk='low'` ≠ "the model is good at everything." It means the synthetic distribution isn't reproducing training samples — read the per-metric breakdown for the structural axes that matter to your strategy.
- `memorization_nn_distance_ratio` may sit at the lower edge of the `Healthy` 0.85–1.15 band (or dip briefly into `Suspicious`) for universes with > 10 jointly-modeled columns (curse of dimensionality on NN search), even when the model is fine. We saw 0.91 on a 14-column panel vs 1.13 on a 7-column one with identical hyperparams. Consider per-regime / per-asset-class sub-models for large universes.

`RobustnessReport.summary()` returns the verdict as a single plain-English sentence, prefixed by any structural-validation or memorization warnings. This is what a customer's CI / Slack alert should print.

### Deflated Sharpe Ratio

The bucketed verdict is intuitive but informal. The DSR is the academic-grade significance test (Bailey & López de Prado 2014). The SDK computes it under **two nulls side-by-side**:

```python
dsr = verdict.deflated_sharpe(strategy_returns=daily_returns, n_trials=1)
print(dsr.realistic)                       # DSR under Sablier realistic null — regime-aware
print(dsr.analytical)                      # DSR under Bailey-LdP IID-Gaussian — regime-blind
print(dsr.expected_max_sr_realistic)       # E[max SR_n] from synthetic distribution
print(dsr.expected_max_sr_analytical)      # E[max SR_n] closed-form
print(dsr.threshold_sr_realistic)          # SR needed for DSR=0.95 under realistic null
print(dsr.threshold_sr_analytical)         # under analytical null
```

The two nulls usually agree when markets are calm and disagree when the customer's training data covers a regime shift. When they disagree, the realistic null is the better answer — it's grounded in the actual statistical fingerprint of the customer's data, not an IID-Gaussian assumption.

`strategy_returns` enables the Bailey-LdP skew/kurtosis correction on the analytical null. Pass the per-period returns of the strategy on real data. If omitted, the SDK falls back to no higher-moment correction (`γ₃ = 0, γ₄ = 3`).

`n_trials=1` for a single backtest. For a family of `M` strategies, use `M` — though typically `evaluate_family(...)` handles this automatically.

### Probability of Backtest Overfitting

PBO via Combinatorially Symmetric Cross-Validation (Bailey et al. 2015) is computed on the *real* history alone — no synthetic data needed. The procedure:

1. Split real history into `S` contiguous chunks (`pbo_cscv_splits`, SDK floor 16 — fewer splits under-detect overfit on the small-strategy-family case and are no longer recommended).
2. For each `C(S, S/2)` partition of chunks into train/test:
   - Identify in-sample-best strategy on the train partition.
   - Compute its rank percentile on the test partition.
3. `pbo = fraction of partitions where in-sample-best ranks below median out-of-sample`.

| `pbo` value | Interpretation |
|---|---|
| ≤ 0.2 | In-sample-best is consistently out-of-sample-best → grid search has signal. |
| ~ 0.5 | No signal — parameter selection is noise. |
| ≥ 0.6 | In-sample-best routinely loses out-of-sample → systematic overfitting. |

### Memorization risk

`ValidationReport.memorization_risk ∈ {'low', 'medium', 'high'}` based on a nearest-neighbor distance ratio between synthetic and training samples:

```
nn_distance_ratio = median(synthetic-to-training NN dist) / median(training-to-training NN dist)
```

The denominator's training-to-training NN search excludes self-pairs (the diagonal is masked out). The holdout slice is *not* part of this ratio — both medians come from the training set; the holdout drives the structural-metric suite, not this denominator.

Thresholds (calibrated for financial-returns flow models):

| `memorization_nn_distance_ratio` | `memorization_risk` | Action |
|---|---|---|
| `> 0.80` | `low` | Synth distributed through the training manifold at training-like density. |
| `[0.50, 0.80]` | `medium` | Synth tighter than training (typical when the model under-disperses tails). Cross-check against `coverage_*` metrics. |
| `< 0.50` | `high` | Synth essentially overlaps training points — the model is regurgitating. Don't trust the overfit verdict on top of it. |

Why these thresholds (and not the off-the-shelf image-diffusion `< 0.95` cutoff): financial daily-returns are drawn from a noisy continuous distribution, so a perfectly-calibrated flow produces synth that lands *within* the training manifold (ratio < 1.0 is normal, not memorization). Empirically, a customer running `validate` with `coverage_95 = 0.951` (essentially nominal calibration) was being flagged "high memorization" at ratio 0.84 — those two readings are mutually exclusive (a memorized model has collapsed intervals, not nominal coverage). Below `0.50` the synth is closer to training than training is to itself, which *is* a genuine signal of literal sample regurgitation.

### Structural validation

`ValidationReport.overall ∈ {'pass', 'warn', 'fail'}` aggregates a suite of ~20 metrics into a weighted-quality score:

- `excellent` quality contributes weight 1.0
- `good` → 0.8
- `acceptable` → 0.5
- `poor` → 0.0

Score ≥ 0.80 → `pass`; 0.50–0.80 → `warn`; < 0.50 → `fail`.

`ValidationReport.metrics` has the full per-metric breakdown. Each entry is a dict with:

```python
{
    "value": 0.0234,                           # raw scalar (lower is better)
    "quality": "good",
    "passed": True,
    "interpretation": "Max KS statistic: ...",  # one-line human read
    "thresholds": {"excellent": 0.05, "good": 0.10, "acceptable": 0.20},
    "category": "distribution",                # 'calibration' | 'distribution' | 'dependence' | 'temporal' | 'extreme'
    "metadata": {...},                          # per-feature breakdown etc.
}
```

The full metric list is grouped by category:

**Calibration** (per-observation): `coverage_50`, `coverage_90`, `coverage_95`, `pit_uniformity`, `crps`. These say "should I trust the uncertainty bands?"

**Distribution** (marginal shape): `marginal_ks`, `energy_distance`, `tail_quantiles`, `tail_heaviness`. These say "do synthetic returns look like real returns at the feature level?"

**Dependence** (joint structure): `pearson_correlation`, `spearman_correlation`, `tail_dependence_upper/lower`, `copula_distance`. These say "is the cross-asset correlation structure preserved?"

**Temporal / dynamics**: `acf_returns`, `volatility_clustering`, `leverage_effect`, `cross_correlation`. The stylized facts of finance.

**Extreme / regime**: `correlation_breakdown`, `drawdown_distribution`. Behavior in tail events.

Pass thresholds are server-side; the SDK echoes each metric's threshold dict in its `thresholds` field for transparency.

---

## Live drift monitoring

Once a strategy is in production, run a continuous "is reality still consistent with the synthetic distribution we trained on?" check:

```python
import pickle, sablier_flow as sf

# --- Pre-deployment, once ---
verdict = sf.robustness(my_backtest(historical_data), synth_results)
with open("baseline.pkl", "wb") as f:
    pickle.dump(verdict, f)

# --- Post-deployment, every monitoring window ---
with open("baseline.pkl", "rb") as f:
    baseline = pickle.load(f)

realized_sharpe = compute_live_sharpe()      # e.g. trailing-12-month annualised
drift = sf.consistency_check(realized_sharpe, baseline=baseline)

print(drift.summary())
print(drift.verdict)                         # 'consistent' | 'drifting' | 'out_of_distribution'
print(drift.drift_score)                     # signed; ~0 = at median, ±1 = at CI edge
```

| `verdict` | Geometric meaning | Operational signal |
|---|---|---|
| `consistent` | Realized value inside baseline 5%–95% CI | No action |
| `drifting` | Outside the CI but inside the observed envelope | Watch the next window; consider re-training |
| `out_of_distribution` | Outside the baseline `[min, max]` envelope | Regime has shifted; retrain before relying on the model again |

You can also pass a raw `Sequence[float]` as the baseline (e.g., `FamilyReport.synthetic_max_values`) when monitoring against the family-best-of-N distribution rather than a single strategy.

---

## Demo datasets

```python
sf.demo_data()                                       # default: us_equities_macro_2010_2024
sf.demo_data("us_equities_2010_2024")                # SPY/QQQ/IWM/TLT only, no macros
sf.demo_data("us_equities_macro_5min_3mo")           # 5-min intraday — 7 tickers, 3 months
sf.available_demo_datasets()                         # list all bundled names
```

Bundled parquets ship inside the wheel (`pip install sablier-flow` includes them). Zero network access required to load.

---

## Engine adapters

`sf.generate` returns a `GenerationResult` with `paths_prices: ndarray` of shape `(n_paths, horizon, n_features)`. Adapters convert this into whatever shape the customer's engine consumes.

### Universal (always available)

```python
from sablier_flow.adapters import as_dataframes, as_array

dfs = as_dataframes(result, index=pd.bdate_range(...))   # list[pd.DataFrame]
arr = as_array(result)                                    # ndarray (n_paths, horizon, n_features)
```

Or the convenience method on the result itself: `result.as_dataframes(index=...)`.

### backtrader (requires `[adapters-backtrader]` extra)

```python
from sablier_flow.adapters import as_backtrader_feeds

feeds = as_backtrader_feeds(result, ticker_column="SPY", index=pd.bdate_range(...))
# list[bt.feeds.PandasData] — each one's OHLC is synthesized from a single
# close-price column (the ticker_column) since the flow model emits prices, not OHLCV.
```

**Gotcha**: backtrader's default `SharpeRatio` analyzer uses `timeframe=Years` and returns `None` on windows shorter than ~2 years. Use:

```python
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                    timeframe=bt.TimeFrame.Days, riskfreerate=0.0, annualize=True)
```

### vectorbt (requires `[adapters-vectorbt]` extra)

```python
from sablier_flow.adapters import as_vectorbt_panel

panel = as_vectorbt_panel(result, ticker_column="SPY", index=pd.bdate_range(...))
# wide (horizon, n_paths) DataFrame — vectorbt treats the n_paths dimension as
# a parameter sweep, so the customer's strategy is vectorized across every
# alt-history in a single call.
```

### LEAN / QuantConnect

```python
from sablier_flow.adapters import write_lean_csv_universe

write_lean_csv_universe(result, "lean-data/", index=pd.bdate_range(...))
# Writes one CSV per path under lean-data/path_NNN/ in LEAN universe format.
```

### Custom engines

For engines without a Python adapter today, use the CLI to write Parquet:

```bash
sablier-flow generate --input prices.parquet --n 100 --out ./paths/
# Writes ./paths/path_0000.parquet, path_0001.parquet, ...
```

Or register a community adapter via Python entry-points:

```toml
[project.entry-points."sablier_flow.adapters"]
nautilus = "your_pkg:as_nautilus_catalog"
```

After install, `sablier_flow.adapters.as_nautilus_catalog` resolves and `sablier_flow.adapters.available_adapters()` lists it.

---

## Model management

Fitted models persist server-side for ~30 days (TTL refreshes on every successful generate / validate call). The SDK exposes management operations:

```python
models = sf.list_models()                            # list[Model], most-recent first
m = sf.get_model("959c9b30-d2ed-4cc3-9edb-...")      # Model dataclass
sf.delete_model("959c9b30-...")                      # idempotent; marks expired + best-effort blob cleanup
```

`Model` fields include `model_id`, `features`, `training_horizon`, `training_start_date`, `training_end_date`, `holdout_start_date`, `holdout_end_date`, `train_split`, `embargo_days`, `n_assets`, `status`, `training_loss`, `created_at`, `last_used_at`, `expires_at`.

---

## CLI reference

For non-Python languages (R, KDB+, shell pipelines, CI/CD).

```bash
sablier-flow version
sablier-flow adapters list

# Fit + generate in one call; writes one Parquet per synthetic path.
sablier-flow generate \
    --input prices.parquet --n 100 --out ./paths/ \
    [--horizon 252] [--seed 42] [--features SPY,QQQ] \
    [--api-key sk_live_...] [--idempotency-key job-123]

# Compute robustness from real + synthetic backtest results saved as JSON
sablier-flow robustness --real real.json --synthetic synth.json [--metric sharpe]
```

Exit code `0` = success; non-zero indicates the failure category. All subcommands accept `--json` to write structured output to stdout.

---

## Full API reference

Every signature is verbatim from the source.

### `Client`

```python
sf.Client(
    api_key: str,
    *,
    endpoint: str | None = None,            # falls back to "https://flow.sablier.ai/v1"
    pinned_image_digest: str | None = None,
    attestation_mode: str = "production",   # "production" | "fake-for-dev"
    transport: Transport | None = None,     # for tests / in-process simulation
    timeout_s: float = 60.0,
    poll_interval_s: float = 2.0,
    poll_timeout_s: float = 30 * 60,
    verify: bool | str | None = None,       # None/True = system CA; False = skip; str = pin PEM
    cache_dir: str | os.PathLike | bool | None = None,
)
```

Methods:

```python
Client.fit(
    real_data: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,    # default: every numeric column of real_data
    frequency: str | None = None,             # 'daily' | 'intraday' | 'weekly' | 'monthly' | pandas offset alias
    horizon: int | None = None,
    train_split: float | None = 0.8,          # set to None to skip the OOS split
    embargo_days: int = 21,
    seed: int | None = None,
    idempotency_key: str | None = None,
) -> FitResult

Client.generate(
    model_id: str,
    *,
    n_paths: int = 1000,
    horizon: int | None = None,               # any length; defaults to training horizon
    anchor_data: pd.DataFrame | None = None,  # None → use server-stored training tail
    like: pd.DataFrame | None = None,         # convenience: derive horizon + index + anchor from this window
    seed: int | None = None,
    idempotency_key: str | None = None,
) -> GenerationResult

Client.validate(
    model_id: str,
    *,
    holdout_data: pd.DataFrame | None = None, # None → use the OOS slice persisted at fit time
    n_paths: int = 500,
    seed: int | None = None,
    idempotency_key: str | None = None,
) -> ValidationReport

Client.list_models(*, limit: int = 50) -> list[Model]
Client.get_model(model_id: str) -> Model
Client.delete_model(model_id: str) -> None
```

### Module-level shortcuts

Core workflow:

```python
sf.fit(real_data, *, api_key=None,
       features=None, data_types,                     # data_types REQUIRED
       frequency=None, horizon=None,
       train_split=0.8, embargo_days=21, seed=None,
       idempotency_key=None,
       # connection-shape kwargs (env-var fallback) ───────────────
       endpoint=None, pinned_image_digest=None,
       attestation_mode="production", verify=None,
       cache_dir=None, profile="default") -> FitResult

sf.generate(model_id, *, api_key=None,
            data_types,                                # data_types REQUIRED
            n_paths=1000, horizon=None,
            anchor_data=None, like=None, seed=None,
            idempotency_key=None,
            endpoint=None, pinned_image_digest=None,
            attestation_mode="production", verify=None,
            cache_dir=None, profile="default") -> GenerationResult

sf.validate(model_id, *, api_key=None,
            data_types=None,                           # REQUIRED only when holdout_data is supplied
            holdout_data=None, n_paths=500, seed=None,
            idempotency_key=None,
            endpoint=None, pinned_image_digest=None,
            attestation_mode="production", verify=None,
            cache_dir=None, profile="default") -> ValidationReport
```

`data_types` is a `dict[str, str]` mapping every column in `features=` to one of `{'price', 'return', 'rate', 'index', 'volatility'}`. Missing the kwarg raises `TypeError` with the allowed-set message; an unknown value raises `ValueError`. Demo DataFrames attach the canonical map on `df.attrs['data_types']`. On `sf.validate(model_id)` without `holdout_data` the server reuses the `data_types` registered at fit time — passing the kwarg in that mode is a no-op and is silently ignored.

Async workflow:

```python
sf.fit_async(real_data, ...)        -> JobHandle      # same kwargs as sf.fit
sf.generate_async(model_id, ...)    -> JobHandle      # same kwargs as sf.generate
sf.validate_async(model_id, ...)    -> JobHandle      # same kwargs as sf.validate
sf.fetch_result(handle)             -> FitResult | GenerationResult | ValidationReport
sf.list_jobs(*, status=None, limit=50, api_key=None, **kw) -> list[JobSummary]
sf.cancel_job(handle_or_id, *, api_key=None, **kw)         -> None
```

Predictive validity (post-hoc analytic; pure numpy + scipy, no path generation):

```python
sf.predictive_rank_score(real_results, synth_results, *,
                         primary_metric=None,
                         n_bootstrap=10000,
                         seed=0) -> PredictiveRankReport
```

Auth + credentials:

```python
sf.login(*, endpoint=None, profile="default",
         open_browser=True, poll_timeout_s=600.0,
         verify=None) -> LoginResult
sf.logout(*, profile="default") -> bool                 # True if a profile was dropped
```

Model management:

```python
sf.list_models(*, limit=50, api_key=None, **kw) -> list[Model]
sf.get_model(model_id, *, api_key=None, **kw)   -> Model
sf.delete_model(model_id, *, api_key=None, **kw) -> None
```

Account / pre-flight:

```python
sf.ping(*, api_key=None, **kw)         -> dict[str, Any]
sf.whoami(*, api_key=None, **kw)       -> dict[str, Any]
sf.credits(*, api_key=None, **kw)      -> dict[str, Any]
sf.usage(*, since=None, until=None, kind=None, limit=100, api_key=None, **kw) -> list[dict]
sf.usage_summary(*, period="month", api_key=None, **kw) -> dict[str, Any]
sf.estimate_cost(kind, *, real_data=None, features=None, horizon=None, n_paths=None, api_key=None, **kw) -> dict[str, Any]
```

Local helpers (no network):

```python
sf.validate_data(real_data) -> None      # raise on schema violations BEFORE the network round-trip
sf.demo_data(name="us_equities_macro_2010_2024") -> pd.DataFrame
sf.available_demo_datasets() -> list[str]
```

Each shortcut constructs a one-shot `Client`. Connection-shape settings (`endpoint`, `verify`, `pinned_image_digest`, `attestation_mode`) fall back to env vars `SABLIER_FLOW_ENDPOINT`, `SABLIER_FLOW_CERT`, `SABLIER_FLOW_PINNED_IMAGE_DIGEST`, `SABLIER_FLOW_ATTESTATION_MODE`. `api_key` falls back to `SABLIER_FLOW_API_KEY`, then to `~/.sablier/credentials` (written by `sf.login()`).

Unknown kwargs raise `TypeError` with the offending name (no `**kwargs` swallow), so IDE autocomplete and `inspect.signature()` see the real parameter list.

### `JobHandle`

Returned by `sf.fit_async` / `sf.generate_async` / `sf.validate_async`. Persistable across processes via `to_dict()` / `from_dict(d)`.

```python
@dataclass(frozen=True)
class JobHandle:
    job_id: str
    kind: str                # 'fit' | 'generate' | 'validate'
    result_key_b64: str      # standard-base64 of the AES-256-GCM key — treat as a secret

    def to_dict(self) -> dict[str, str]: ...
    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "JobHandle": ...
```

Pair with `sf.fetch_result(handle)` to block on completion and materialize the typed result.

### `FitResult`

```python
@dataclass(frozen=True)
class FitResult:
    model_id: str
    features: list[str]
    training_horizon: int
    training_end_date: str | None
    sdk_version: str
    expires_at: str | None = None
    training_loss: float | None = None
    loss_source: str | None = None             # 'validation' | 'training_proxy'
    training_start_date: str | None = None
    holdout_start_date: str | None = None
    holdout_end_date: str | None = None
```

### `Model`

```python
@dataclass(frozen=True)
class Model:
    model_id: str
    features: list[str]
    training_horizon: int
    n_assets: int
    status: str                                # 'ready' | 'failed' | 'expired'
    training_start_date: str | None = None
    training_end_date: str | None = None
    holdout_start_date: str | None = None
    holdout_end_date: str | None = None
    train_split: float | None = None
    embargo_days: int | None = None
    sdk_version: str | None = None
    training_loss: float | None = None
    created_at: str | None = None
    last_used_at: str | None = None
    expires_at: str | None = None
```

### `GenerationResult`

```python
@dataclass(frozen=True)
class GenerationResult:
    paths_returns: np.ndarray                  # (n_paths, horizon, n_features), z-scored
    paths_prices: np.ndarray                   # (n_paths, horizon, n_features), price-level
    feature_names: list[str]                   # internal cyclical embeddings stripped
    last_prices: np.ndarray
    horizon: int
    n_paths: int
    seed: int | None
    sdk_version: str
    memorization_risk: str | None              # 'low' | 'medium' | 'high'
    memorization_nn_distance_ratio: float | None
    paths_index: pd.DatetimeIndex | None       # set when generate was called with like=window

    def as_dataframes(self, index=None) -> list[pd.DataFrame]: ...
```

### `ValidationReport`

```python
@dataclass(frozen=True)
class ValidationReport:
    overall: str                               # 'pass' | 'warn' | 'fail'
    metrics: dict[str, Any]                    # per-metric breakdown (see "Structural validation")
    memorization_risk: str | None              # 'low' | 'medium' | 'high'
    memorization_nn_distance_ratio: float | None
    n_paths_used: int | None
    holdout: bool = False                      # True when validated against a held-out OOS slice
```

### `robustness`

```python
sf.robustness(
    real_result: float | dict[str, float],
    synthetic_results: Sequence[float | dict[str, float]],
    *,
    primary_metric: str | None = None,
    higher_is_better: bool = True,
) -> RobustnessReport
```

### `RobustnessReport`

```python
@dataclass(frozen=True)
class RobustnessReport:
    overfit_score: float
    verdict: Literal["robust", "borderline", "overfit", "highly_overfit"]
    primary_metric: str
    real_value: float
    synthetic_mean: float
    synthetic_median: float
    synthetic_std: float
    synthetic_min: float
    synthetic_max: float
    synthetic_p5: float
    synthetic_p25: float
    synthetic_p75: float
    synthetic_p95: float
    synthetic_ci_95: tuple[float, float]
    n_synthetic: int
    per_metric: dict[str, dict[str, float]]
    notes: list[str]
    synthetic_values: tuple[float, ...]        # raw per-path values
    higher_is_better: bool

    @property
    def acceptable(self) -> bool: ...          # True for 'robust' or 'borderline'
    def summary(self) -> str: ...
    def deflated_sharpe(self, *, strategy_returns=None, n_trials=1) -> DeflatedSharpeReport: ...
    def to_html(self, path=None, *, title="Robustness Report") -> str: ...
```

### `deflated_sharpe`

```python
sf.deflated_sharpe(
    *,
    observed_sr: float,
    synthetic_sharpes: Sequence[float] | np.ndarray,
    strategy_returns: Sequence[float] | np.ndarray | None = None,
    n_trials: int = 1,
    significance_level: float = 0.95,
) -> DeflatedSharpeReport
```

### `DeflatedSharpeReport`

```python
@dataclass(frozen=True)
class DeflatedSharpeReport:
    observed_sr: float
    n_trials: int
    realistic: float                           # DSR under Sablier synthetic-best-of-N null
    analytical: float                          # DSR under Bailey-LdP IID-Gaussian null
    expected_max_sr_realistic: float
    expected_max_sr_analytical: float
    threshold_sr_realistic: float              # SR needed for DSR=0.95 (realistic)
    threshold_sr_analytical: float             # ... (analytical)

    def to_dict(self) -> dict: ...
```

### `evaluate_family`

```python
sf.evaluate_family(
    strategies: Mapping[str, Callable[..., float | dict]],
    real_data: pd.DataFrame,
    *,
    n_paths: int = 100,
    primary_metric: str | None = None,
    higher_is_better: bool = True,
    pbo_cscv_splits: int = 16,                 # SDK floor; lower values under-detect overfit
    executor: Literal["serial", "thread"] = "serial",
    max_workers: int | None = None,
    progress: bool = False,
    raise_on_failure: bool = False,
    **fit_or_generate_kwargs,                  # features, horizon, seed, train_split, etc.
) -> FamilyReport
```

### `FamilyReport`

```python
@dataclass(frozen=True)
class FamilyReport:
    strategy_names: tuple[str, ...]
    primary_metric: str
    real_metrics: tuple[dict[str, float], ...]
    synthetic_metrics: tuple[tuple[dict[str, float], ...], ...]
    real_max_value: float
    real_argmax_strategy: str
    synthetic_max_values: np.ndarray           # (n_paths,) best-of-N per path
    deflated_sharpe: DeflatedSharpeReport
    pbo: float
    pbo_n_partitions: int
    pbo_cscv_splits: int
    n_paths: int
    per_strategy_real_metric: dict[str, float]
    per_strategy_overfit_score: dict[str, float]
    per_strategy_synthetic_median: dict[str, float]
    failures: tuple[str, ...]
    notes: tuple[str, ...]

    def summary(self) -> str: ...
    def most_overfit_variants(self, *, top: int = 5) -> list[tuple[str, float]]: ...
    def to_dict(self) -> dict: ...
```

### `probability_of_backtest_overfitting`

```python
sf.probability_of_backtest_overfitting(
    strategies: Mapping[str, Callable[..., float | dict]],
    real_data: pd.DataFrame,
    *,
    primary_metric: str = "sharpe",
    higher_is_better: bool = True,
    cscv_splits: int = 16,
    executor: Literal["serial", "thread"] = "serial",
    max_workers: int | None = None,
) -> tuple[float, int]                          # (pbo_value, n_partitions_used)
```

### `consistency_check`

```python
sf.consistency_check(
    realized_value: float,
    baseline: RobustnessReport | Sequence[float] | np.ndarray,
    *,
    higher_is_better: bool | None = None,
    significance_level: float = 0.95,
) -> ConsistencyReport
```

### `ConsistencyReport`

```python
@dataclass(frozen=True)
class ConsistencyReport:
    realized_value: float
    baseline_median: float
    baseline_p5: float
    baseline_p95: float
    baseline_min: float
    baseline_max: float
    verdict: Literal["consistent", "drifting", "out_of_distribution"]
    drift_score: float
    empirical_cdf: float
    higher_is_better: bool
    n_baseline_paths: int
    notes: list[str]

    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
```

### `PredictiveRankReport`

```python
@dataclass(frozen=True)
class PredictiveRankReport:
    spearman_rho: float                       # rank correlation real vs synth-forward
    p_value: float                            # scipy.stats.spearmanr two-sided p
    ci_95: tuple[float, float]                # bootstrap percentile CI (10000 resamples)
    n_strategies: int                         # intersection of {real_results, synth_results}
    mean_abs_metric_gap: float                # magnitude bias (rank can be right while abs is biased)
    primary_metric: str                       # 'sharpe' if dicts; 'value' if scalars
    real_values: dict[str, float]
    synth_values: dict[str, float]
    n_bootstrap: int
    notes: list[str]

    @property
    def verdict(self) -> Literal[
        "well_calibrated", "weakly_calibrated", "uncalibrated", "inverted",
    ]: ...
    @property
    def acceptable(self) -> bool: ...         # True if verdict ∈ {well_calibrated, weakly_calibrated}
    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
```

### `AttestationVerifier`, `envelope_encrypt`, `envelope_decrypt`

Lower-level primitives. Not needed for the standard workflow — `Client` invokes them internally on every request.

```python
sf.AttestationVerifier(
    *,
    expected_image_digest: str,
    expected_tee_type: str = "CONFIDENTIAL_SPACE_A3_H100",
    expected_hardware: str = "NVIDIA_H100",
    expected_measurements: dict[str, str] | None = None,
    mode: Literal["production", "fake-for-dev"] = "production",
    root_key_registry: Any = None,
    required_issuers: tuple[str, ...] = ("google", "nvidia"),
)

sf.envelope_encrypt(plaintext: bytes, recipient_pubkey: bytes) -> EnvelopeEncrypted
sf.envelope_decrypt(env: EnvelopeEncrypted, recipient_privkey: bytes) -> bytes
```

The two helper-returned dataclasses below are exported as well so callers
implementing a custom transport or doing manual attestation inspection
have a typed surface to work against. The standard workflow never touches
them — `Client` handles every quote and envelope internally.

```python
@dataclass(frozen=True)
class AttestationQuote:
    """The verified attestation token a Client receives from the TEE.
    Returned by ``AttestationVerifier.verify(...)`` only on successful
    verification; never constructed directly by user code."""
    protocol_version: int
    image_digest: str                          # SHA-256 of the TEE image (must match PINNED_IMAGE_DIGEST)
    tee_type: str                              # "CONFIDENTIAL_SPACE_A3_H100"
    hardware: str                              # "NVIDIA_H100"
    measurements: dict[str, str]
    issued_at: datetime
    expires_at: datetime
    ephemeral_pubkey: bytes                    # X25519 pubkey to envelope-encrypt the upload to
    signatures: tuple[AttestationSignature, ...]

    @classmethod
    def from_wire(cls, wire_bytes: bytes) -> "AttestationQuote": ...

@dataclass(frozen=True)
class EnvelopeEncrypted:
    """Output of ``sf.envelope_encrypt``: an X25519 ephemeral pubkey +
    AES-256-GCM ciphertext + nonce + auth tag. Wire-format
    serializable via ``to_bytes()`` / ``from_bytes()`` for upload + replay."""
    ephemeral_pubkey: bytes                    # 32-byte X25519 public key
    nonce: bytes                               # 12-byte AES-GCM nonce
    ciphertext: bytes                          # AES-256-GCM ciphertext (includes auth tag)

    def to_bytes(self) -> bytes: ...
    @classmethod
    def from_bytes(cls, raw: bytes) -> "EnvelopeEncrypted": ...
```

---

## Known limitations

FLOW v1 is honest about what it learns and what it doesn't. The
generator is a joint flow trained with the pure CFM objective — no
auxiliary tail or volatility-clustering losses (every variant we tried
made things worse; see the `flow_auxiliary_losses_failed` note in the
internal methodology log). Two structural shapes routinely fail to
fully transfer to the synthetic distribution even when training
converges cleanly:

- **Heavy-tailed inputs** — when the real series has fat tails (high
  realized kurtosis, frequent ≥4σ moves), the synthetic distribution
  often under-reproduces the extreme quantiles. Strategies whose edge
  lives in the tail (vol-of-vol, jump-driven, deep-OTM options) will
  see narrower synthetic distributions than the real one and an
  overfit verdict that leans optimistic. Always check
  `report.metrics['extreme.*']` before trusting a tail-dependent
  verdict.
- **Vol clustering on GARCH-like inputs** — when the realized volatility
  process has strong persistence (long-memory GARCH, regime-switching
  vol), the synthetic paths often miss part of the autocorrelation in
  squared returns. Strategies that explicitly trade vol regime
  (volatility breakout, vol-target, GARCH-aware sizing) should treat
  the verdict as a lower bound on overfitting and verify against
  `report.metrics['dynamics.*']`.

Both shapes show up in the structural-validation suite — `sf.validate`
flags them as `warn` or `fail` on the relevant metric groups before
you build a verdict on top. The rule of thumb: if `report.overall ==
'fail'` or any `extreme.*` / `dynamics.*` metric is `'fail'`, treat
the robustness verdict as informational, not load-bearing, on
tail/vol-cluster-dependent strategies.

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `ValueError: api_key is required` | `SABLIER_FLOW_API_KEY` not set and not passed as kwarg | `export SABLIER_FLOW_API_KEY=sk_live_...` |
| `ValueError: real_data.index must be a pd.DatetimeIndex` | DataFrame index is not date-like | `df.index = pd.to_datetime(df.index)` |
| `ValueError: real_data has N rows; need at least 200` | Training data too short | Use a longer history |
| `ValueError: real_data has non-numeric columns: [...]` | Non-numeric column present | Drop / convert before passing |
| `ValueError: synthetic_results is empty` | `robustness()` called with no synth results | Generate paths first via `sf.generate(...)` |
| `ValueError: deflated_sharpe is only defined for higher-is-better metrics` | Called `.deflated_sharpe()` on a drawdown-style report | DSR is for return-style metrics. Use the raw `overfit_score` for lower-is-better. |
| `TransportError: retry budget exhausted` | Network blip during status polling exceeded retry window | The worker may still be training — call `sf.list_models()` to recover the model_id and resume with `sf.validate(model_id)` |
| `AttestationVerificationError` | Image digest mismatch between SDK and TEE | The SDK release pins a TEE image digest; if they disagree either upgrade the SDK or wait for the matching server rollout |
| `RemoteJobError` | Worker failed inside the TEE | Re-raise carries the safe error message from the worker; inspect and retry if transient |

---

## Glossary

| Term | SDK name | Notes |
|---|---|---|
| **Sablier flow model** | (server-side, not exposed) | The trained model itself. Customer holds only an opaque `model_id`. |
| **Realistic null** | `DeflatedSharpeReport.realistic` | Empirical CDF of observed SR in synthetic best-of-N distribution. Regime-aware. |
| **Analytical IID-Gaussian null** | `DeflatedSharpeReport.analytical` | Closed-form Bailey-LdP (2014). Regime-blind. |
| **`E[max_n SR_n]`** | `DeflatedSharpeReport.expected_max_sr_*` | Both nulls. |
| **Deflated Sharpe Ratio (DSR)** | `RobustnessReport.deflated_sharpe()` / `DeflatedSharpeReport` | Significance test for SR under selection bias. |
| **Probability of Backtest Overfitting (PBO)** | `FamilyReport.pbo` / `sf.probability_of_backtest_overfitting()` | Bailey-Borwein-LdP-Zhu (2015) CSCV. |
| **CSCV partitions** | `FamilyReport.pbo_n_partitions` / `pbo_cscv_splits` | `C(S, S/2)` partitions where `S = pbo_cscv_splits`. |
| **Memorization NN-distance ratio** | `ValidationReport.memorization_nn_distance_ratio` | Synth-to-train / train-to-train NN distance. `> 0.80` = low risk; `0.50–0.80` = medium; `< 0.50` = memorisation. |
| **Structural-validation suite** | `ValidationReport.metrics` | Per-metric breakdown across calibration, distribution, dependence, dynamics, extreme. |
| **Embargo** | `embargo_days` kwarg on `fit` | Bar gap between train end and OOS start. |

---

## Versioning

The public API follows semantic versioning. Major releases (`X.0.0`) may
introduce breaking changes; minor (`X.Y.0`) and patch (`X.Y.Z`) releases
preserve backwards compatibility. The current version is exposed at
`sablier_flow.__version__`.
