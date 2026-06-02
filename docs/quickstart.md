# Quickstart — from `pip install` to overfit verdict in 5 minutes

The actual workflow a quant follows from a clean machine to a real verdict.

## 0. Sign up + verify email (one-time)

1. Go to [sablier.ai](https://sablier.ai) → **Sign up**. Email/password or "Sign in with Google" — both work.
2. **Click the verification link** in the email Sablier sends. (Skip-able with Google OAuth — Google has already verified your email.)
3. New accounts receive free credits — enough to fit + validate + generate against the bundled demo dataset a couple of times to see the full loop end-to-end.

You only do this once per account. Subsequent machines authenticate via `sf.login()` (next section).

## 1. Install

```bash
# Thin client (~30 MB, no GPU deps)
pip install sablier-flow

# Plus your engine of choice
pip install 'sablier-flow[adapters-backtrader]'    # backtrader
pip install 'sablier-flow[adapters-vectorbt]'      # vectorbt
# LEAN works via write_lean_csv_universe — no extra deps
```

## 2. Authenticate

The recommended path is interactive — `sf.login()` does an OAuth-style device flow and writes the minted API key to `~/.sablier/credentials`:

```python
import sablier_flow as sf

sf.login()                      # prints a code + opens https://sablier.ai/auth/device
                                # click Authorize on any signed-in device; the SDK picks the key up
client = sf.Client()            # no api_key kwarg needed — credentials file does it
```

For CI / containers, set `SABLIER_FLOW_API_KEY=sk_live_...` in the environment instead; the SDK reads env vars before the credentials file. Explicit `api_key=` kwarg always wins.

## 3. The end-to-end loop — canonical workflow

Copy-pasteable against `sf.demo_data()` — runs against the hosted API
with no setup beyond `sf.login()` from step 2:

```python
import numpy as np
import sablier_flow as sf

df              = sf.demo_data()                       # SPY/QQQ/IWM/TLT + macro, 2010-2023
backtest_window = df.iloc[-21:]                        # the slice your strategy will evaluate

def my_backtest(prices):                               # YOUR backtest, unchanged
    rets = prices['SPY'].pct_change().dropna()
    return {'sharpe': float(rets.mean() / rets.std() * np.sqrt(252))
            if rets.std() > 0 else 0.0}

fit  = sf.fit(df,
              features=list(df.columns),
              data_types=df.attrs['data_types'],       # REQUIRED — per-column annotation
              horizon=21)
gen  = sf.generate(fit.model_id, n_paths=100, like=backtest_window)
synth_results = [my_backtest(d) for d in gen.as_dataframes()]
verdict = sf.robustness(my_backtest(backtest_window),  # same 21-bar window on both sides
                        synth_results,
                        primary_metric='sharpe')
print(verdict.summary())
```

`gen.as_dataframes()` (see `GenerationResult.as_dataframes`) returns
`list[pd.DataFrame]`, one per synthetic alternative-history path, with
the same columns as `df` and (when `like=` was passed) the same index
shape as the window you handed in — your existing `my_backtest`
function runs on each `d` unchanged.

> **Symmetric window matters.** Both `my_backtest(backtest_window)` and
> each `my_backtest(d)` evaluate on the same 21-bar window. Comparing
> the real Sharpe over the full 3500-bar `df` against synth Sharpes over
> 21-bar windows is asymmetric and mechanically produces
> `'highly_overfit'` — a 167× sample-size difference, not a real signal.

The variant below adds `sf.validate(...)` for a cheap OOS structural
check and uses a longer history + named backtest window — same loop,
just more explicit. Replace `sf.demo_data()` with `pd.read_parquet(...)`
or any other DataFrame loader when you're ready to use your own data:

```python
import numpy as np
import pandas as pd
import sablier_flow as sf

real = pd.read_parquet("my_universe.parquet")            # YOUR DataFrame, DatetimeIndex
real.attrs['data_types'] = {col: 'price' for col in real.columns}    # per-column annotation
backtest_window = real.loc["2023-01-01":"2024-01-01"]    # the slice you'll evaluate

fit    = sf.fit(real,
                features=list(real.columns),
                data_types=real.attrs['data_types'],
                horizon=252, seed=42)                    # trains the joint model
report = sf.validate(fit.model_id)                       # cheap OOS check
paths  = sf.generate(fit.model_id, n_paths=1000, like=backtest_window)     # synthetic alternative histories
verdict = sf.robustness(my_backtest(backtest_window),
                        [my_backtest(df) for df in paths.as_dataframes()],
                        primary_metric="sharpe")
```

`my_backtest(df) -> dict` is **your existing code**. It returns a dict containing at least the primary metric (`sharpe`, `return`, whatever). Anything that runs on the real DataFrame runs unchanged on a synthetic one — same columns, same index, same dtype.

## 4. The verdict

```python
print(verdict.summary())                 # plain-English one-liner
print(verdict.verdict)                   # 'robust' | 'borderline' | 'overfit' | 'highly_overfit'
print(verdict.overfit_score)             # 0.04 = real beat only 4% of alt-histories
print(verdict.synthetic_median)          # +0.51 — typical Sharpe across alt-histories
print(verdict.synthetic_p5, verdict.synthetic_p95)
```

### Verdict bands

| Band | `overfit_score` | What it means |
|---|---|---|
| `robust` | `< 0.70` | Real result is consistent with the synthetic distribution. **No overfit signal** — but read the value sign separately: a `robust` Sharpe of `-1.2` means "consistently bad, not overfit." |
| `borderline` | `0.70 – 0.85` | Real result is in the upper synthetic decile. Tighten parameters; some luck baked in. |
| `overfit` | `0.85 – 0.95` | Real result is in the top 5–15% of synthetic outcomes. Probably curve-fitting. |
| `highly_overfit` | `> 0.95` | Real beats essentially every synthetic alternative. Don't deploy live without out-of-sample revalidation. |

The `summary()` string makes the "robust ≠ profitable" distinction explicit when real is outside the synth 5–95 CI.

## 5. Sanity-check the synthetic data

```python
print(report.overall)               # 'pass' / 'warn' / 'fail' — weighted structural-validation verdict
print(report.memorization_risk)     # 'low' / 'medium' / 'high'
print(report.memorization_nn_distance_ratio)
```

`memorization_risk='high'` means the model is reproducing training samples — the overfit verdict above would be unreliable in that case. `memorization_nn_distance_ratio` can sit in the borderline 0.8–1.0 range for universes with > 10 jointly-modeled columns even when the model is fine; partition large universes into per-regime / per-asset-class sub-models if you see it.

## 6. Forward forecasting — same workflow, future-looking data

Everything above frames the SDK around backtest augmentation (synth paths parallel to a past window). The same generator also runs **forward** from your most recent bar — useful for predicting the distribution of strategy performance you'll see in deployment.

Same `fit` → same `generate` → same backtest function. Only the anchor moves to "today":

```python
# Anchor forward generation at real.index[-1] by passing the recent tail as anchor_data.
# The default 80/20 fit split is preserved so sf.validate(model_id) still works.
forward_paths = sf.generate(
    fit.model_id,
    n_paths=1000,
    horizon=60,                      # bars to project forward
    anchor_data=real.iloc[-200:],    # last 200 bars = today's conditioning context
)

forward_dfs = forward_paths.as_dataframes()
forward_sharpes = np.array([my_backtest(df)["sharpe"] for df in forward_dfs])

print(f"expected sharpe (next 60 bars): {np.median(forward_sharpes):+.2f}")
print(f"90% CI: [{np.percentile(forward_sharpes, 5):+.2f}, "
      f"{np.percentile(forward_sharpes, 95):+.2f}]")
```

### How much to trust the forecast — `sf.predictive_rank_score`

A generator that nails the marginals but inverts the strategy ranking is worse than useless for backtesting — a practitioner training a strategy family on it would systematically pick the worst real-market variant. Distributional metrics alone do not catch this. `sf.predictive_rank_score` runs the rank-validity check directly on the customer's own model + strategy family:

```python
import numpy as np

# Define your strategy family. >= 20 variants is the recommended floor
# for a stable Spearman ρ; the demo here uses an SMA-crossover grid.
def sma_crossover_backtest(df, fast, slow):
    px = df['SPY']
    fast_ma = px.rolling(fast).mean()
    slow_ma = px.rolling(slow).mean()
    pos  = (fast_ma > slow_ma).shift(1, fill_value=False).astype(int)
    rets = px.pct_change().fillna(0.0) * pos
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    return {'sharpe': sharpe}

strategies = {
    f'sma_{fast}_{slow}': (lambda df, f=fast, s=slow: sma_crossover_backtest(df, f, s))
    for fast in (3, 5, 7, 10, 15, 20)
    for slow in (20, 30, 50, 100)
}

# Use the slice sf.fit held out at train time as the OOS reference so the
# calibration runs on truly unseen data. A naive last-N-row slice would
# overlap the server's held-out OOS slice (last 20% minus embargo) and
# bias the rank correlation upward.
real_oos       = real.loc[fit.holdout_start_date:fit.holdout_end_date]
anchor_end_pos = real.index.get_indexer([real_oos.index[0]])[0]
ref_anchor     = real.iloc[anchor_end_pos - 200:anchor_end_pos]

forward_paths = sf.generate(fit.model_id, n_paths=200, horizon=len(real_oos),
                             anchor_data=ref_anchor,
                             data_types=real.attrs['data_types'])

real_sharpes  = {name: fn(real_oos)["sharpe"] for name, fn in strategies.items()}
synth_sharpes = {
    name: float(np.mean([fn(df)["sharpe"] for df in forward_paths.as_dataframes()]))
    for name, fn in strategies.items()
}

score = sf.predictive_rank_score(real_sharpes, synth_sharpes)
print(score.summary())
print(score.verdict)   # 'well_calibrated' | 'weakly_calibrated' | 'uncalibrated' | 'inverted'
```

If `score.verdict == "inverted"`, do not deploy on the forward-forecast ranking — the model is misranking strategies on your universe.

See [`SDK.md`](SDK.md#forward-generation-deployment-forecasting) for the full recipe + caveats.

## 7. Async + cross-process workflows

For long fits you may not want to block the kernel:

```python
handle = sf.fit_async(real, features=list(real.columns), horizon=252)
# ... do other work, restart the kernel, walk away

# Later (same or different process):
fit = sf.fetch_result(handle)
```

Check progress at any time from any process:

```python
sf.list_jobs(limit=10)             # most-recent first; each row has
                                   # status, progress (dict), last_progress_at
```

`progress` is a dict like `{'step': N, 'total_steps': M, 'phase': 'training', 'message': '...', 'metrics': {...}}`. `last_progress_at` is the wall-clock heartbeat — useful for detecting a stuck job.

Persist the handle across processes:

```python
import json
json.dump(handle.to_dict(), open("job-handle.json", "w"))

# Different machine / Python interpreter:
handle = sf.JobHandle.from_dict(json.load(open("job-handle.json")))
fit = sf.fetch_result(handle)
```

See what's in flight + cancel a stuck job:

```python
sf.list_jobs(status="running")
sf.cancel_job(handle)              # or pass a raw job_id string
```

The handle holds the one-shot AES key that decrypts the result — treat it like a secret.

## What's actually happening on the wire

```
your laptop ──HTTPS──> Sablier API ────────────────────────────> GPU worker
     │                                                                   │
     │   1. POST /v1/jobs                                                 │
     │   ◄── 2. ephemeral X25519 pubkey + pinned image digest             │
     │                                                                    │
     │   3. envelope-encrypt your DataFrame (X25519 + AES-256-GCM)        │
     │   ──> PUT /v1/jobs/{id}/data ─────────────────────────────────────►│
     │                                                                    │
     │                                          4. decrypt in worker RAM, │
     │                                             train + generate,      │
     │                                             AES-GCM-encrypt back   │
     │                                                                    │
     │   5. GET /v1/jobs/{id}/result ◄───────────────────────────────────-│
     │   6. decrypt locally                                               │
     ▼
backtester
```

See [Security posture](SDK.md#security-posture-today-alpha) for the threat model if you need it.

## Bundled demo dataset — `sf.demo_data()`

```python
import sablier_flow as sf

real = sf.demo_data()                              # daily SPY/QQQ/IWM/TLT + 3 macro series, 2010-2023
print(real.shape, real.attrs['data_types'])        # df.attrs carries the per-column annotation

# 5-min intraday alternative — same shape, same data_types contract, intraday cadence.
# intraday = sf.demo_data('us_equities_macro_5min_3mo')
```

`sablier_flow.demo_data()` returns a clean aligned `pd.DataFrame` from a parquet bundled inside the wheel — no third-party data feed required. You still need an API key (the `fit`/`generate`/`validate` calls reach the hosted service); the data load itself is offline.

## Next

- [`SDK.md`](SDK.md) — full reference with every method, kwarg, and return type.
- [`recipes.md`](recipes.md) — copy-pasteable patterns for common quant workflows.
- [Examples gallery](examples/00_getting_started.ipynb) — full set of runnable notebooks:
  - **Getting started** — end-to-end SDK tour (login, fit, validate, generate, robustness, async, management)
  - **Backtest robustness** — catch lucky-overfit strategies via per-strategy overfit_score on a family
  - **TSTR predictive rank** — verify synth-ranks predict real-OOS-ranks (Spearman ρ + CI)
  - **Memorization audit** — confirm synth is genuinely new vs replay-memorizer baseline
