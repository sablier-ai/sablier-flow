# Quickstart — `pip install` to verdict in 5 minutes

## 1. Install + sign up

```bash
pip install sablier-flow

# Optional engine extras
pip install 'sablier-flow[adapters-backtrader]'
pip install 'sablier-flow[adapters-vectorbt]'
```

Sign up at [sablier.ai](https://sablier.ai) (email/password or Google OAuth — both work; verify your email if you used password). New accounts get free credits, enough to run the full loop against the bundled demo dataset.

## 2. Authenticate

```python
import sablier_flow as sf
sf.login()        # opens browser, prompts Authorize, writes ~/.sablier/credentials
```

For CI / containers: `export SABLIER_FLOW_API_KEY=sk_live_...` and skip `sf.login()`.

## 3. The end-to-end loop

```python
import numpy as np
import sablier_flow as sf

df              = sf.demo_data()                       # SPY/QQQ/IWM/TLT + macro, 2010-2023
backtest_window = df.iloc[-252:]                       # the slice your strategy will evaluate

def my_backtest(prices):                               # YOUR backtest, unchanged
    rets = prices['SPY'].pct_change().dropna()
    return {'sharpe': float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0}

fit     = sf.fit(df, features=list(df.columns), data_types=df.attrs['data_types'], horizon=252)
report  = sf.validate(fit.model_id)                    # cheap OOS structural check
paths   = sf.generate(fit.model_id, n_paths=1000, like=backtest_window)
verdict = sf.robustness(
    my_backtest(backtest_window),
    [my_backtest(d) for d in paths.as_dataframes()],
    primary_metric='sharpe',
)
print(verdict.summary())
```

> **Symmetric window.** `my_backtest(backtest_window)` and each `my_backtest(d)` must evaluate the same window length. Comparing real Sharpe on full history against synth Sharpes on a 252-bar slice is asymmetric and mechanically produces `'highly_overfit'`.

## 4. Reading the verdict

```python
print(verdict.verdict)                   # 'robust' | 'borderline' | 'overfit' | 'highly_overfit'
print(verdict.overfit_score)             # 0.04 = real beat only 4% of alt-histories
print(verdict.synthetic_p5, verdict.synthetic_p95)
print(report.overall, report.memorization_risk)
```

| Band | `overfit_score` | Meaning |
|---|---|---|
| `robust` | `< 0.70` | No overfit signal. Read the Sharpe sign separately — `robust` ≠ profitable. |
| `borderline` | `0.70 – 0.85` | Real in top quartile of synth; defensible but not unambiguous. |
| `overfit` | `0.85 – 0.95` | Likely curve-fit. |
| `highly_overfit` | `> 0.95` | Don't deploy without OOS re-validation. |

If `report.memorization_risk == 'high'`, the verdict above isn't reliable — see [SDK reference](SDK.md#interpreting-the-output).

## 5. Forward forecasting

Same generator, anchored at "today":

```python
forward = sf.generate(fit.model_id, n_paths=1000, horizon=60, anchor_data=df.iloc[-200:])
forward_sharpes = np.array([my_backtest(d)['sharpe'] for d in forward.as_dataframes()])
print(f"median: {np.median(forward_sharpes):+.2f}, 90% CI: "
      f"[{np.percentile(forward_sharpes, 5):+.2f}, {np.percentile(forward_sharpes, 95):+.2f}]")
```

To validate that synth rankings predict real rankings across a strategy family, use `sf.predictive_rank_score(real_sharpes, synth_sharpes)` — see [notebook 02](examples/02_tstr_predictive_rank.ipynb).

## 6. Async + cross-process

```python
handle = sf.fit_async(df, features=list(df.columns), data_types=df.attrs['data_types'], horizon=252)
# ... walk away, restart kernel, whatever
fit = sf.fetch_result(handle)         # blocks until done
sf.list_jobs(status='running')
```

`handle.to_dict()` / `JobHandle.from_dict(...)` persist across processes. Treat the handle as a bearer secret.

## Next

- [Examples](examples/00_getting_started.ipynb) — full tutorial + three value-proof notebooks
- [SDK reference](SDK.md) — every method, kwarg, return type
