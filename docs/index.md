# sablier-flow

> Synthetic alternative-history generation for backtest overfitting detection — and calibrated forward forecasting for deployment.

`sablier-flow` is a Python SDK that lets you run your existing backtest on **N alternative versions of the same data** — turning a single P&L number into a distribution of P&L curves. If a strategy works on real history but falls apart on synthetically-generated alternatives that share the same statistical properties, it's overfit. If it holds up, you have real evidence.

A purpose-built generative model is trained on your data on hosted GPUs over an envelope-encryption + image-digest-pinning wire protocol; your backtest engine doesn't change. The hosted service is in **alpha** today — confidential-compute substrate is on the roadmap, not yet live; see [Security posture](SDK.md#security-posture-today-alpha) for the current threat model.

```python
import sablier_flow as sf

sf.login()                                                    # device-auth flow → ~/.sablier/credentials

real_data       = sf.demo_data()                              # bundled SPY/QQQ/IWM/TLT + VIX/TNX/DXY, 2010-2023
backtest_window = real_data.iloc[-252:]                       # the slice your strategy will evaluate against

def my_backtest(prices):                                       # YOUR backtest, unchanged
    rets = prices['SPY'].pct_change().dropna()
    return {'sharpe': float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0.0}

fit     = sf.fit(real_data,
                 features=list(real_data.columns),
                 data_types=real_data.attrs['data_types'],     # required: per-column annotation
                 horizon=252)
report  = sf.validate(fit.model_id)                            # zero-config OOS structural check
paths   = sf.generate(fit.model_id, n_paths=1000, like=backtest_window)
verdict = sf.robustness(
    my_backtest(backtest_window),
    [my_backtest(df) for df in paths.as_dataframes()],
    primary_metric="sharpe",
)
print(verdict.summary())
```

## Get started

- **[Quickstart](quickstart.md)** — `pip install` to overfit verdict in 5 minutes
- **[SDK reference](SDK.md)** — every method, kwarg, return type; the canonical reference
- **[Recipes](recipes.md)** — copy-pasteable patterns for common quant workflows
- **[Concepts](concepts/index.md)** — why in-sample training is correct, what predictive validity means, data sourcing, engine integration

## Worked examples — empirical demos with executed outputs

Each notebook ships with a download link at the top — click it to grab the `.ipynb` and run it locally against your account.

- **[Getting started](examples/00_getting_started.ipynb)** — end-to-end SDK tour: login → fit → validate → generate → robustness → forward forecast
- **[Backtest robustness](examples/01_backtest_robustness.ipynb)** — at the **0.7** `overfit_score` threshold, flags **29 of 30** selection-biased lucky strategies vs **0 of 12** honest false positives
- **[TSTR predictive rank](examples/02_tstr_predictive_rank.ipynb)** — Spearman ρ = **+0.7774**, 95% CI [+0.55, +0.89], p = 7.8e-06, n = 24
- **[Memorization audit](examples/03_memorization_audit.ipynb)** — NN-distance ratio **R = 0.93** vs replay-floor R = 0.02 (57.8× separation)

## What's distinctive

Synthetic financial paths have a two-axis quality definition: **distributional fidelity AND predictive-rank validity**. A generator that nails the marginals but inverts the strategy ranking is worse than useless for backtesting — a practitioner training a strategy family on it would systematically pick the worst real-market variant. The distributional metric suite alone does not catch this.

`sf.predictive_rank_score` runs the rank-validity check on your own model + strategy family: it computes Spearman ρ between the per-strategy Sharpe ranking on synthetic forward paths and the ranking on a realized OOS window, with a bootstrap 95% CI. You can quantify how much to trust forward forecasts on your universe before deploying capital on them, and reject any generator whose rank correlation crosses zero. The [TSTR predictive rank notebook](examples/02_tstr_predictive_rank.ipynb) walks the methodology end-to-end (TSTR = Train-on-Synthetic-Test-on-Real, the standard rank-validity protocol from the synthetic-data literature).
