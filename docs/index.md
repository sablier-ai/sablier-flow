# sablier-flow

> Synthetic alternative-history generation for backtest overfitting detection — and calibrated forward forecasting for deployment.

`sablier-flow` is a Python SDK that lets you run your existing backtest on **N alternative versions of the same data** — turning a single P&L number into a distribution of P&L curves. If a strategy works on real history but falls apart on synthetically-generated alternatives that share the same statistical properties, it's overfit. If it holds up, you have real evidence.

A purpose-built generative model is trained on your data inside a hardware-attested confidential GPU; the data never leaves that enclave; your backtest engine doesn't change.

```python
import sablier_flow as sf

sf.login()                                        # device-auth flow → ~/.sablier/credentials

fit     = sf.fit(real_data, features=[...], horizon=252)
report  = sf.validate(fit.model_id)               # zero-config OOS structural check
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

## What's distinctive

Synthetic financial paths have a two-axis quality definition: **distributional fidelity AND predictive-rank validity**. A generator that nails the marginals but inverts the strategy ranking is worse than useless for backtesting — a practitioner training a strategy family on it would systematically pick the worst real-market variant. The distributional metric suite alone does not catch this.

`sf.predictive_rank_score` runs the rank-validity check on your own model + strategy family: it computes Spearman ρ between the per-strategy Sharpe ranking on synthetic forward paths and the ranking on a realized OOS window, with a bootstrap 95% CI. You can quantify how much to trust forward forecasts on your universe before deploying capital on them, and reject any generator whose rank correlation crosses zero.
