<p align="center">
  <img src="https://raw.githubusercontent.com/sablier-ai/sablier-flow/main/docs/assets/logo.svg" alt="sablier-flow" width="180">
</p>

<h1 align="center">sablier-flow</h1>

<p align="center">
  <strong>Stop shipping overfit backtests.</strong><br>
  Run your strategy on <em>N alternative versions of history</em> that share your data's statistical fingerprint.
</p>

<p align="center">
  <a href="https://pypi.org/project/sablier-flow/"><img src="https://img.shields.io/pypi/v/sablier-flow.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/sablier-flow/"><img src="https://img.shields.io/pypi/pyversions/sablier-flow.svg" alt="Python versions"></a>
  <a href="https://github.com/sablier-ai/sablier-flow/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="https://docs.sablier.ai"><img src="https://img.shields.io/badge/docs-docs.sablier.ai-black" alt="Docs"></a>
</p>

---

## What it is

`sablier-flow` is a Python SDK that learns the joint dynamics of your market data — cross-asset correlations, vol clustering, regime structure — and generates **synthetic alternative histories** that share those statistics but produce different specific paths.

Run your existing backtest on **N alternative versions** of the same data and turn a single P&L number into a *distribution*. If your strategy only works on the one history that happened, that's overfit — now measurable.

## Install

```bash
pip install sablier-flow
```

Sign up at [sablier.ai](https://sablier.ai) for an API key (free starter credits cover the entire getting-started notebook).

## Quickstart

```python
import sablier_flow as sf
import numpy as np

# 1. Auth (one-time device flow; sets ~/.sablier/credentials)
sf.login()

# 2. Your backtest. Takes a price DataFrame, returns dict[str, float].
def my_backtest(prices):
    rets = prices['SPY'].pct_change().dropna()
    return {'sharpe': float(rets.mean() / rets.std() * np.sqrt(252))} if rets.std() > 0 else {'sharpe': 0.0}

# 3. Load data — bundled demo or your own DataFrame.
df = sf.demo_data()                          # SPY/QQQ/IWM/TLT + 3 macro features, 2010-2024
backtest_window = df.iloc[-252:]             # the slice you'll evaluate

# 4. Train + generate synthetic alternative versions of the backtest window.
fit   = sf.fit(df, features=list(df.columns), data_types=df.attrs['data_types'], horizon=252)
paths = sf.generate(fit.model_id, n_paths=200, like=backtest_window)

# 5. Run your backtest on each synth path and score robustness.
real_result   = my_backtest(backtest_window)
synth_results = [my_backtest(p) for p in paths.as_dataframes()]
report = sf.robustness(real_result, synth_results, primary_metric='sharpe')

print(report.summary())
```

## Examples

Live empirical demos with executed outputs baked in — open them on GitHub and see the numbers immediately:

| Notebook | What it proves |
|---|---|
| [📓 **Backtest Robustness**](examples/01_backtest_robustness.ipynb) | At the **0.7** `overfit_score` threshold: flags **29 of 30** selection-biased lucky strategies (top 30 of a 500-strategy pure-noise search; max honest = 0.725, min lucky = 0.695) vs **1 of 12** false positives on a designed honest family |
| [📓 **TSTR Predictive Rank**](examples/02_tstr_predictive_rank.ipynb) | Spearman ρ = **+0.74**, 95% CI **[+0.55, +0.83]**, p < 1e-4 — synth ranks predict real OOS ranks |
| [📓 **Memorization Audit**](examples/03_memorization_audit.ipynb) | NN-distance ratio **R = 0.9309** vs replay-floor R = 0.0161 — **57.8× separation**, synth is genuinely new |
| [📓 **Getting Started**](examples/00_getting_started.ipynb) | End-to-end SDK tour: login → fit → validate → generate → robustness |

## Why use it

- **One call** trains a model that handles cross-asset dependence, regime structure, and tail behavior — no hand-rolled copulas, no block-length tuning
- **Per-strategy overfit detection** that classical CSCV-PBO can't surface (selection bias from parameter searches)
- **Train on synth, deploy on real** — `sf.predictive_rank_score` proves the ranking carries forward, so you don't have to burn real OOS data on strategy selection
- **Engine-agnostic**: works with pandas, backtrader, vectorbt; LEAN/QuantConnect adapter in 1.0.x

## Docs

- [**docs.sablier.ai**](https://docs.sablier.ai) — full SDK reference, recipes, concepts
- [Quickstart](https://docs.sablier.ai/quickstart) — `pip install` to first verdict in 5 minutes
- [Concepts: why in-sample training is correct](https://docs.sablier.ai/concepts/in-sample-is-correct/)
- [Concepts: data sourcing + engine integration](https://docs.sablier.ai/concepts/data-sourcing/)

## License

Apache 2.0 (code) · CC BY 4.0 (docs)
