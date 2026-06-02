# Examples

Four executed notebooks demonstrating `sablier-flow` end-to-end. Each is a complete, runnable case — clone the repo or download the individual `.ipynb` from the link in each notebook's top banner.

## Where to read them

There are two ways to view a notebook in this folder:

1. **Rendered (recommended): [docs.sablier.ai/examples](https://docs.sablier.ai/examples/00_getting_started/)** — server-rendered via `mkdocs-jupyter`, no client-side timeout, full ToC, dark mode. Always works.
2. **Source on GitHub** — click any `.ipynb` file below. When GitHub's notebook viewer (`notebooks.githubusercontent.com`) is up, the file renders inline; when it's having an outage (it returns HTTP 500 across the platform during rendering-service incidents), use the docs link.

`git clone` + `jupyter notebook examples/` runs them locally with full interactivity.

## The notebooks

| File | Rendered | Headline result |
|---|---|---|
| [`00_getting_started.ipynb`](00_getting_started.ipynb) | [docs.sablier.ai](https://docs.sablier.ai/examples/00_getting_started/) | End-to-end SDK tour: login → fit → validate → generate → robustness → forward forecast → predictive-rank calibration |
| [`01_backtest_robustness.ipynb`](01_backtest_robustness.ipynb) | [docs.sablier.ai](https://docs.sablier.ai/examples/01_backtest_robustness/) | At `overfit_score` threshold 0.7: catches **29 of 30** lucky strategies (lucky family min = 0.670, max = 0.875) vs **0 of 12** honest false positives (honest max = 0.690) |
| [`02_tstr_predictive_rank.ipynb`](02_tstr_predictive_rank.ipynb) | [docs.sablier.ai](https://docs.sablier.ai/examples/02_tstr_predictive_rank/) | Spearman ρ = **+0.7774**, 95% bootstrap CI [+0.55, +0.89], p = 7.8e-06, n = 24 — synth ranks predict real OOS ranks |
| [`03_memorization_audit.ipynb`](03_memorization_audit.ipynb) | [docs.sablier.ai](https://docs.sablier.ai/examples/03_memorization_audit/) | NN-distance ratio R = **0.9288** vs replay-floor R = 0.0161 — 57.7× separation, synth is genuinely new |

## Running locally

```bash
git clone https://github.com/sablier-ai/sablier-flow
cd sablier-flow
pip install sablier-flow jupyter matplotlib
sablier-flow login
jupyter notebook examples/
```

Each notebook anchors on the bundled demo dataset (`sf.demo_data()` — SPY/QQQ/IWM/TLT plus VIX/TNX/DXY macro, 2010–2023), so the only setup is `pip install` and `sablier-flow login`. Swap the demo for your own DataFrame whenever you're ready — every step works unchanged on any `pd.DataFrame` with a `DatetimeIndex`.

## Methodology + interpretation

For the verdict semantics (what `overfit_score` thresholds mean, why the two DSR nulls disagree on some strategies, how to read the memorization-risk band), see the [SDK reference's "Interpreting the output" section](https://docs.sablier.ai/SDK/#interpreting-the-output) and the [concepts pages](https://docs.sablier.ai/concepts/).
