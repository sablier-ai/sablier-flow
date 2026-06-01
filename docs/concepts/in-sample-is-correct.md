# Why in-sample training is correct

The natural objection to `sablier-flow` is that we train the generator on the *same* historical window the customer's backtest uses. Isn't that contaminating the test?

**No.** This page explains why — and what the one technical concern is (memorization vs generalization).

## The argument in one sentence

The model is the thing that saw the data; the strategy isn't. When the strategy is applied to synthetic samples, it's evaluating new data points that share the statistical properties of the period — not the historical points it was already tuned on. A strategy that overfit to specific historical noise will fail on synthetic alternative versions, because that specific noise is by construction not present in the learned distribution.

## What the literature says

Every serious treatment of synthetic data for backtest robustness uses in-sample training:

- **[López de Prado & Bailey (2014)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)** — Combinatorially Symmetric Cross-Validation (CSCV), Probability of Backtest Overfitting (PBO), and the [Deflated Sharpe Ratio (Bailey & López de Prado, 2014)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) all fit models to historical data and generate Monte Carlo paths from the learned distribution to characterize strategy variance.

- **[AWS HPC blog (2024)](https://aws.amazon.com/blogs/hpc/enhancing-equity-strategy-backtesting-with-synthetic-data-part-1-overview-and-toolkit-for-strategy-evaluation/)** — the "Enhancing Equity Strategy Backtesting with Synthetic Data" series uses agent-based models fit to historical equity markets to generate alternative paths for the same period.
- **[TimeGAN (Yoon, Jarrett, van der Schaar, NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html)**, **[TSGBench (Ang et al., VLDB 2024)](https://www.vldb.org/pvldb/vol17/p305-ang.pdf)**, and **signature-kernel methods** (e.g. [Salvi et al., 2020](https://arxiv.org/abs/2006.14794)) — every standard generative-model benchmark for financial time series trains the model on the data being analyzed.

Nobody holds out a separate period to train a generator. The shared methodology across literature and industry is the same: fit, sample, evaluate.

## Why a strategy can't "cheat" through the synthetic data

Consider two strategies on the same backtest data:

**Strategy A (overfit)** — discovered that buying at 9:33am Tuesdays in 2019 worked. Its parameters encode that specific calendar regularity.

**Strategy B (robust)** — exploits genuine momentum that persists across regimes. Its parameters encode a statistical pattern.

When we train FLOW on the same 2019 data:

- FLOW learns the **distribution** — moments, dependence, vol clustering, leverage. It does NOT learn "September 17, 2019 had this specific return". It learns "returns of this magnitude occur with this frequency, conditional on these features".
- When FLOW samples alternative versions, "9:33am Tuesday in 2019" is just another timestamp drawn from the joint distribution. The specific historical sequence isn't reproduced.

Run Strategy A on 1,000 synthetic versions:
- The specific quirk it's tuned to doesn't appear in any of them.
- A's average Sharpe across 1,000 paths is much lower than its real-backtest Sharpe.
- We flag it as overfit.

Run Strategy B on 1,000 synthetic versions:
- The momentum signal IS in the joint distribution, so it's present in synthetic samples too.
- B's average Sharpe across synthetic paths is comparable to its real-backtest Sharpe.
- We flag it as robust.

The strategy never sees the training data through the synthetic samples. The synthetic samples are new draws from a smooth approximation of the data-generating process.

## The one technical caveat — memorization

The above only holds if **the generator generalizes rather than memorizes**.

If FLOW's parameters happened to encode "September 17, 2019 had this specific return", then the synthetic samples would reproduce 2019's specific noise. Strategy A's calendar quirk would appear in synthetic data. We'd falsely flag it as robust.

This is a real phenomenon — strong diffusion models on images can memorize and re-emit training samples when capacity vastly exceeds the data manifold. But for financial returns it's structurally unlikely:

1. **Returns space, not raw price levels.** Lower dimensional, fewer addressable points.
2. **Limited capacity vs data.** The architecture today is a few-million-parameter conditional flow-matching model trained on thousands of days × tens-to-hundreds of features — small by foundation-model standards. Vision diffusion models that demonstrably memorize are billions of parameters trained on billions of images, a different regime entirely (see e.g. Carlini et al., [Extracting Training Data from Diffusion Models](https://arxiv.org/abs/2301.13188), 2023, which required ≳100M-image scale and explicit attack queries).
3. **Z-scoring + the model's internal smoothing act as information bottlenecks.** The model can't address training points "by date" — no date input.
4. **The existing validation suite enforces stylized-fact matching.** A generator that just regurgitated training samples would fail those tests.

To make this empirically verifiable, every `sablier-flow` model reports a **memorization-risk verdict** computed at validation time:

- **NN-distance ratio** = median synthetic-to-training nearest-neighbor distance / median training-to-training nearest-neighbor distance. The SDK reports the raw ratio and a banded verdict:
  - **`'low'` risk** (verdict): ratio in the `0.85 – 1.15` healthy band — synth distributed through the training manifold at training-like density, near-perfect on the population score.
  - **`'medium'` risk**: ratio outside that band but above ~0.5 — synth is structurally tighter than training; cross-check **`coverage_*`** (the per-bin empirical-coverage metrics returned alongside, e.g. `coverage_0.5 / coverage_0.9 / coverage_0.95` — fractions of real samples falling inside the corresponding synthetic-quantile intervals).
  - **`'high'` risk**: ratio < ~0.5 — synth closer to training than training is to itself. Memorization detected.

The full verdict is exposed as `ValidationReport.memorization_risk` ∈ `{"low", "medium", "high"}` on the result of `sf.validate(model_id)`, plus the raw ratio at `ValidationReport.memorization_nn_distance_ratio`. The SDK does not gate `generate(...)` on this verdict — the customer reads it and decides whether to keep using the model, retrain on a smaller in-sample slice, or partition the universe into per-asset-class sub-models. The signal is the falsifiability mechanism; the action is the customer's.

## When you want a held-out FLOW fit

The default works for the typical case: a rule-based or factor strategy with a small parameter count. The strategy didn't itself "see" the training data in any meaningful sense — its parameters were chosen by the researcher.

The exception: **the strategy itself is an ML model trained on the same window.** If both FLOW and a deep-neural-net strategy were trained on 2010-2023, both could have learned the same noise features, and FLOW's synthetic samples would falsely confirm the strategy's robustness.

For these customers the recipe is to slice the DataFrame manually and call `sf.fit` only on the window the strategy hasn't touched — e.g. `fit_train = real.loc[:'2018']` for a strategy validated on 2019-2023. The synthetic distribution then provides a clean OOS evaluation. The tradeoff: less statistical power (smaller training window) and the customer carries the responsibility for picking a defensible held-out range.

## Summary

| Concern | Resolution |
|---|---|
| "Isn't training on the same period contamination?" | No — the strategy never sees training data; only new samples from the learned distribution. |
| "What if the generator memorizes?" | Reported as `ValidationReport.memorization_risk` after `sf.validate(model_id)`; partition large universes or retrain on a smaller slice if `'high'`. |
| "What about ML-trained strategies?" | Fit FLOW on a window your ML strategy didn't see (`sf.fit(real.loc[:'2018'], ...)`). |
| "Is this standard?" | Yes — [López de Prado & Bailey (2014)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253), [AWS HPC (2024)](https://aws.amazon.com/blogs/hpc/enhancing-equity-strategy-backtesting-with-synthetic-data-part-1-overview-and-toolkit-for-strategy-evaluation/), [TimeGAN (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html), [TSGBench (VLDB 2024)](https://www.vldb.org/pvldb/vol17/p305-ang.pdf) all use in-sample training. |
