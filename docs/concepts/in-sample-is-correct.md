# Why in-sample training is correct

The natural objection to `sablier-flow` is that we train the generator on the *same* historical window the customer's backtest uses. Isn't that contaminating the test?

**No.** This page explains why — and what the one technical concern is (memorization vs generalization).

## The argument in one sentence

The model is the thing that saw the data; the strategy isn't. When the strategy is applied to synthetic samples, it's evaluating new data points that share the statistical properties of the period — not the historical points it was already tuned on. A strategy that overfit to specific historical noise will fail on synthetic alternative versions, because that specific noise is by construction not present in the learned distribution.

## What the literature says

Every serious treatment of synthetic data for backtest robustness uses in-sample training:

- **López de Prado & Bailey** — Combinatorially Symmetric Cross-Validation (CSCV), Probability of Backtest Overfitting (PBO), and Deflated Sharpe Ratio all fit models to historical data and generate Monte Carlo paths from the learned distribution to characterize strategy variance.

- **AWS HPC blog (Q3 2024)** — the "Enhancing Equity Strategy Backtesting with Synthetic Data" series uses agent-based models fit to historical equity markets to generate alternative paths for the same period.
- **TimeGAN / TSGBench / signature-kernel methods** — generative-model evaluations for financial time series train the model on the data being analyzed.

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
2. **Limited capacity vs data.** ~5M parameters trained on thousands of days × tens-to-hundreds of features. Vision diffusion models that memorize are billions of parameters trained on billions of images — a different regime.
3. **Z-scoring + the model's internal smoothing act as information bottlenecks.** The model can't address training points "by date" — no date input.
4. **The existing validation suite enforces stylized-fact matching.** A generator that just regurgitated training samples would fail those tests.

To make this empirically verifiable, every `sablier-flow` model reports a **memorization-risk verdict** computed at validation time:

- **NN-distance ratio** = median synthetic-to-training nearest-neighbor distance / median training-to-training nearest-neighbor distance.
  - `> 0.80` — generator generalizing well (synth distributed through the training manifold at training-like density).
  - `0.50 – 0.80` — borderline (synth tighter than training; cross-check coverage_*).
  - `< 0.50` — pathological (synth closer to training than training is to itself). Memorization detected.

The full verdict is exposed as `MemorizationReport.risk` ∈ `{"low", "medium", "high"}`. The SDK refuses to return synthetic samples when risk is `high` unless the customer explicitly opts into `strict_oos_mode`, which retrains the generator on a held-out slice.

This makes the trust story falsifiable: every customer can verify the memorization risk for their own model, on their own data.

## When you DO want strict-OOS mode

The default works for the typical case: a rule-based or factor strategy with a small parameter count. The strategy didn't itself "see" the training data in any meaningful sense — its parameters were chosen by the researcher.

The exception: **the strategy itself is an ML model trained on the same window.** If both FLOW and a deep-neural-net strategy were trained on 2010-2023, both could have learned the same noise features, and FLOW's synthetic samples would falsely confirm the strategy's robustness.

For these customers we offer an opt-in `strict_oos_mode=True` parameter that trains FLOW on a held-out slice the strategy never touched. The synthetic distribution then provides a clean OOS evaluation. The tradeoff: less statistical power (smaller training window) and the customer has to specify the held-out range.

## Summary

| Concern | Resolution |
|---|---|
| "Isn't training on the same period contamination?" | No — the strategy never sees training data; only new samples from the learned distribution. |
| "What if the generator memorizes?" | Reported as a per-model risk score; SDK refuses to return synthetic if `risk='high'`. |
| "What about ML-trained strategies?" | Opt into `strict_oos_mode=True` for a held-out-trained generator. |
| "Is this standard?" | Yes — López de Prado, Bailey, AWS, TimeGAN, TSGBench all use in-sample training. |
