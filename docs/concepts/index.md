# Concepts

The conceptual underpinnings of `sablier-flow`: what the SDK does, why the methodology is defensible, and how to read what it returns. This page answers the four objections every senior quant raises in the first ten minutes. The longer companion essays follow.

## The four objections (and the short answers)

### 1. "Isn't training a generator on history and then evaluating on the same history just data leakage?"

No, because the strategy never sees the training data. The flow model learns a *distribution* over alternative-history paths that share your data's joint statistics; the strategy is then evaluated against samples drawn from that distribution. The customer's backtest function operates on synthetic paths it has never seen. What's being tested is whether the strategy's edge survives realization-specific noise — not whether the model can memorize a single realization.

This is the standard methodology in every published synthetic-data benchmark for finance (see the literature pile in [Why in-sample training is correct](in-sample-is-correct.md)). The legitimate concern this objection collides with is **memorization** — that's a separate failure mode the SDK tests for explicitly via `ValidationReport.memorization_risk` and the NN-distance ratio.

### 2. "How do I know the synthetic distribution is actually realistic?"

Two independent axes. **Distributional fidelity** — does the synthetic distribution match the real one on the things finance practitioners care about (fat tails, volatility clustering, leverage effect, cross-asset tail co-movement, drawdown profile)? Twenty numeric scores across five families, reported by `sf.validate(model_id)`; see the [SDK reference's "Interpreting the output" section](../SDK.md#interpreting-the-output) and the [FinBench leaderboard](https://github.com/sablier-ai/finbench) for the full battery.

**Predictive-rank validity** — does the strategy ranking on synthetic forward paths predict the ranking on real OOS data? A generator can match the marginals perfectly and still invert the ranking (a practitioner picking strategies on synth would systematically choose the worst real-market variant). `sf.predictive_rank_score` runs this check directly on your model and your strategy family. The [TSTR predictive rank notebook](../examples/02_tstr_predictive_rank.ipynb) walks the methodology end-to-end; the published `Spearman ρ = +0.7687, 95% CI [+0.47, +0.95]` is the headline.

Either axis can pass while the other fails. A generator is only useful when **both** pass.

### 3. "Why should I trust the model didn't just memorize my training set?"

Because the SDK reports the answer for your specific model on your specific data, not as a claim — as a number you can verify. The NN-distance ratio is the median synthetic-to-training nearest-neighbour distance divided by the median training-to-training nearest-neighbour distance:

- Ratio in `0.85 – 1.15` (`memorization_risk = 'low'`): synth distributed through the training manifold at training-like density. Healthy.
- Ratio outside that band but `≥ ~0.5` (`'medium'`): synth is structurally tighter than training. Cross-check the per-bin `coverage_*` metrics.
- Ratio `< ~0.5` (`'high'`): synth closer to training than training is to itself. Memorization detected. The SDK does not gate on this — you read it and decide whether to retrain on a smaller in-sample slice, partition the universe, or reject the model.

The model architecture also makes memorization structurally difficult — operating in returns space, no date input, the existing validation suite would flag it; see the full argument in [Why in-sample training is correct](in-sample-is-correct.md#the-one-technical-caveat-memorization).

### 4. "Why should I trust the data I send doesn't leak?"

The hosted service is in **alpha** today — envelope-encryption + image-digest pinning are live on the wire protocol, AMD SEV-SNP + NVIDIA H100 confidential-compute substrate is on the roadmap, not yet live. The honest threat model is documented in the [security posture section of the SDK reference](../SDK.md#security-posture-today-alpha). If your security review requires hardware memory encryption today, hold until the SEV-SNP rollout. If TLS + KMS + ephemeral keys + image-digest pinning clears your bar (most quant-tech reviews do), the current release is usable.

The customer-facing wire protocol stays identical when the confidential-compute substrate ships — only the underlying VM changes.

## The companion essays

- **[Why in-sample training is correct](in-sample-is-correct.md)** — the central methodological argument, with the published-literature pile and the memorization-risk falsifier in detail. Read this when a quant counterpart pushes back hard on objection 1.
- **[Data sourcing](data-sourcing.md)** — bring your own data, what shape it needs to be in, the `data_types` annotation contract, why we never integrate with data vendors. Read this when you're wiring up production data.
- **[Engine integration](engine-integration.md)** — every backtest engine, including in-house C++, kdb+, LEAN, backtrader, vectorbt. Read this when you're hooking up to your firm's existing backtester.

## Operational semantics (where to find what)

| You want… | Read… |
|---|---|
| Verdict semantics — what `'highly_overfit'` / `'looks_like_noise'` / `'overfit_selection'` actually mean | [SDK reference — Interpreting the output](../SDK.md#interpreting-the-output) |
| DSR under two nulls (realistic vs Bailey-LdP analytical) | [SDK reference — Deflated Sharpe Ratio](../SDK.md#deflated-sharpe-ratio) |
| PBO via CSCV, when to trust it vs not | [SDK reference — Strategy families and parameter sweeps](../SDK.md#strategy-families-and-parameter-sweeps) |
| End-to-end on the demo data | [Getting started notebook](../examples/00_getting_started.ipynb) |
| Lucky-vs-honest live demonstration | [Backtest robustness notebook](../examples/01_backtest_robustness.ipynb) |
| Predictive-rank methodology + benchmark | [TSTR predictive rank notebook](../examples/02_tstr_predictive_rank.ipynb) |
| Memorization audit live test | [Memorization audit notebook](../examples/03_memorization_audit.ipynb) |
