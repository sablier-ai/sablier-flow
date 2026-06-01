# Concepts

The conceptual underpinnings of `sablier-flow`: what the model does, why the methodology is correct, and how to read the outputs.

- [Why in-sample training is correct](in-sample-is-correct.md) — the central methodological argument. Read this if your first reaction is "isn't training on the test data contamination?"
- [Data sourcing](data-sourcing.md) — bring your own data, what shape it needs to be in.
- [Engine integration](engine-integration.md) — every backtest engine, including in-house C++.

For the operational story (how to interpret robustness reports, predictive-rank scores, deflated Sharpe under two nulls, etc.) see the [SDK reference's "Interpreting the output" section](../SDK.md#interpreting-the-output) — it's the canonical place for verdict semantics.
