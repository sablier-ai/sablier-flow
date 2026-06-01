# Hou-Xue-Zhang 452-anomaly validation study

The **predictive-validity gate** (Gate 2 in the project plan). Determines whether `sablier-flow`'s synthetic alternative-history methodology actually predicts strategy decay better than block bootstrap.

## The argument

For each of the 452 anomalies in the Hou-Xue-Zhang factor zoo:

1. Take the strategy's published in-sample window. Compute `S_IS` (in-sample Sharpe — what the original paper reported).
2. Train `sablier-flow` on the pre-publication data only.
3. Generate 100 synthetic alternative histories of the pre-publication window.
4. Run the same strategy on each synthetic version → `S_OOS_synth` distribution (mean over 100 paths).
5. Compute `S_OOS_real` on the *actual* post-publication window — the held-out ground truth from the HXZ replication report.
6. Record the triple `(S_IS, S_OOS_real, S_OOS_synth)`.

Then across all 452 anomalies, ask:

- **Calibration**: does `S_OOS_synth ≈ S_OOS_real`? (Both should drop from `S_IS` for overfit factors.)
- **Predictive power**: does the synthetic mean correlate with the live out-of-sample reality better than the in-sample backtest does?
- **Overfit detection**: when `S_IS - S_OOS_real > 0.5` (strongly overfit), does our synthetic catch it (`S_IS - S_OOS_synth > 0.3`) in ≥80% of cases?

If yes, the product works. If no, we have a major problem and the pivot needs re-thinking.

## Pass criteria (Gate 2)

- Spearman correlation between `S_OOS_synth` and `S_OOS_real` across all anomalies **≥ 0.75**
- Median absolute Sharpe error **≤ 0.15**
- For strongly-overfit anomalies, overfit-catch rate **≥ 80%**

## Status

**Data not yet acquired.** This directory is the harness skeleton.
The user is responsible for downloading the q-factor data library from <https://global-q.org> after confirming the academic license terms allow our intended use.

Once the data lands:

```bash
cd benchmarks/hxz
python run_study.py --n_anomalies 452 --n_paths 100 --output results.json
python analyze.py results.json   # produces the verdict + the publication plots
```

Estimated compute: ~38 H100-hours total (~$150-400 on confidential A3 spot).
