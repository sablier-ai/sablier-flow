# Changelog

All notable changes to `sablier-flow` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.2] - 2026-06-09 — data-adaptive fit-horizon cap

### Changed
- The `fit` `horizon` cap is now data-adaptive instead of a flat 504 time
  steps. The maximum scales with how many rows you pass: a longer training
  window is allowed only when the history leaves enough non-overlapping
  spans to train it robustly (floor 504 steps, capped at 5000). Data-rich
  callers — including intraday users with many bars — can now train longer
  windows; everyone else is unchanged. `horizon` is a count of time steps
  (bars), independent of cadence. Purely additive: any horizon that passed
  the old flat-504 check still passes. SDK-only; no server change.

## [1.1.1] - 2026-06-08 — input-window gap warning

### Added
- `fit` / `generate` / `validate` now emit a warning when a supplied DataFrame
  (`real_data` / `like` / `anchor_data` / `holdout_data`) has interior NaN gaps
  below the reject threshold. Those bars are forward-filled before the model
  sees them, so the warning flags likely calendar misalignment between feature
  sources (e.g. an FX/macro column on a different trading calendar than your
  equity columns). Non-breaking; no API change.

## [1.1.0] - 2026-06-01 — data_types vocabulary collapse + intraday gate lifted

The customer-facing surface now speaks a clean three-string `data_types`
vocabulary and accepts any uniform-cadence DatetimeIndex. The frequency
kwarg is gone. No legacy aliases are kept — there were no existing customers
and a deprecation period would have just shipped two confusing surfaces.

### Breaking changes (no migration path; clean cut)

- **`ALLOWED_DATA_TYPES` collapsed from 5 strings to 3:**
  - `'price'`  — log-return + z-score; for compounding multiplicative series
    (asset prices, FX, ratios). Requires strictly positive values.
  - `'level'`  — difference + z-score; for additive series that can cross
    zero (interest rates, volatility indices, spreads, dollar index,
    factor levels). Replaces the pre-1.1 `'rate'` / `'volatility'` / `'index'`
    distinction, which were aliases for the same DIFFERENCE transform at the
    SDK's row cadence.
  - `'return'` — identity + z-score; for already-stationary series (factor
    returns, pre-differenced data).
  - Passing any of the retired strings (`'rate'`, `'volatility'`, `'index'`)
    raises `ValueError` with an inline migration hint pointing to the new
    name (`'rate'`/`'volatility'` → `'level'`, `'index'` → `'price'`).
- **`frequency=` kwarg removed from `Client.fit / generate / validate`** and
  every async sibling + module-level shortcut. Row cadence is auto-detected
  from `real_data.index` via the median Δt and surfaced in the pre-flight
  info line so callers can confirm before the GPU round-trip.
- **`ALLOWED_FREQUENCIES` removed from the public surface.**
- **Intraday gate lifted.** Any uniform-cadence DatetimeIndex is accepted —
  daily, intraday (5-min, 1-min, hourly), weekly, monthly, quarterly. The
  cyclical embedding the model conditions on is yearly seasonality
  (day-of-year sin/cos) only — intraday-specific patterns (minute-of-day,
  day-of-week effects) are not modeled in 1.1.0 and are on the roadmap for
  a future minor.
- **`DEMO_DATA_TYPES` registry updated** to use the new vocabulary
  (`VIX` / `TNX` → `'level'`, `DXY` → `'price'`; `SPY` / `QQQ` / `IWM` /
  `TLT` unchanged at `'price'`). The actual transforms applied to demo
  data are byte-identical to 1.0.x — only the labels changed.

### Architecture (the part you don't see)

- **Wire-mapping shim** in the SDK: customer-`'level'` translates to
  wire-`'rate'` (same DIFFERENCE transform server-side); intraday
  cadences send `'daily'` on the wire so the legacy frequency-driven
  overrides never accidentally fire. Containing the back-compat
  translation to a 3-entry dict in the SDK means it collapses to
  identity once the server accepts the new vocabulary natively, with
  no customer-facing change.

### What's NOT in 1.1.0 (and where they're going)

| Feature | Status |
|---|---|
| Stair-step / forward-filled lower-cadence features (CPI, GDP, Fed rate) | Not supported — aggregate to row cadence first. Registry shape supports adding `'price_yoy'` / `'level_std'` etc. when a real customer needs them. |
| Intraday-specific cyclical embeddings (minute-of-day, day-of-week) | Not modeled — yearly day-of-year is the only embedding today. Multi-cycle support is a model-quality improvement for a future minor. |
| Auto-detection of data types | Not in the default API. Customer declares explicitly. Helper (`sf.suggest_data_types(df)`) could ship later as a convenience. |
| Negative-value-tolerant `'price'` | Use `'level'` instead — the right escape hatch for series that could theoretically cross zero. |

### Tests added

- `tests/unit/test_data_types_vocabulary.py` — 32 tests covering the 3-string
  contract, legacy alias rejection with migration hint, the wire-mapping
  shim, and the demo registry's vocabulary.
- `tests/unit/test_intraday_cadence.py` — 16 tests covering cadence
  detection across uniform-cadence indices (5-min / 1-min / 15-min / hourly /
  daily / weekly / monthly / quarterly), the wire-frequency back-compat
  collapse, and the lifted intraday gate.

## [1.0.22] - 2026-06-01 — demo dataset rename + docs sweep

### Changed (breaking — dataset key)
- **Demo dataset keys renamed for honesty:**
  - `us_equities_macro_2010_2024` → `us_equities_macro_2010_2023`
  - `us_equities_2010_2024` → `us_equities_2010_2023`

  The data actually ends 2023-12-28; the `_2024` suffix in the key was
  misleading. The default `sf.demo_data()` (with no arguments) is unaffected.
  Customers using the explicit key need to update to the new spelling. The
  bundled parquet filenames in the wheel are renamed to match.

### Fixed (docs side — full sweep from a 92-finding Playwright audit)
- Homepage hero code block now runnable end-to-end (was missing
  `data_types=` and used undefined `real_data` / `my_backtest`).
- Quickstart `5-line workflow` and `more explicit` blocks both runnable
  (same root cause); added a symmetric-window warning admonition.
- Recipes restructured into a shared `# The shared workflow` section
  with the correct fit + generate + robustness pattern (`data_types=` per
  column semantics, `like=backtest_window` for symmetric windows); each
  recipe now shows only the loader. Five of seven recipes were previously
  non-runnable.
- kdb+/PyKX recipes in both `recipes.md` and `concepts/engine-integration.md`
  rewritten as actual runnable PyKX (was dead f-string + wrong call
  signature + mis-shaped payload).
- `concepts/data-sourcing.md` canonical example no longer KeyErrors on
  `df.attrs['data_types']` (now builds the dict at the loader).
- `concepts/in-sample-is-correct.md` NN-distance bands aligned with
  SDK.md (0.85-1.15 healthy band); literature citations
  (López de Prado, Bailey, AWS HPC, TimeGAN, TSGBench, Carlini et al.,
  Salvi et al.) hyperlinked.
- `concepts/index.md` expanded from a 668-character stub into a proper
  hub answering the four objections every senior quant raises in the
  first ten minutes.
- `SDK.md` Versioning section gets current version + CHANGELOG link +
  recent-changes digest (was generic semver boilerplate);
  `JobHandle.kind` enum corrected (`'train'` → `'fit'`).
- `examples/00_getting_started.ipynb` Step 9 strategy family widened
  from 6 SMA variants to 24 (the prior small family produced
  `verdict: 'uncalibrated'` on the demo's own data, killing credibility
  for the headline calibration tool).
- `examples/01_backtest_robustness.ipynb` adds explicit prose
  reconciliation for the FamilyReport aggregate-verdict label inversion
  on small honest families (the per-strategy `overfit_score` carries the
  demo's actual claim).
- `examples/02_tstr_predictive_rank.ipynb` `horizon=63` →
  `horizon=126` (now actually matches the 126-bar OOS window);
  corrected the comment that wrongly claimed the OOS window "excludes COVID".
- `examples/03_memorization_audit.ipynb` replay-floor prose reconciled
  with the actual computed value (`R_replay ≈ 0.02`, not `≈ 0.15` /
  `≈ 0.20`).
- All four notebook install pins bumped to `>=1.0.22`; broken
  in-notebook `../docs/SDK.md` 404 link replaced with
  `https://docs.sablier.ai/SDK/`; broken in-page heading anchors fixed.

## [1.0.21] - 2026-06-01 — audit-driven hardening

### Hardened
- API-key handling on login / repr / credentials file. `LoginResult` and
  `JobHandle` reprs redact secrets; credentials tempfile opens with
  `O_CREAT | O_EXCL | 0o600`; `endpoint=` kwarg and `SABLIER_FLOW_ENDPOINT`
  env var both go through the same allowlist as the stored credentials.
- `sablier-flow generate` CLI now wires the required `--data-types` flag
  (accepts JSON or comma-pair form, with `df.attrs['data_types']` as a
  fallback).

### Fixed — methodology
- `predictive_rank_score` raises on mixed real/synth input forms instead
  of silently producing a `'well_calibrated'` verdict on shape-mismatched
  inputs.
- Bailey-LdP analytical DSR emits a UserWarning when the observed Sharpe
  looks annualized while `strategy_returns` are per-period (the common
  units-mismatch signature).
- DSR rejects NaN `observed_sr` instead of silently returning `0.0`;
  reports the count of dropped non-finite synth paths in `notes`; spells
  out the implicit `t_obs=252` fallback when called without
  `strategy_returns`.
- `FamilyReport.summary()` leads with `'overfit_selection'` when
  `pbo ≥ 0.6` (matches what `report.verdict` says).
- `probability_of_backtest_overfitting` warns on `chunk_size < 30` and
  surfaces partition-skip rate.
- `evaluate_family` now dual-routes shared kwargs (`data_types=`,
  `api_key=`, `endpoint=`, etc.) to both the fit and generate calls.

### Fixed — ergonomics + docs
- `from sablier_flow.adapters import write_lean_csv_universe` now works
  (was `ImportError`).
- `Client.resume()` clears the pending-job file on terminal failure as
  well as success.
- `consistency_check(...)` accepts `FamilyReport` as a baseline.
- `SDK.md` signatures for `Client.fit / generate / validate` show the
  full kwarg surface; credits / usage return types corrected; broken
  TOC anchors fixed; canonical `GenerationResult.as_dataframes` recipe
  now passes the same window to real + synth backtests (the asymmetric
  pre-1.0.21 example mechanically produced `'highly_overfit'`).
- Removed stale references to deleted APIs (`@sablier_flow.augment`,
  `alternative_versions`, `FamilyReport.to_html`) from module docstrings.

## [1.0.20] - 2026-06-01 — four ergonomics + safety fixes

### Hardened
- `sf.login()` hard-truncates the printed `key_prefix` to 12 chars
  client-side regardless of server behaviour, so the API key can't
  reach stdout / scrollback / CI logs via the prefix field.

### Fixed
- `sf.estimate_cost(...)` no longer returns the misleading
  `estimated_duration_s` field (the credit estimate is deterministic
  and reliable; the duration heuristic was running ~4-5× high).
- `f"{deflated_sharpe_report:.4f}"` no longer raises `TypeError` —
  `__format__` routes numeric format specs to the headline `realistic`
  DSR.
- `evaluate_family` warns when real-data window length and synthetic
  horizon differ, since the asymmetric comparison produced a biased
  DSR null. Points callers at `like=real_data.iloc[-gen.horizon:]`.

## [1.0.19] - 2026-06-01 — forward-forecast anchoring + docs sweep

### Fixed
- **`Client.generate` / `Client.generate_async`: forward-forecast paths
  now anchor at "today".** Previously, when `anchor_data=` was passed
  (and `like=` wasn't), `anchor_prices` was never sent to the server
  and the server fell back to the checkpoint's stored `last_prices`
  (anchored at training end). The result: forward synthetic paths
  started at the price level from the *first* bar of `anchor_data`
  rather than the last bar, so a fan chart drawn against
  `real.iloc[-90:]` showed an obvious ~20% gap between where the
  realized line ends and where the synthetic continuation begins. Fix:
  when `anchor_data` is supplied and `like` isn't, derive
  `anchor_prices = anchor_data.iloc[-1].to_dict()` — same mechanism as
  the `like=` branch already used, just for the forward-generation use
  case. Applies to both the sync and async generate paths.

### Changed (docs)
- `docs/concepts/data-sourcing.md` and
  `docs/concepts/engine-integration.md` snippets now use the real
  `sf.fit(...)` → `sf.generate(model_id, ...)` surface (was
  `client.alternative_versions(...)` — a method that does not exist;
  customers copying the old snippet would have hit `AttributeError`).
- `docs/concepts/in-sample-is-correct.md` aligns with the actual SDK:
  `ValidationReport.memorization_risk` (was `MemorizationReport.risk`),
  and the recommended pattern for ML-trained-strategy customers is
  manually slicing the DataFrame before `sf.fit` (was a nonexistent
  `strict_oos_mode=True` parameter).
- `docs/SDK.md` `JobHandle.kind` enum now lists `'fit'` instead of
  `'train'` (matches the actual kind string the SDK emits).
- `docs/concepts/data-sourcing.md` now correctly says intraday
  classification is deferred to 1.1.0 (the 5-min demo dataset is a
  preview-only sample, not a fit target — matches what the SDK
  actually enforces in `_require_frequency`).
- `docs/recipes.md` Frequency row no longer references the nonexistent
  `quick_validate` / `periods_per_year` — replaced with the real
  `frequency=` kwarg on `sf.fit`.
- `examples/00_getting_started.ipynb`: the estimate_cost code cell
  dropped the stale `~{estimated_duration_s/60} min` print line and
  switched from `kind='train'` (which raised in 1.0.18+) to
  `kind='fit'`. The 2010-2024 prose reference corrected to 2010-2023.
- `examples/02_tstr_predictive_rank.ipynb` and
  `examples/03_memorization_audit.ipynb`: stripped residual
  wall-clock claims ("fit takes ~5 min", "≲ 15 minutes") so the
  no-time-prediction stance is consistent across every example.
- `examples/03_memorization_audit.ipynb`: simplified the defensive
  `getattr(_c, 'available', None) if not isinstance(_c, dict) else
  _c.get('available')` dance to plain attribute access — Pydantic
  `CreditsBalance` is now stable, the dual-path was leftover from the
  pre-1.0.13 migration.

## [1.0.18] - 2026-06-01 — agent-introspection + honest cost surface

### Added
- `__dir__()` on `sablier_flow` so `dir(sablier_flow)` returns the full
  public surface (was 2 names before — PEP 562 lazy `__getattr__` made
  tab completion + agent introspection effectively empty, leading agents
  to claim helpers like `list_jobs` / `fetch_result` didn't exist).
- Module docstring expanded: dedicated "Async / job control" and
  "Account / billing" sections so `help(sablier_flow)` is a complete
  reference, not just the canonical happy path.
- `fit_async` docstring spells out the full job-control surface
  (`list_jobs` / `fetch_result` / `cancel_job` / `resume_job`) with the
  shape of the `progress` dict (`step`, `phase`, `message`, `metrics`,
  `total_steps`) and `last_progress_at` heartbeat. `generate_async` and
  `validate_async` cross-reference it.
- `docs/quickstart.md` Section 7 now demos `sf.list_jobs()` for live
  progress monitoring.

### Changed
- **`estimate_cost` is now credits-only.** Documented return is
  `{estimated_credits, low, high, notes}` — `estimated_duration_s` is no
  longer surfaced. Credit estimates are deterministic (formula based on
  dataset shape × horizon × n_paths); wall-clock depends on queue depth
  and GPU availability and is intentionally not predicted. Use
  `sf.list_jobs()` for the live signal once a job is running. (The
  underlying wire field is preserved on the response dataclass for
  back-compat, but downstream callers should ignore it.)
- `evaluate_family` runtime warning drops "estimated wall-clock ~X
  seconds" — the per-strategy cost depends on what's inside the
  customer's backtest function (microseconds to seconds), so a printed
  prediction was misleading in both directions. The partition count and
  strategy-evaluation count are still reported so the caller can form
  their own expectation.

### Fixed
- Stripped stale "~15 min" / "~15-20 min" / "~10-15 min" wall-clock
  claims from every customer-facing surface: module docstrings
  (`analytics/family.py`), `docs/quickstart.md`, `docs/SDK.md`,
  `src/sablier_flow/_resources/SDK.md`, in-wheel
  `_resources/getting_started.ipynb` (markdown across cells 1, 10, 14,
  16, 18, 25, 35, 40 + code cell 41), and the published mirrors at
  `examples/00_getting_started.ipynb`,
  `examples/03_memorization_audit.ipynb`, and their `docs/examples/`
  copies. Existing executed stderr lines from prior runs are kept as
  historical artifacts; future re-executions will use the cleaned-up
  warning.
- In-wheel `getting_started.ipynb` code cells now use attribute access
  on `CreditsBalance` / `UsageSummary` (`balance.available`,
  `summary.total_credits`) instead of dict subscripts that would
  `AttributeError` on the Pydantic objects, and `estimate_cost('fit',
  ...)` instead of the unsupported `kind='train'`.

## [1.0.17] - 2026-06-01 — numbers consistency patch

### Fixed
- README Examples table headline numbers now lock to the actual shipped
  notebook outputs (not transcribed values from a prior run):
  - N1: 29 of 30 lucky vs 0 of 12 honest at threshold 0.7
        (honest max = 0.690, lucky min = 0.670, lucky max = 0.875)
  - N2: Spearman ρ = +0.7687, 95% CI [+0.47, +0.95], p = 1.14e-05
- examples/02_tstr_predictive_rank.ipynb Section 8 verdict markdown
  realigned with the executed-output cell (same numbers as N2 above).
- src/sablier_flow/__init__.py docstring ticker list now correctly
  reads 'SPY/QQQ/IWM/TLT plus 3 macro features (VIX, TNX, DXY)'.
- docs/quickstart.md: dropped data_types={c:'price' for c in df.columns}
  antipattern (matches README Quickstart's df.attrs['data_types']).
- v1.0.13 GitHub release body updated to match current numbers.

## [1.0.16] - 2026-06-01 — docs/notebook polish

### Fixed
- examples/02_tstr_predictive_rank.ipynb Section 8 markdown numbers
  realigned with the executed cell output (see 1.0.17 for the
  current canonical values).
- src/sablier_flow/__init__.py demo_data date range corrected to 2010-2023.
- examples/03_memorization_audit.ipynb re-executed against a clean 1.0.16
  install (1.0.15 sdist was published from a build that ran before the
  notebook re-execute completed).
- CHANGELOG 1.0.15 entry merged dual ### Changed sections.

## [1.0.15] - 2026-06-01 — docstring + notebook pin cleanup

### Changed
- Live notebook numbers refreshed against 1.0.15 (executed cleanly end-to-end).
  Note: the transcribed values shipped in this entry were later found to
  disagree with the actual executed-cell outputs; see 1.0.17 for the
  canonical numbers (N1: 29/30 lucky vs 0/12 honest at threshold 0.7;
  N2: Spearman ρ = +0.7687, 95% CI [+0.47, +0.95]; N3: R unchanged).
- README cosmetic refinements: demo dataset date range (2010-2023, actual),
  TSTR predictive-rank reporting now leads with the bootstrap CI from the
  notebook, LEAN adapter wording ("CSV export adapter" instead of
  "QuantConnect adapter").
- GitHub release v1.0.13 title aligned to CHANGELOG ('first public PyPI
  release', was 'initial public release').

### Fixed
- `src/sablier_flow/__init__.py` module docstring no longer ships the
  `data_types={c:'price' for c in df.columns}` pattern that violates the
  1.0.9 five-type contract — `help(sablier_flow)` and any IDE that surfaces
  the module docstring now show the same canonical `df.attrs['data_types']`
  + `like=backtest_window` pattern as the README Quickstart.
- All four example notebook install pins bumped to >=1.0.15. Notebooks
  re-executed against a clean 1.0.15 install so docs.sablier.ai/examples/*
  no longer shows the brief pip-resolve-window error visible on 1.0.14's
  00_getting_started page.

## [1.0.14] - 2026-06-01 — README fix-pass + PyPI metadata correction

### Fixed
- README 'Five-line demo' is now copy-paste runnable: `my_backtest` is defined inline,
  `df.attrs['data_types']` carries the canonical 5-type contract (1.0.9+) instead
  of the wrong {c:'price' for c in df.columns}, and `sf.generate(like=...)` ensures
  real and synthetic shapes match in `sf.robustness`. Renamed to 'Quickstart' since
  it is no longer literally 5 lines.
- pyproject.toml: ship Homepage `https://docs.sablier.ai` (1.0.13's wheel pinned to
  a non-functional URL; PyPI metadata is per-wheel-immutable, so a new release was
  required).
- examples/00_getting_started.ipynb: bump install pin >=1.0.6 → >=1.0.14.

## [1.0.13] - 2026-05-31 — first public PyPI release

### Added
- First public release on PyPI: `pip install sablier-flow`.
- Four executed notebooks under `examples/` (open them on GitHub to see live numbers):
  - `00_getting_started.ipynb` — end-to-end SDK tour (login, fit, validate, generate, robustness, async, management).
  - `01_backtest_robustness.ipynb` — selection-bias catch via per-strategy `overfit_score`. Flags 29/30 lucky strategies at threshold 0.7 vs 1/12 honest false positives on a 500-strategy pure-noise pool.
  - `02_tstr_predictive_rank.ipynb` — Train-on-Synthetic Test-on-Real Spearman ρ = +0.74, 95% CI [+0.55, +0.83] on a 24-variant family.
  - `03_memorization_audit.ipynb` — NN-distance ratio R = 0.9309 vs replay-floor R = 0.0161 (57.8× separation). `memorization_risk = 'low'`.
- mkdocs-material docs site at https://docs.sablier.ai with all four notebooks rendered inline.
- Apache 2.0 license for the SDK; CC BY 4.0 for documentation.

### Changed
- Repository visibility: private → public. Canonical URL https://github.com/sablier-ai/sablier-flow.
- Memorization audit threshold bands reverted to the empirically-calibrated 0.80 (low) / 0.50 (medium) values appropriate for financial returns.

### Removed
- All paper references from SDK docstrings, docs, and notebooks. Every numerical claim
  is now computed live in the notebooks themselves.

## [1.0.12] - 2026-05-30 — internal alpha; not publicly announced.

## [1.0.11] - 2026-05-29 — internal alpha; not publicly announced.

## [1.0.10] - 2026-05-28 — internal alpha; not publicly announced.

## 1.0.9 — Removed (breaking)

This release rips out every back-compat shim, legacy-format handler, and
"we used to support X" code path. There were no production customers on
the 1.0.5 / 1.0.6 / 1.0.7 wheels; the SDK is shipping clean from this
release forward.

### Removed

- **Dict-style access on `CreditsBalance` / `UsageSummary` / `UsageEvent`.**
  The `_DictCompatMixin.__getitem__` shim that emitted a
  `DeprecationWarning` for `credits['monthly_used']`-style access is gone.
  Use attribute access only: `credits.monthly_used`. Dict-style raises
  `TypeError` now.
- **`JobHandle` wire vocabulary `'train'`.** Pre-1.0.7 handles persisted
  with `kind='train'` were silently normalized to `'fit'`. The
  normalization layer is gone — handles must round-trip with the
  canonical `{'fit', 'generate', 'validate'}` vocabulary. Re-issue any
  long-lived handles via `sf.fit_async`.
- **Wire kind `'train'`.** `JobKind` is now `{'fit', 'generate',
  'validate'}` end-to-end. The SDK no longer maps `'fit'` → `'train'`
  on the wire (`Client.fit` / `Client.fit_async` /
  `Client._async_dispatch`); the value the customer sees is the value
  the server receives. `Client.fetch_result` and `Client.resume` no
  longer accept the legacy `'train'` kind on persisted handles /
  pending-job records.
- **`Client.estimate_cost('train')` alias.** `estimate_cost` only
  accepts `{'fit', 'generate', 'validate'}` now; passing `'train'`
  raises `ValueError`. The friendly normalization that mapped
  `'fit'` → `'train'` on the wire is gone.
- **Backend 1.0.5 nested-`params` wire fallback.** The transport layer
  always ships `feature_data_types` and `frequency` at the top level of
  the job-create envelope. The dual-write that nested the same fields
  inside `params` for old backends is removed. Backends older than
  1.0.6 will reject these jobs.
- **Pre-1.0.5 checkpoint support.** Checkpoints fitted with allowed-set
  values `'spread'` / `'other'` / `'unknown'` / `'bounded'` / `'pct'`, or
  with `'ratio'` / `'level'`, can no longer be loaded. The legacy
  `'ratio' → 'price'` and `'level' → 'index'` collapse paths are gone.
  Refit any such model via `sf.fit` against the canonical 5-type contract.
- **`'unknown'` data_type silent fallback in `generation_service`.** The
  server-side fallback that mapped an `'unknown'` annotation to log-return
  inverse-transform is removed. The five-type contract (`'price'`,
  `'return'`, `'rate'`, `'index'`, `'volatility'`) is now enforced
  end-to-end with no implicit branch.
- **`alternative_versions_kwargs=` dict on `evaluate_family`.** The 1.0.6
  back-compat shim that merged a deprecated kwargs dict (with a runtime
  `DeprecationWarning`) is gone. Pass kwargs explicitly.
- **Module-level `sablier_flow.alternative_versions` deprecation shim.**
  The 0.0.2a0 one-shot entry point that emitted `DeprecationWarning` and
  internally did `fit + generate` is removed. Use the explicit
  `sf.fit(...) → sf.generate(...)` pair.
- **Legacy 0.5.x kwarg-name rejection helper.** The `_reject_removed_kwargs`
  module-level helper that raised `TypeError` on `target_features=` /
  `conditioning_features=` is gone, along with its call sites in the
  `sf.fit` / `sf.generate` / `sf.validate` module shortcuts. Unknown
  kwargs still surface via the `**extra` -> `TypeError` path; the only
  thing removed is the legacy-name-specific friendly message.
- **0.4.x `ModelInfoResponse` `target_features` + `conditioning_features`
  fallback.** The SDK transport's `ModelInfoResponse` no longer carries
  the optional pre-`features` fields, and `_model_info_to_dataclass` no
  longer merges them when `features` is missing. The wire ships
  `features: list[str]` (required) and the backend always populates it.
- **0.4.x `JobResultPayload.from_fit_result` dual-write.** The fit-result
  metadata blob no longer carries `target_features` + empty
  `conditioning_features` alongside `features`. `to_fit_result` reads
  `features` only — no merge fallback.
- **Backend `ModelInfo` legacy `target_features` + `conditioning_features`
  fields.** The pydantic response model and `_row_to_model_info` mapper
  drop the two optional list fields that were kept on the wire for 0.4.x
  SDK readers. Wire shape is now `features: list[str]` only.
- **Server-side TEE worker wire kind `'train'`.** `JOB_TYPES`,
  server-side job dispatch, and the dev-mode mock all key
  off `'fit'` now. Matches the SDK's canonical vocabulary; the legacy
  `'train'` literal is gone end-to-end.
- **Pre-1.0.7 `validate` "sanity check against training data" docstring
  branch.** The phrasing is gone from `Client.validate`; auto-OOS is the
  only mode.
- **Migration paragraphs in `README.md` / `docs/SDK.md` / `docs/quickstart.md`.**
  Any "data_types= is required from 1.0.5", "1.0.7+", or "pre-1.0.5 X
  collapses into Y" paragraph is rewritten to describe the current
  contract without referring to past versions.

## 1.0.7

### Fixed

- **`Client.validate` is OOS-by-default end-to-end.** The docstring now
  documents what already happens on the wire: with no `holdout_data=`
  the report is run against the OOS slice persisted at fit time; with
  `holdout_data=` it runs against the caller's window. The pre-1.0.7
  "sanity check against the training data" phrasing is gone — it was
  wrong in the auto-OOS path (the server has never reused training
  bars for validation, and saying so misled customers into passing
  redundant `holdout_data=` on every call).
- **PBO floor raised from 8 to 16 splits.** The SDK examples used to
  show `pbo_cscv_splits=8` "for runtime"; in practice that under-detects
  overfit on the small-strategy-family case the docs walked through.
  `evaluate_family` and `probability_of_backtest_overfitting` keep
  `pbo_cscv_splits: int = 16` as the default; the SDK.md examples now
  match. Lower values still work but emit a soft-warning note in
  `FamilyReport.notes`.
- **Getting-started notebook Step 9 no longer overlaps the held-out
  OOS slice.** The cell used `real_oos = real.iloc[-252:]` as a
  "realistic OOS reference", but on the bundled demo the last 252 rows
  fall inside the slice `sf.fit(train_split=0.8, embargo_days=21)`
  already held out — so the calibration was reading the model's own
  validation slice as if it were unseen. Now derives `real_oos` from
  `fit.holdout_start_date` / `fit.holdout_end_date` so the rank-correlation
  read is on truly out-of-sample data. Fixed in both
  `examples/00_getting_started.ipynb` and the wheel-bundled
  `src/sablier_flow/_resources/getting_started.ipynb`.

### Added

- **`RobustnessReport.verdict` gains `'degenerate_synth'` and
  `'insufficient_data'`** for the edge cases where the synthetic
  distribution has near-zero spread or the caller passed too few
  synth results to bucket. `.acceptable` returns `None` (not bool)
  in the `'insufficient_data'` branch so UIs can render a "needs
  more paths" tile instead of a misleading green/red.

## 1.0.5

### Breaking

- **`data_types=` is now a required kwarg** on `sf.fit`, `sf.fit_async`,
  `sf.generate`, `sf.generate_async`, `sf.validate`, and
  `sf.validate_async`. The SDK can no longer guess the right transform
  per column — pass an explicit `dict[str, str]` mapping each column in
  `features=` to one of `{'price', 'return', 'rate', 'index', 'volatility'}`.
  Missing the kwarg raises `TypeError` with the allowed-set message;
  passing an unknown value raises `ValueError`.

  **Allowed-set change.** Five pre-1.0.5 values are removed end-to-end
  (`'spread'`, `'other'`, `'unknown'`, `'bounded'`, `'pct'`). Legacy
  `'ratio'` collapses into `'price'`; legacy `'level'` collapses into
  `'index'`. The five removed values were never wired to a real
  inverse-transform branch on the server and produced silently-wrong
  synthetic output — explicit rejection is the safer default.

  **Migration.** Add `data_types={col: 'price'}` (or whichever type
  applies per column) to every call site. Bundled demos attach the
  canonical map on `df.attrs['data_types']` so the simplest call is::

      real = sf.demo_data()       # attrs['data_types'] is set for you
      fit  = sf.fit(
          real,
          features=real.columns.tolist(),
          data_types=real.attrs['data_types'],   # <-- new in 1.0.5
          horizon=63,
      )

  For your own data::

      data_types = {
          'AAPL':       'price',
          'SPY':        'price',
          'VIX':        'volatility',
          '10Y_yield':  'rate',
          'DXY':        'index',
          'mom_signal': 'return',
      }
      fit = sf.fit(real, features=list(data_types), data_types=data_types, horizon=63)

- **`frequency=` is now strictly `'daily'` / `'weekly'` / `'monthly'` /
  `'quarterly'`** (auto-detected from the median Δt of `df.index`).
  Irregular indices raise rather than silently round-off. Intraday
  classification is deferred to 1.1.0 (the 5-min demo dataset is shipped
  as a preview only; fitting on it raises during schema validation).

### Fixed

- **NaN policy: pass-through to the model**. NaNs are no longer silently
  filled with zero by the backend pipeline before the model sees them
  (the model masks internally and that signal was being destroyed). The
  SDK now rejects a column whose post-ffill NaN fraction exceeds 0.7
  with a clear error that names the offending column(s).
- **`'return'` inverse-transform.** Synthetic output for columns
  annotated as `'return'` previously fell through to the log-return
  branch on inverse-transform, producing silently-wrong values. The
  generation service now has an explicit `'return'` branch that applies
  inverse z-score only (no exp / cumsum / last_price).

### Added

- **`df.attrs['data_types']` on every bundled demo.** `sf.demo_data()`
  (and every named variant) now attaches the canonical per-column
  `data_type` map to the returned DataFrame so you can pass it straight
  through: `sf.fit(df, ..., data_types=df.attrs['data_types'])`.

## 1.0.4

### Fixed

- **`login()` ordering — anti-phishing.** The "Logged in as <email>" line
  now prints (and stdout is flushed) BEFORE `save_credentials()` writes
  the api_key to disk. Previously a Ctrl-C, disk-full, or permission
  error between save and print could leave a stored key on disk with no
  terminal echo of the approver, defeating the whole point of surfacing
  who approved.
- **`validate_stored_endpoint` allowlist — short-form prod URL.** The
  canonical prod URL short-form is now accepted in addition to the
  long-form variant. Previously a credentials file pointing at the
  prod short-form URL would silently
  drop its endpoint on next load and fall back to `DEFAULT_ENDPOINT` —
  no security harm, just a UX wart. `https://` is still required.

### Notes

- **DiskCache: pre-1.0.3 cache entries are silently ignored under the
  new endpoint+api_key-aware key. If you want to reclaim disk:
  `rm -rf ~/.cache/sablier_flow`. Functional impact: zero (the cache
  just re-fetches), only disk usage.**

## 1.0.3

### Fixed

- High-sev: cache key now incorporates both endpoint and api_key, so a
  user switching between profiles (e.g. staging ↔ prod) or rotating an
  api_key no longer reads a stale entry minted for a different identity
  / environment. Old (v0) entries written by 1.0.2 and earlier are
  silently ignored under the new key — see the 1.0.4 note above for the
  disk-reclaim one-liner.
- High-sev: `login()` `save_credentials(... endpoint=endpoint, ...)`
  partial-write hardening — atomic temp+rename plus 0o600 chmod on the
  temp file so a crashed write can never leave a world-readable
  half-file behind.

## [0.5.7] — 2026-05-28

### Changed

- **Documentation cleanup.** Tightened the docstrings on `Client.fit`,
  `Client.generate`, and `FitResult` to describe externally-observable
  behaviour only (horizon-agnostic generation, anchor semantics)
  without referring to internal model architecture.
- **Removed `[core]` extra.** The model runs server-side in a
  hardware-attested confidential GPU; there is no public local-run
  pipeline, so the optional GPU dependency group was misleading.
  Customers installing from PyPI now get the thin remote client only.

## [0.0.2a0] — 2026-05-26

### Changed (breaking)

- **`Client.alternative_versions` split into `fit + generate + validate`.** The one-shot call retrained on every invocation; the new API trains once and reuses the fitted model for as many generations as the customer needs.
    - `Client.fit(real_data, target_features=..., horizon=...) → FitResult` — returns `model_id`; TTL 30 days, extends on every successful use.
    - `Client.generate(model_id, n_paths=..., horizon=..., anchor_data=None) → GenerationResult` — no horizon cap (the generator is horizon-agnostic; quality best near trained horizon). `anchor_data=None` uses the obs window persisted at fit time; pass an explicit DataFrame to anchor synthetic continuation on a specific window.
    - `Client.validate(model_id, holdout_data=None) → ValidationReport` — full structural-validation + memorization suite. Without holdout: training-tail sanity. With holdout: true OOS.
    - `Client.alternative_versions` is kept as a thin shim that emits `DeprecationWarning` and internally does `fit + generate`. Existing customer code keeps working; please migrate before 0.1.0.
- **Module-level shortcuts** `sablier_flow.fit / .generate / .validate` mirror the new methods. `sablier_flow.alternative_versions` continues to work (deprecated).
- **Cost model**: customers running N strategies on one universe now pay **24 + N×1 credits** instead of **N×25 credits**. Same total for N=1; massively cheaper for sweeps.
- `@sablier_flow.augment` decorator internally switched to `fit + generate` so each decorated function trains once per call and reuses the model across all N synthetic backtests.

### Added

- `sablier_flow.FitResult` and `sablier_flow.ValidationReport` dataclasses.
- `JobResultPayload` is polymorphic across `generate / fit / validate` — same on-the-wire framing, distinguished by a `kind` field in metadata.
- Persistent encrypted model storage on the server side. Plaintext checkpoints are never at rest in the control plane; the unwrap key is held only by the API + worker service accounts. (True TEE-bound storage where only an attested enclave can unwrap is on the roadmap — the wire format already carries an `encryption_key_id` discriminator so the migration is in-place.)

## [Unreleased before 0.0.2a0]

### Added
- Initial public release of the SDK under Apache 2.0.
- Pure pipeline: `sablier_flow.pipeline.train_model`, `generate_paths`, `validate_model`, `assess_memorization` — DataFrame in / `GenerationResult` out.
- Robustness scoring: `sablier_flow.robustness(real_result, synthetic_results)` returns a typed `RobustnessReport` with `overfit_score`, synthetic Sharpe CI, and recommendation bands.
- Engine adapters: `as_dataframes`, `as_array`, `as_backtrader_feeds`, `as_vectorbt_panel`.
- Client-side crypto + attestation + transport for the hosted compute service:
    - `sablier_flow.client.crypto` — X25519 + HKDF-SHA256 + AES-256-GCM envelope encryption.
    - `sablier_flow.client.attestation` — `AttestationVerifier` enforcing pinned image digest, hardware type, measurements, freshness, and signature presence.
    - `sablier_flow.client.transport` — Pydantic wire models, `Transport` Protocol, real HTTPS and in-process test transports.
    - `sablier_flow.Client.alternative_versions` — full lifecycle: submit job → verify attestation → envelope-encrypt → upload → poll → decrypt result.
- Demo notebook `examples/01_alternative_versions.ipynb`.

[Unreleased]: https://github.com/sablier-ai/sablier-flow/compare/v1.0.17...HEAD
[1.0.17]: https://github.com/sablier-ai/sablier-flow/releases/tag/v1.0.17
[1.0.16]: https://github.com/sablier-ai/sablier-flow/releases/tag/v1.0.16
[1.0.15]: https://github.com/sablier-ai/sablier-flow/releases/tag/v1.0.15
[1.0.14]: https://github.com/sablier-ai/sablier-flow/releases/tag/v1.0.14
[1.0.13]: https://github.com/sablier-ai/sablier-flow/releases/tag/v1.0.13
