# Changelog

All notable changes to `sablier-flow` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  `server.tee.runner` dispatch, and `server.api.main` dev mock all key
  off `'fit'` now. Matches the SDK's canonical vocabulary; the legacy
  `'train'` literal is gone end-to-end.
- **Pre-1.0.7 `validate` "sanity check against training data" docstring
  branch.** The phrasing is gone from `Client.validate`; auto-OOS is the
  only mode.
- **Migration paragraphs in `README.md` / `docs/SDK.md` / `docs/quickstart.md`.**
  Any "data_types= is required from 1.0.5", "1.0.7+", or "pre-1.0.5 X
  collapses into Y" paragraph is rewritten to describe the current
  contract without referring to past versions.

## 1.0.7 (unreleased)

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

## 1.0.5 (unreleased)

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

## 1.0.4 (unreleased)

### Fixed

- **`login()` ordering — anti-phishing.** The "Logged in as <email>" line
  now prints (and stdout is flushed) BEFORE `save_credentials()` writes
  the api_key to disk. Previously a Ctrl-C, disk-full, or permission
  error between save and print could leave a stored key on disk with no
  terminal echo of the approver, defeating the whole point of surfacing
  who approved.
- **`validate_stored_endpoint` allowlist — Cloud Run short alias.** The
  canonical prod URL `https://sablier-api-<hash>-uc.a.run.app` (the form
  used by our deploy scripts and smoke tests) is now accepted in
  addition to the long `*.us-central1.run.app` form. Previously a
  credentials file pointing at the prod short-form URL would silently
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
- Persistent encrypted model storage on the server side (`flow_sdk_models` table, AES-256-GCM checkpoint blobs in GCS). Sablier control plane never sees the plaintext checkpoint at rest; the key is held by the API + worker service accounts. (True TEE-bound storage where only the attested enclave can unwrap is v1.x — wire format already carries an `encryption_key_id` discriminator so the migration is in-place.)

## [Unreleased before 0.0.2a0]

### Added
- Initial repo scaffolding (Workstream A): `pyproject.toml`, Apache-2.0 LICENSE, README, CI workflow skeleton, package directory tree.
- Apache-2.0 license boundary established. `server/` (TEE container) excluded from the wheel and remains proprietary.
- **Pure pipeline (Workstream B.2)**: `sablier_flow.pipeline.train_model`, `generate_paths`, `validate_model`, `assess_memorization` — DataFrame in / GenerationResult out, no DB or auth coupling.
- **Robustness scoring (Workstream E)**: `sablier_flow.robustness(real_result, synthetic_results)` returns a typed `RobustnessReport` with overfit_score, synthetic Sharpe CI, and recommendation bands.
- **Engine adapters (Workstream E)**: `as_dataframes`, `as_array`, `as_backtrader_feeds`, `as_vectorbt_panel`.
- **Client ↔ TEE wire protocol (Workstream D + E)**:
    - `sablier_flow.client.crypto` — X25519 ECDH + HKDF-SHA256 + AES-256-GCM envelope encryption with `EnvelopeEncrypted.to_bytes` / `from_bytes` wire format.
    - `sablier_flow.client.attestation` — `AttestationVerifier` enforces pinned image digest, TEE type / hardware, measurements, freshness, and signature presence. Modes: `production` (strict) and `fake-for-dev` (staging).
    - `sablier_flow.client.transport` — Pydantic wire models (`CreateJobRequest`, `AttestationQuoteResponse`, `JobStatusResponse`, `ResultResponse`), `Transport` Protocol, `HttpxTransport` (real HTTPS), `InMemoryTransport` (in-process fake).
    - `sablier_flow.client.payload` — `JobUploadPayload` (Parquet + params + 32B AES key) and `JobResultPayload` (numpy arrays + metadata); both length-prefixed binary with magic + version.
    - `sablier_flow.Client.alternative_versions` — full lifecycle: POST /v1/jobs → verify attestation → envelope-encrypt → upload → poll → decrypt result.
- **TEE-side mirror**:
    - `server.tee.crypto.TEEKeyState` — per-boot X25519 keypair holder.
    - `server.tee.attestation.generate_attestation_quote` — wire-format quote generator.
    - `server.tee.runner.run_job_real` — real-pipeline runner (train + generate inside the Confidential VM).
- **FastAPI control plane (`server/api/`)**: `POST /v1/jobs`, `PUT /v1/jobs/{id}/data`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/result`, plus `/health` and `/v1/version`. Thread-safe `JobStore` with injected keypair factory + quote generator + runner so dev/test use a mock pipeline and production uses the torch-backed runner.
- **HXZ 452-anomaly validation harness scaffolded** (`benchmarks/hxz/`): `run_study.py` per-anomaly study + `analyze.py` Gate 2 verdict (Spearman ≥ 0.75, median abs Sharpe error ≤ 0.15, overfit-catch rate ≥ 80%).
- **End-to-end tests** (no live server, no torch, no real TEE):
    - `tests/integration/test_attestation_handshake.py` — full client↔TEE handshake roundtrip.
    - `tests/integration/test_client_end_to_end.py` — Client → InMemoryTransport → fake TEE → GenerationResult.
    - `tests/integration/test_server_end_to_end.py` — Client → HttpxTransport → real FastAPI app via TestClient → GenerationResult.
- **Demo notebook** `examples/01_alternative_versions.ipynb`.

[Unreleased]: https://sablier.ai/flow
