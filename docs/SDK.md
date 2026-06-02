# sablier-flow SDK — Full Reference

> Canonical reference for the `sablier-flow` Python SDK. Type signatures and
> code examples are copy-paste runnable against the current PyPI release
> (`sablier_flow.__version__`). If a behavior or API you expect is not
> documented here, it does not exist in the SDK yet — don't hallucinate
> features.

---

## What sablier-flow does

The customer has a backtest function `f(prices) -> {"sharpe": ...}`. They run it on their real history and want to know whether the result is genuine signal or overfit to the specific realization their data took. `sablier-flow` answers that by:

1. Training a generative model on the customer's history on a remote GPU worker (see [Security posture](#security-posture-today-alpha) for the current and target deployment specifics).
2. Generating `N` synthetic *alternative versions* of the same history — different paths, same statistical fingerprint.
3. Running the customer's backtest on every synthetic alt-history.
4. Comparing the real result to the distribution of synthetic results.

If the real result sits at the extreme tail of the synthetic distribution, the strategy is exploiting realization-specific noise — **overfit**. If it sits in the bulk, the strategy is **robust**.

Two additional outputs surface for serious quants:
- **Deflated Sharpe Ratio (DSR)** under two nulls (empirical synthetic-best-of-N + analytical Bailey–LdP IID-Gaussian).
- **Probability of Backtest Overfitting (PBO)** via Combinatorially Symmetric Cross-Validation on the real history alone.

---

## Contents

- [Installation](#installation)
- [Authentication](#authentication)
- [Security posture today (alpha)](#security-posture-today-alpha)
- [The workflow: `fit` → `generate` → `validate`](#the-workflow-fit-generate-validate)
- [Forward generation](#forward-generation-deployment-forecasting)
- [Strategy families](#strategy-families)
- [Interpreting the output](#interpreting-the-output)
- [Demo datasets](#demo-datasets)
- [Adapters + model management](#adapters-model-management)
- [Full API reference](#full-api-reference)
- [Common errors](#common-errors)
- [Versioning](#versioning)

---

## Installation

```bash
pip install sablier-flow                            # thin client only (~30 MB)
pip install 'sablier-flow[adapters-backtrader]'     # + backtrader integration
pip install 'sablier-flow[adapters-vectorbt]'       # + vectorbt integration
```

Python 3.10 or newer.

---

## Authentication

Sign up at <https://sablier.ai> (email/password or Google OAuth — both work; verify your email if you used password). Then authenticate one of three ways, resolved in this order:

1. **Explicit kwarg** — `sf.fit(real, api_key="sk_live_...")` or `sf.Client(api_key=...)`. Always wins.
2. **`SABLIER_FLOW_API_KEY` env var** — set in CI, containers, headless scripts.
3. **`~/.sablier/credentials` file** — written by `sf.login()` for interactive use.

```python
import sablier_flow as sf
sf.login()              # opens browser, prompts Authorize, writes ~/.sablier/credentials (mode 0600)
sf.Client()             # auto-picks the stored credential from then on
sf.logout()             # drops the local credential (does NOT revoke server-side; use dashboard)
```

For non-interactive use: `export SABLIER_FLOW_API_KEY=sk_live_...` and skip `sf.login()` entirely.

---

## Security posture today (alpha)

TLS 1.3 in transit, KMS-encrypted at rest, one-shot per-job symmetric keys, image-digest pinning on every request. Hardware memory encryption (AMD SEV-SNP + NVIDIA H100 CC mode) is on the roadmap — until that ships, plaintext customer data exists in worker RAM during the minutes-long training job

---

## The workflow: `fit` → `generate` → `validate`

The SDK splits the lifecycle into three explicit calls so you train once and reuse the trained model across as many windows / strategies as you want.

```python
# 1. Train once (~minutes, scales with data size). The server splits 80/20 with
#    a 21-bar embargo by default and keeps the held-out OOS slice encrypted
#    alongside the model so sf.validate(model_id) picks it up automatically.
fit = sf.fit(
    real,
    features=list(real.columns),         # all columns are co-generated jointly
    data_types=real.attrs["data_types"], # per-column transform annotation
    horizon=252,                         # training-window length (bars, not days)
    train_split=0.8,                     # 80% train, 20% OOS held out for validate()
    embargo_days=21,                     # bar gap between train end + OOS start
    seed=42,
)
print(fit.model_id)                 # opaque handle; pass to generate / validate / get_model
print(fit.training_loss, fit.loss_source)
# loss_source ∈ {'validation', 'training_proxy'} — the latter means the inner
# val split was too small to form a single (obs_length + horizon) window, so
# the loss reported is the training-loss proxy; the real OOS check still
# happens via sf.validate(...) on the persisted holdout.

# 2. Validate the model on the held-out OOS slice (zero-config — no holdout
#    DataFrame argument needed). Returns a ValidationReport with `overall`,
#    `memorization_risk`, and ~20 per-metric entries.
report = sf.validate(fit.model_id)

# 3. Generate N synthetic paths shaped like any window you want. `like=df`
#    derives length + index + price anchor from the window — synth paths
#    overlay your real series directly.
paths = sf.generate(fit.model_id, n_paths=1000, like=backtest_window,
                    data_types=real.attrs["data_types"], seed=42)
```

### Schema contract — what `real_data` must look like

| Field | Requirement |
|---|---|
| `df.index` | `pd.DatetimeIndex`, monotonic increasing, no duplicates (tz-naive or tz-aware) |
| `df.columns` | numeric dtype on every column listed in `features=`; NaNs masked, columns with post-ffill NaN fraction > 0.7 rejected |
| `data_types=` (kwarg) | **required dict** mapping every `features=` column to one of `{'price', 'level', 'return'}`. Bundled demos attach the canonical map on `df.attrs['data_types']`. |
| Row cadence | auto-detected from median Δt; any uniform cadence accepted (daily, intraday, weekly, monthly, quarterly); irregular indices raise |
| Length | ≥ 200 rows on `fit`; shorter slices allowed for `like=` / `anchor_data=` / `holdout_data=` |

### Async path

Every sync method has an async sibling returning a `JobHandle` (carries `job_id`, kind, and one-shot result key). The handle survives process restarts via `handle.to_dict()` / `JobHandle.from_dict(...)`. Treat it as a bearer secret.

```python
handle = sf.fit_async(real, features=list(real.columns), data_types=real.attrs["data_types"], horizon=252)
result = sf.fetch_result(handle)        # blocks until done; FitResult/GenerationResult/ValidationReport by kind
sf.list_jobs(status="running"); sf.cancel_job(handle)
```

---

## Forward generation — deployment forecasting

Same generator, different anchor: instead of paralleling a past window, project forward from your most recent bar.

| Use case | Call | Anchor |
|---|---|---|
| Alt-history (overfit audit) | `sf.generate(model_id, like=backtest_window)` | `like.iloc[0]` |
| Forward forecast (deployment) | `sf.generate(model_id, horizon=N, anchor_data=real.iloc[-200:])` | `anchor_data.iloc[-1]` ("today") |

```python
forward = sf.generate(fit.model_id, n_paths=1000, horizon=60,
                      anchor_data=real.iloc[-200:],
                      data_types=real.attrs["data_types"])
forward_sharpes = np.array([my_backtest(df)["sharpe"] for df in forward.as_dataframes()])
print(f"median: {np.median(forward_sharpes):+.2f}, 90% CI: "
      f"[{np.percentile(forward_sharpes, 5):+.2f}, {np.percentile(forward_sharpes, 95):+.2f}]")
```

`sf.predictive_rank_score(real_sharpes, synth_sharpes)` returns a Spearman ρ + bootstrap CI + verdict (`well_calibrated` / `weakly_calibrated` / `uncalibrated` / `inverted`) testing whether your strategy ranking on synth forwards predicts the ranking on real OOS data. See [notebook 02](examples/02_tstr_predictive_rank.ipynb) for the worked example.

---

## Strategy families

For multiple strategy variants tested simultaneously, `sf.evaluate_family(strategies_dict, real, n_paths=100)` runs every strategy on every synthetic path and returns family-best DSR + CSCV PBO. See [notebook 01](examples/01_backtest_robustness.ipynb) for the worked example. Standalone PBO is `sf.probability_of_backtest_overfitting(strategies, real)`.

---

## Interpreting the output

### `RobustnessReport.verdict`

| Bucket | `overfit_score` | Meaning |
|---|---|---|
| `robust` | `[0.00, 0.70)` | Real result is consistent with the synthetic distribution. No overfit signal. |
| `borderline` | `[0.70, 0.85)` | Real result is in the top quartile of synth. |
| `overfit` | `[0.85, 0.95)` | Real result exceeds 85%+ of synthetic alt-histories. |
| `highly_overfit` | `[0.95, 1.00]` | Real result is in the top 5%. |

For higher-is-better metrics, `overfit_score = mean(synthetic < real)`. `robust` is orthogonal to profitable — read the Sharpe sign separately. `RobustnessReport.summary()` returns a plain-English sentence including any structural / memorization warnings.

### Deflated Sharpe Ratio

```python
dsr = verdict.deflated_sharpe(strategy_returns=daily_returns, n_trials=1)
dsr.realistic, dsr.analytical          # DSR under realistic (regime-aware) + Bailey-LdP IID-Gaussian nulls
dsr.expected_max_sr_realistic, dsr.expected_max_sr_analytical
dsr.threshold_sr_realistic, dsr.threshold_sr_analytical
```

For a family of M strategies, pass `n_trials=M` (or use `evaluate_family` which handles it).

### PBO (Probability of Backtest Overfitting)

Computed on real history alone via Combinatorially Symmetric Cross-Validation (Bailey et al. 2015).

| `pbo` | Interpretation |
|---|---|
| ≤ 0.2 | Grid search has signal. |
| ~ 0.5 | No signal — parameter selection is noise. |
| ≥ 0.6 | Systematic overfitting. |

### Memorization risk

| `memorization_nn_distance_ratio` | `memorization_risk` | Action |
|---|---|---|
| `> 0.80` | `low` | Synth distributed through the training manifold. |
| `[0.50, 0.80]` | `medium` | Cross-check against `coverage_*` metrics. |
| `< 0.50` | `high` | Model is regurgitating. Don't trust the overfit verdict on top. |

### Structural validation

`ValidationReport.overall ∈ {'pass', 'warn', 'fail'}` aggregates ~20 per-metric scores grouped into calibration, distribution, dependence, temporal, and extreme categories. The full per-metric breakdown is on `ValidationReport.metrics`; the underlying metric suite + thresholds are documented at [github.com/sablier-ai/finval](https://github.com/sablier-ai/finval).

---

## Demo datasets

```python
sf.demo_data()                                       # default: us_equities_macro_2010_2023
sf.demo_data("us_equities_2010_2023")                # SPY/QQQ/IWM/TLT only, no macros
sf.demo_data("us_equities_macro_5min_3mo")           # 5-min intraday — 7 tickers, 3 months
sf.available_demo_datasets()                         # list all bundled names
```

Bundled parquets ship inside the wheel (`pip install sablier-flow` includes them). Zero network access required to load.

---

## Adapters + model management

Engine adapters live under `sablier_flow.adapters` — `as_dataframes`, `as_array`, `as_backtrader_feeds` (extra: `[adapters-backtrader]`), `as_vectorbt_panel` (extra: `[adapters-vectorbt]`), `write_lean_csv_universe`. See the [getting-started notebook](examples/00_getting_started.ipynb) for a worked example.

Fitted models persist server-side for ~30 days. Manage via `sf.list_models()`, `sf.get_model(model_id)`, `sf.delete_model(model_id)`. Full signatures below.

---

## Full API reference

Every signature is verbatim from the source.

### `Client`

```python
sf.Client(
    api_key: str,
    *,
    endpoint: str | None = None,            # falls back to "https://flow.sablier.ai/v1"
    pinned_image_digest: str | None = None,
    attestation_mode: str = "production",   # "production" | "fake-for-dev"
    transport: Transport | None = None,     # for tests / in-process simulation
    timeout_s: float = 60.0,
    poll_interval_s: float = 2.0,
    poll_timeout_s: float = 30 * 60,
    verify: bool | str | None = None,       # None/True = system CA; False = skip; str = pin PEM
    cache_dir: str | os.PathLike | bool | None = None,
)
```

Methods:

```python
Client.fit(
    real_data: pd.DataFrame,
    *,
    data_types: dict[str, str],               # REQUIRED — per-column annotation: 'price' | 'level' | 'return'
    features: Sequence[str] | None = None,    # default: every numeric column of real_data
    horizon: int | None = None,
    train_split: float | None = 0.8,          # set to None to skip the OOS split
    embargo_days: int = 21,
    seed: int | None = None,
    quiet: bool = False,                      # suppress the stderr cost-estimate / actual-cost lines
    idempotency_key: str | None = None,
) -> FitResult

Client.generate(
    model_id: str,
    *,
    n_paths: int = 1000,
    horizon: int | None = None,               # any length; defaults to training horizon
    anchor_data: pd.DataFrame | None = None,  # None → use server-stored training tail
    like: pd.DataFrame | None = None,         # convenience: derive horizon + index + anchor from this window
    data_types: dict[str, str] | None = None, # required when `like=` or `anchor_data=` is set (carries fresh data)
    seed: int | None = None,
    quiet: bool = False,
    idempotency_key: str | None = None,
) -> GenerationResult

Client.validate(
    model_id: str,
    *,
    holdout_data: pd.DataFrame | None = None, # None → use the OOS slice persisted at fit time
    data_types: dict[str, str] | None = None, # required when holdout_data is set
    n_paths: int = 500,
    seed: int | None = None,
    quiet: bool = False,
    idempotency_key: str | None = None,
) -> ValidationReport

Client.list_models(*, limit: int = 50) -> list[Model]
Client.get_model(model_id: str) -> Model
Client.delete_model(model_id: str) -> None
```

### Module-level shortcuts

Core workflow:

```python
sf.fit(real_data, *, api_key=None,
       features=None, data_types,                     # data_types REQUIRED
       horizon=None,
       train_split=0.8, embargo_days=21, seed=None,
       idempotency_key=None,
       # connection-shape kwargs (env-var fallback) ───────────────
       endpoint=None, pinned_image_digest=None,
       attestation_mode="production", verify=None,
       cache_dir=None, profile="default") -> FitResult

sf.generate(model_id, *, api_key=None,
            data_types,                                # data_types REQUIRED
            n_paths=1000, horizon=None,
            anchor_data=None, like=None, seed=None,
            idempotency_key=None,
            endpoint=None, pinned_image_digest=None,
            attestation_mode="production", verify=None,
            cache_dir=None, profile="default") -> GenerationResult

sf.validate(model_id, *, api_key=None,
            data_types=None,                           # REQUIRED only when holdout_data is supplied
            holdout_data=None, n_paths=500, seed=None,
            idempotency_key=None,
            endpoint=None, pinned_image_digest=None,
            attestation_mode="production", verify=None,
            cache_dir=None, profile="default") -> ValidationReport
```

`data_types` is a `dict[str, str]` mapping every column in `features=` to one of `{'price', 'level', 'return'}`. Missing the kwarg raises `TypeError` with the allowed-set message; an unknown value raises `ValueError`. Demo DataFrames attach the canonical map on `df.attrs['data_types']`. On `sf.validate(model_id)` without `holdout_data` the server reuses the `data_types` registered at fit time — passing the kwarg in that mode is a no-op and is silently ignored.

Async workflow:

```python
sf.fit_async(real_data, ...)        -> JobHandle      # same kwargs as sf.fit
sf.generate_async(model_id, ...)    -> JobHandle      # same kwargs as sf.generate
sf.validate_async(model_id, ...)    -> JobHandle      # same kwargs as sf.validate
sf.fetch_result(handle)             -> FitResult | GenerationResult | ValidationReport
sf.list_jobs(*, status=None, limit=50, api_key=None, **kw) -> list[JobSummary]
sf.cancel_job(handle_or_id, *, api_key=None, **kw)         -> None
```

Predictive validity (post-hoc analytic; pure numpy + scipy, no path generation):

```python
sf.predictive_rank_score(real_results, synth_results, *,
                         primary_metric=None,
                         n_bootstrap=10000,
                         seed=0) -> PredictiveRankReport
```

Auth + credentials:

```python
sf.login(*, endpoint=None, profile="default",
         open_browser=True, poll_timeout_s=600.0,
         verify=None) -> LoginResult
sf.logout(*, profile="default") -> bool                 # True if a profile was dropped
```

Model management:

```python
sf.list_models(*, limit=50, api_key=None, **kw) -> list[Model]
sf.get_model(model_id, *, api_key=None, **kw)   -> Model
sf.delete_model(model_id, *, api_key=None, **kw) -> None
```

Account / pre-flight:

```python
sf.ping(*, api_key=None, **kw)         -> dict[str, Any]
sf.whoami(*, api_key=None, **kw)       -> dict[str, Any]
sf.credits(*, api_key=None, **kw)      -> CreditsBalance      # Pydantic — use attribute access (balance.available, .monthly_used, ...)
sf.usage(*, since=None, until=None, kind=None, limit=100, api_key=None, **kw) -> list[UsageEvent]
sf.usage_summary(*, period="month", api_key=None, **kw) -> UsageSummary   # Pydantic — summary.total_credits, .by_kind, ...
sf.estimate_cost(kind, *, real_data=None, features=None, horizon=None, n_paths=None, n_features=None, n_rows=None, api_key=None, **kw) -> dict[str, Any]
    # Returns {estimated_credits, low, high, notes}. Wall-clock duration is NOT returned.
    # `kind` must be one of 'fit' | 'generate' | 'validate' — 'train' is rejected.
```

Local helpers (no network):

```python
sf.validate_data(real_data) -> None      # raise on schema violations BEFORE the network round-trip
sf.demo_data(name="us_equities_macro_2010_2023") -> pd.DataFrame
sf.available_demo_datasets() -> list[str]
```

Each shortcut constructs a one-shot `Client`. Connection-shape settings (`endpoint`, `verify`, `pinned_image_digest`, `attestation_mode`) fall back to env vars `SABLIER_FLOW_ENDPOINT`, `SABLIER_FLOW_CERT`, `SABLIER_FLOW_PINNED_IMAGE_DIGEST`, `SABLIER_FLOW_ATTESTATION_MODE`. `api_key` falls back to `SABLIER_FLOW_API_KEY`, then to `~/.sablier/credentials` (written by `sf.login()`).

Unknown kwargs raise `TypeError` with the offending name (no `**kwargs` swallow), so IDE autocomplete and `inspect.signature()` see the real parameter list.

### `JobHandle`

Returned by `sf.fit_async` / `sf.generate_async` / `sf.validate_async`. Persistable across processes via `to_dict()` / `from_dict(d)`.

```python
@dataclass(frozen=True)
class JobHandle:
    job_id: str
    kind: str                # 'fit' | 'generate' | 'validate'
    result_key_b64: str      # standard-base64 of the AES-256-GCM key — treat as a secret

    def to_dict(self) -> dict[str, str]: ...
    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "JobHandle": ...
```

Pair with `sf.fetch_result(handle)` to block on completion and materialize the typed result.

### `FitResult`

```python
@dataclass(frozen=True)
class FitResult:
    model_id: str
    features: list[str]
    training_horizon: int
    training_end_date: str | None
    sdk_version: str
    expires_at: str | None = None
    training_loss: float | None = None
    loss_source: str | None = None             # 'validation' | 'training_proxy'
    training_start_date: str | None = None
    holdout_start_date: str | None = None
    holdout_end_date: str | None = None
```

### `Model`

```python
@dataclass(frozen=True)
class Model:
    model_id: str
    features: list[str]
    training_horizon: int
    n_assets: int
    status: str                                # 'ready' | 'failed' | 'expired'
    training_start_date: str | None = None
    training_end_date: str | None = None
    holdout_start_date: str | None = None
    holdout_end_date: str | None = None
    train_split: float | None = None
    embargo_days: int | None = None
    sdk_version: str | None = None
    training_loss: float | None = None
    created_at: str | None = None
    last_used_at: str | None = None
    expires_at: str | None = None
```

### `GenerationResult`

```python
@dataclass(frozen=True)
class GenerationResult:
    paths_returns: np.ndarray                  # (n_paths, horizon, n_features), z-scored
    paths_prices: np.ndarray                   # (n_paths, horizon, n_features), price-level
    feature_names: list[str]                   # original input columns only
    last_prices: np.ndarray
    horizon: int
    n_paths: int
    seed: int | None
    sdk_version: str
    memorization_risk: str | None              # 'low' | 'medium' | 'high'
    memorization_nn_distance_ratio: float | None
    paths_index: pd.DatetimeIndex | None       # set when generate was called with like=window

    def as_dataframes(self, index=None) -> list[pd.DataFrame]: ...
```

### `ValidationReport`

```python
@dataclass(frozen=True)
class ValidationReport:
    overall: str                               # 'pass' | 'warn' | 'fail'
    metrics: dict[str, Any]                    # per-metric breakdown (see "Structural validation")
    memorization_risk: str | None              # 'low' | 'medium' | 'high'
    memorization_nn_distance_ratio: float | None
    n_paths_used: int | None
    holdout: bool = False                      # True when validated against a held-out OOS slice
```

### `robustness`

```python
sf.robustness(
    real_result: float | dict[str, float],
    synthetic_results: Sequence[float | dict[str, float]],
    *,
    primary_metric: str | None = None,
    higher_is_better: bool = True,
) -> RobustnessReport
```

### `RobustnessReport`

```python
@dataclass(frozen=True)
class RobustnessReport:
    overfit_score: float
    verdict: Literal["robust", "borderline", "overfit", "highly_overfit"]
    primary_metric: str
    real_value: float
    synthetic_mean: float
    synthetic_median: float
    synthetic_std: float
    synthetic_min: float
    synthetic_max: float
    synthetic_p5: float
    synthetic_p25: float
    synthetic_p75: float
    synthetic_p95: float
    synthetic_ci_95: tuple[float, float]
    n_synthetic: int
    per_metric: dict[str, dict[str, float]]
    notes: list[str]
    synthetic_values: tuple[float, ...]        # raw per-path values
    higher_is_better: bool

    @property
    def acceptable(self) -> bool: ...          # True for 'robust' or 'borderline'
    def summary(self) -> str: ...
    def deflated_sharpe(self, *, strategy_returns=None, n_trials=1) -> DeflatedSharpeReport: ...
    def to_html(self, path=None, *, title="Robustness Report") -> str: ...
```

### `deflated_sharpe`

```python
sf.deflated_sharpe(
    *,
    observed_sr: float,
    synthetic_sharpes: Sequence[float] | np.ndarray,
    strategy_returns: Sequence[float] | np.ndarray | None = None,
    n_trials: int = 1,
    significance_level: float = 0.95,
) -> DeflatedSharpeReport
```

### `DeflatedSharpeReport`

```python
@dataclass(frozen=True)
class DeflatedSharpeReport:
    observed_sr: float
    n_trials: int
    realistic: float                           # DSR under Sablier synthetic-best-of-N null
    analytical: float                          # DSR under Bailey-LdP IID-Gaussian null
    expected_max_sr_realistic: float
    expected_max_sr_analytical: float
    threshold_sr_realistic: float              # SR needed for DSR=0.95 (realistic)
    threshold_sr_analytical: float             # ... (analytical)

    def to_dict(self) -> dict: ...
```

### `evaluate_family`

```python
sf.evaluate_family(
    strategies: Mapping[str, Callable[..., float | dict]],
    real_data: pd.DataFrame,
    *,
    n_paths: int = 100,
    primary_metric: str | None = None,
    higher_is_better: bool = True,
    pbo_cscv_splits: int = 16,                 # SDK floor; lower values under-detect overfit
    executor: Literal["serial", "thread"] = "serial",
    max_workers: int | None = None,
    progress: bool = False,
    raise_on_failure: bool = False,
    **fit_or_generate_kwargs,                  # features, horizon, seed, train_split, etc.
) -> FamilyReport
```

### `FamilyReport`

```python
@dataclass(frozen=True)
class FamilyReport:
    strategy_names: tuple[str, ...]
    primary_metric: str
    real_metrics: tuple[dict[str, float], ...]
    synthetic_metrics: tuple[tuple[dict[str, float], ...], ...]
    real_max_value: float
    real_argmax_strategy: str
    synthetic_max_values: np.ndarray           # (n_paths,) best-of-N per path
    deflated_sharpe: DeflatedSharpeReport
    pbo: float
    pbo_n_partitions: int
    pbo_cscv_splits: int
    n_paths: int
    per_strategy_real_metric: dict[str, float]
    per_strategy_overfit_score: dict[str, float]
    per_strategy_synthetic_median: dict[str, float]
    failures: tuple[str, ...]
    notes: tuple[str, ...]

    def summary(self) -> str: ...
    def most_overfit_variants(self, *, top: int = 5) -> list[tuple[str, float]]: ...
    def to_dict(self) -> dict: ...
```

### `probability_of_backtest_overfitting`

```python
sf.probability_of_backtest_overfitting(
    strategies: Mapping[str, Callable[..., float | dict]],
    real_data: pd.DataFrame,
    *,
    primary_metric: str = "sharpe",
    higher_is_better: bool = True,
    cscv_splits: int = 16,
    executor: Literal["serial", "thread"] = "serial",
    max_workers: int | None = None,
) -> tuple[float, int]                          # (pbo_value, n_partitions_used)
```


### `PredictiveRankReport`

```python
@dataclass(frozen=True)
class PredictiveRankReport:
    spearman_rho: float                       # rank correlation real vs synth-forward
    p_value: float                            # scipy.stats.spearmanr two-sided p
    ci_95: tuple[float, float]                # bootstrap percentile CI (10000 resamples)
    n_strategies: int                         # intersection of {real_results, synth_results}
    mean_abs_metric_gap: float                # magnitude bias (rank can be right while abs is biased)
    primary_metric: str                       # 'sharpe' if dicts; 'value' if scalars
    real_values: dict[str, float]
    synth_values: dict[str, float]
    n_bootstrap: int
    notes: list[str]

    @property
    def verdict(self) -> Literal[
        "well_calibrated", "weakly_calibrated", "uncalibrated", "inverted",
    ]: ...
    @property
    def acceptable(self) -> bool: ...         # True if verdict ∈ {well_calibrated, weakly_calibrated}
    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
```

Lower-level primitives (`sf.AttestationVerifier`, `sf.envelope_encrypt`, `sf.envelope_decrypt`, `AttestationQuote`, `EnvelopeEncrypted`) are exported for custom-transport implementers. `Client` invokes them internally on every request — the standard workflow never touches them.

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `ValueError: api_key is required` | `SABLIER_FLOW_API_KEY` not set and not passed as kwarg | `export SABLIER_FLOW_API_KEY=sk_live_...` |
| `ValueError: real_data.index must be a pd.DatetimeIndex` | DataFrame index is not date-like | `df.index = pd.to_datetime(df.index)` |
| `ValueError: real_data has N rows; need at least 200` | Training data too short | Use a longer history |
| `ValueError: real_data has non-numeric columns: [...]` | Non-numeric column present | Drop / convert before passing |
| `ValueError: synthetic_results is empty` | `robustness()` called with no synth results | Generate paths first via `sf.generate(...)` |
| `ValueError: deflated_sharpe is only defined for higher-is-better metrics` | Called `.deflated_sharpe()` on a drawdown-style report | DSR is for return-style metrics. Use the raw `overfit_score` for lower-is-better. |
| `TransportError: retry budget exhausted` | Network blip during status polling exceeded retry window | The worker may still be training — call `sf.list_models()` to recover the model_id and resume with `sf.validate(model_id)` |
| `AttestationVerificationError` | Image digest mismatch between SDK and TEE | The SDK release pins a TEE image digest; if they disagree either upgrade the SDK or wait for the matching server rollout |
| `RemoteJobError` | Worker failed inside the TEE | Re-raise carries the safe error message from the worker; inspect and retry if transient |

---

## Versioning

The public API follows semantic versioning. Major releases (`X.0.0`) may
introduce breaking changes; minor (`X.Y.0`) and patch (`X.Y.Z`) releases
preserve backwards compatibility. The current version is exposed at
`sablier_flow.__version__`.

See [PyPI](https://pypi.org/project/sablier-flow/) for the latest release, the [CHANGELOG](https://github.com/sablier-ai/sablier-flow/blob/main/CHANGELOG.md) for the full history, and [GitHub releases](https://github.com/sablier-ai/sablier-flow/releases) for per-release notes.

Pin behaviour you care about explicitly.
