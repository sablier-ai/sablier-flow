"""Public wire dataclasses shipped in the ``sablier-flow`` thin client.

Anything in this module is safe to import without any heavy GPU deps —
``pip install sablier-flow`` pulls only pandas + numpy + httpx +
cryptography + pydantic. The dataclasses here describe what crosses
the wire between the customer SDK and the hosted TEE:

  - :class:`FitResult`        — returned by :meth:`Client.fit`
  - :class:`GenerationResult` — returned by :meth:`Client.generate`
  - :class:`ValidationReport` — returned by :meth:`Client.validate`

Report-uniformity protocol
--------------------------
Every report dataclass shipped by sablier-flow exposes the same two
attributes so dashboards, CI gates, and notebook cells can iterate
over heterogeneous reports without special-casing:

  - ``verdict``    — a short string label (``'pass'``/``'warn'``/``'fail'``
    on :class:`ValidationReport`; a per-class verdict bucket on the
    analytics reports). Where a class historically stored the verdict
    under a different field name (e.g. :class:`ValidationReport`'s
    ``overall``), ``verdict`` is exposed as a ``@property`` alias.
  - ``acceptable`` — a ``bool`` (or ``bool | None`` for the
    insufficient-data case on :class:`RobustnessReport`) answering
    'is this finding green?'. The exact bucketing per report type
    is documented on each class's ``acceptable`` property.

This is a duck-typed contract — there's no abstract base class — but
the unit tests in ``tests/unit/test_types_1_0_7.py`` assert every
report type satisfies it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel

__all__ = [
    "CreditsBalance",
    "FitResult",
    "GenerationResult",
    "JobHandle",
    "Model",
    "UsageEvent",
    "UsageSummary",
    "ValidationReport",
]


# Canonical job-kind vocabulary. Both :meth:`JobHandle.__post_init__` and
# :meth:`JobHandle.from_dict` reject anything outside this set so the legacy
# ``'train'`` wire word (pre-1.0.8) and arbitrary strings can't sneak in via
# persisted handles or hand-built dicts.
_ALLOWED_JOB_KINDS = frozenset({"fit", "generate", "validate"})


# ============================================================================
# Account response models — mirror the wire shape of the server's
# ``/v1/credits``, ``/v1/usage``, ``/v1/usage/summary`` routes. See
# ``backend/api/flow_sdk/account.py`` for the source of truth on field names.
# Attribute access only (``c.monthly_used``); subscripting raises TypeError.
# ============================================================================


class CreditsBalance(BaseModel):
    """Current credit balance for the authenticated user. Returned by
    :meth:`Client.credits` / :func:`sablier_flow.credits`.

    Mirrors the server's ``CreditsResponse`` exactly so customer code
    that introspects the shape ports straight across. All values are
    integer credit units (the unit the dashboard displays).
    """

    available: int
    """Credits available to spend NOW
    (``monthly_allocation - monthly_used + purchased``)."""

    monthly_allocation: int
    """Credits granted by the current subscription tier per billing period."""

    monthly_used: int
    """Credits already used during the current billing period."""

    purchased: int
    """One-time purchased credits remaining (carry over across periods)."""

    tier: str
    """Current subscription tier — 'free' / 'pro' / 'enterprise' / etc."""


class UsageEvent(BaseModel):
    """One row of :meth:`Client.usage` — a single flow-SDK job with its
    credit accounting attached. Cancelled / failed jobs that incurred no
    charge still appear here for transparency. Mirrors the server's
    ``UsageItem``.
    """

    job_id: str
    kind: str
    """``'fit'`` | ``'generate'`` | ``'validate'``."""

    status: str
    """``'queued'`` | ``'running'`` | ``'completed'`` | ``'failed'``."""

    credits_charged: float
    n_assets: int | None = None
    created_at: str
    completed_at: str | None = None
    duration_s: float | None = None


class UsageSummary(BaseModel):
    """Aggregate flow-SDK usage over a configurable window. Returned by
    :meth:`Client.usage_summary` / :func:`sablier_flow.usage_summary`.

    Mirrors the server's ``UsageSummaryResponse`` — ``by_kind`` maps each
    job-type to ``{'n_jobs': int, 'credits': float}``.
    """

    period_start: str
    """ISO timestamp — start of the aggregation window."""

    period_end: str
    """ISO timestamp — end of the aggregation window."""

    total_credits: float
    """Sum of ``credits_charged`` across every job in the window."""

    by_kind: dict[str, dict[str, Any]]
    """Map ``kind -> {'n_jobs': int, 'credits': float}``. Only kinds
    with at least one job in the window appear."""


@dataclass(frozen=True)
class JobHandle:
    """A handle to an async job. Returned by ``Client.fit_async`` /
    ``Client.generate_async`` / ``Client.validate_async``.

    The handle is the only state a caller needs to retrieve the result
    later — including from a different Python process. It carries the
    job id, the job kind (so :meth:`Client.fetch_result` knows what
    dataclass to materialize), and the AES-GCM ``result_key`` the TEE
    will encrypt the result with. Persist it via :meth:`to_dict` and
    reconstitute via :meth:`from_dict` to survive interpreter restarts.

    The result_key is a one-shot symmetric key generated client-side at
    job-open time; it never leaves the customer's machine except inside
    the envelope-encrypted upload that only the TEE can decrypt. Anyone
    holding the handle can fetch the result, so treat it like a secret.

    ``kind`` canonical vocabulary: ``'fit'`` | ``'generate'`` |
    ``'validate'``.
    """

    job_id: str
    kind: str  # 'fit' | 'generate' | 'validate'
    result_key_b64: str
    """Standard-base64 of the 32-byte AES-256-GCM key. The TEE encrypts
    the result blob with this key; the client decrypts locally."""

    def __post_init__(self) -> None:
        """Enforce the canonical ``kind`` vocabulary at construction time
        so direct ``JobHandle(...)`` calls fail the same way
        :meth:`from_dict` does on a stale ``'train'`` handle.
        """
        if self.kind not in _ALLOWED_JOB_KINDS:
            raise ValueError(
                f"JobHandle.kind must be one of {sorted(_ALLOWED_JOB_KINDS)}; "
                f"got {self.kind!r}"
            )

    def __repr__(self) -> str:
        """Redact ``result_key_b64`` so ``print(handle)`` / ``%r`` /
        ``logger.info('%r', handle)`` / Jupyter cell-output persistence
        do NOT leak the one-shot AES-256-GCM key that decrypts the
        TEE-side result. Anyone holding the cleartext key plus the
        ``job_id`` can fetch and decrypt the customer's synthetic-paths
        result. Use :meth:`to_dict` explicitly when you actually need to
        cross-process the handle (1.0.21)."""
        return (
            f"JobHandle(job_id={self.job_id!r}, kind={self.kind!r}, "
            f"result_key_b64=<redacted-len={len(self.result_key_b64)}>)"
        )

    def to_dict(self) -> dict[str, str]:
        """Serialize to a plain dict (e.g. for JSON persistence).

        Includes ``result_key_b64`` in cleartext — this is intentional
        because cross-process resume requires it. Persist to a file with
        mode 0600 and treat the bytes the same way you would treat the
        API key itself."""
        return {"job_id": self.job_id, "kind": self.kind, "result_key_b64": self.result_key_b64}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> JobHandle:
        """Reconstitute from the dict produced by :meth:`to_dict`.

        Rejects any ``kind`` outside :data:`_ALLOWED_JOB_KINDS` — most
        notably the legacy ``'train'`` wire word from SDKs older than
        1.0.8, which the server no longer accepts.
        """
        kind = str(d["kind"])
        if kind not in _ALLOWED_JOB_KINDS:
            raise ValueError(
                f"JobHandle.kind must be one of {sorted(_ALLOWED_JOB_KINDS)}; "
                f"got {kind!r}"
            )
        return cls(
            job_id=str(d["job_id"]),
            kind=kind,
            result_key_b64=str(d["result_key_b64"]),
        )


@dataclass(frozen=True)
class FitResult:
    """Reference to a model trained inside the TEE.

    The trained weights stay server-side; the customer holds only a
    ``model_id`` plus the training metadata they need to know what the
    model expects. Subsequent :meth:`Client.generate` /
    :meth:`Client.validate` calls take ``model_id`` and don't re-upload
    data — see the SDK docs for the cost model.
    """

    model_id: str
    """Opaque identifier; pass to :meth:`Client.generate` /
    :meth:`Client.validate`. Scoped to the customer's organization on
    the server side."""

    features: list[str]
    """Feature columns the model was trained on, in column order. The
    model jointly generates every column — there is no target /
    conditioning distinction at the API level."""

    training_horizon: int
    """The window length used during training. Generation defaults to
    this length but is not capped — the flow model integrates over any
    horizon; quality is best near the trained value."""

    training_end_date: str | None
    """ISO date string of the last bar seen during training, or None if
    the input index was not date-like."""

    sdk_version: str
    """The CLIENT-side sablier-flow version that received this result.
    From 1.0.8 onward the client overwrites whatever the worker
    serialized so customers always see the SDK they imported. The
    worker's internal pipeline version (e.g. ``'0.5.1'``) is not
    customer-actionable and is intentionally hidden behind this stamp."""

    expires_at: str | None = None
    """ISO timestamp after which the server may garbage-collect this
    model. Any successful generate/validate call extends the TTL."""

    training_loss: float | None = None
    """Best loss observed during training. Lower is better, but the
    absolute value is only comparable across runs on the same universe.
    Read together with :attr:`loss_source` to know whether the number
    is a true held-out validation loss or a training-loss proxy."""

    loss_source: str | None = None
    """How :attr:`training_loss` was measured. One of:

      - ``'validation'`` — true held-out inner-validation loss (the
        default case, when ``train_split`` is ``None`` or the inner
        validation slice can form ``(obs_length, horizon)`` windows).
      - ``'training_proxy'`` — fallback when the inner validation slice
        is too small for windows (typical with ``train_split=0.8`` on
        short histories at long ``horizon``). The number reflects best
        training loss, not a held-out signal. The customer's real OOS
        check still happens via :meth:`Client.validate`, which uses
        the persisted holdout slice independently of this loss.

    ``None`` on FitResults from SDKs older than 0.3.0."""

    training_start_date: str | None = None
    """ISO date of the first bar seen during training. None if the
    input index was not date-like."""

    holdout_start_date: str | None = None
    """ISO date of the first bar of the OOS holdout. Set only when
    :meth:`Client.fit` was called with ``train_split`` (default 0.8).
    Subsequent :meth:`Client.validate` calls without an explicit
    ``holdout_data`` default to this server-stored slice."""

    holdout_end_date: str | None = None
    """ISO date of the last bar of the OOS holdout, paired with
    :attr:`holdout_start_date`."""


@dataclass(frozen=True)
class Model:
    """A fitted model in the customer's account.

    Returned by :meth:`Client.list_models` and :meth:`Client.get_model`.
    Strict superset of :class:`FitResult`: adds lifecycle metadata
    (``status``, ``created_at``, ``last_used_at``) that the customer
    cares about when deciding to reuse vs refit, but that doesn't make
    sense on the freshly-returned FitResult.

    Customers typically use it as:

        models = sf.list_models()
        equity_model = next(m for m in models if 'SPY' in m.features)
        paths = sf.generate(equity_model.model_id, like=window)
    """

    model_id: str
    """UUID identifying the model. Pass to :meth:`Client.generate`,
    :meth:`Client.validate`, :meth:`Client.delete_model`."""

    features: list[str]
    """Feature columns the model was trained on, in column order. All
    columns are co-generated jointly."""

    training_horizon: int
    """Window length the model was trained against. Generation can use
    shorter or longer horizons; quality is best near this value."""

    n_assets: int
    """Number of asset columns (real features only — internal cyclical
    dims are not counted)."""

    status: str
    """``'ready'`` | ``'failed'`` | ``'expired'``. Only ``'ready'``
    models are usable in :meth:`Client.generate` / :meth:`Client.validate`."""

    training_start_date: str | None = None
    training_end_date: str | None = None
    holdout_start_date: str | None = None
    holdout_end_date: str | None = None
    """ISO date strings (or ``None`` if the original input index was not
    date-like). The OOS holdout pair is set only when ``train_split`` was
    used at fit time."""

    train_split: float | None = None
    """The train/test split fraction the customer chose at fit time.
    ``None`` means the model was fitted on the full DataFrame
    (``train_split=None``)."""

    embargo_days: int | None = None
    """Bar gap between training end and OOS start. ``None`` when
    ``train_split`` was ``None``."""

    sdk_version: str | None = None
    """sablier-flow version that produced the checkpoint. Older
    versions may not deserialize cleanly if the schema changed."""

    training_loss: float | None = None
    """Best training loss. Read together with the model's loss_source
    in the FitResult that produced it — at list-time we just expose
    the number for sortability."""

    loss_source: str | None = None
    """How :attr:`training_loss` was measured. Same vocabulary as
    :attr:`FitResult.loss_source`:

      - ``'validation'`` — true held-out inner-validation loss.
      - ``'training_proxy'`` — fallback when the inner validation slice
        was too small for windows; the number reflects best training
        loss, not a held-out signal.

    May be ``None`` if the server did not record a loss source."""

    created_at: str | None = None
    last_used_at: str | None = None
    expires_at: str | None = None
    """Lifecycle timestamps. ``expires_at`` is refreshed on every
    successful generate/validate; after that the server may
    garbage-collect the encrypted blobs."""


@dataclass(frozen=True)
class ValidationReport:
    """Structural-validation verdict for a fitted model.

    Run on the training data (sanity) when ``holdout_data`` is omitted,
    or on a held-out OOS window the customer supplies. Cheap (~1 credit)
    since no training is involved.
    """

    overall: str
    """Aggregate verdict — ``'pass'`` (model is structurally sound),
    ``'warn'`` (acceptable but watch closely), or ``'fail'`` (synthetic
    drifted past safe thresholds; don't trust overfit verdicts built
    on top). Derived from the platform's weighted-quality score:
    EXCELLENT/GOOD → pass, ACCEPTABLE → warn, POOR → fail."""

    metrics: dict[str, Any]
    """Per-metric details from the full structural-validation suite.
    Same metric registry the platform's ``flow_validate`` uses, grouped
    by category:

      - **temporal**: ``acf_returns``, ``pacf_returns``,
        ``volatility_clustering``, ``leverage_effect``, ``vol_of_vol``,
        ``cross_correlation``
      - **distribution**: ``linear_baseline_crps``, ``non_elliptical``,
        ``tail_dependence``
      - **extreme**: extreme-event preservation
      - **calibration**: per-observation PIT / CRPS coverage (when OOS
        is long enough)

    Each metric entry contains numeric value(s), ``passed`` (bool),
    ``quality`` (``excellent``/``good``/``acceptable``/``poor``), and a
    human-readable ``interpretation``. The weighted-aggregate quality
    is what drives ``overall``.

    1.0.10 honesty pass: the SDK post-processes the per-metric
    ``passed`` field against tighter, value-driven thresholds (see
    :data:`HONEST_METRIC_THRESHOLDS`) so a metric whose value is far
    from training reports ``passed=False`` even when the platform's
    lenient quality bucket said ``acceptable``. The platform-supplied
    ``quality`` string is preserved untouched — only ``passed`` and
    the populated :attr:`caveats` list reflect the tightened view."""

    memorization_risk: str | None = None
    """``'low'`` / ``'medium'`` / ``'high'`` from the NN-distance ratio.
    ``'high'`` means the model is reproducing training samples too
    closely — synthetic data may leak real data."""

    memorization_nn_distance_ratio: float | None = None
    """Synthetic-to-training NN distance over training-to-training NN
    distance. ``> 0.80`` = low risk; ``0.50–0.80`` = medium;
    ``< 0.50`` = high (thresholds calibrated for financial-returns
    flow models, not image diffusion — a customer with
    ``coverage_95 ≈ 0.951`` was being flagged ``'high'`` at ratio
    0.84, mathematically incompatible with literal sample
    regurgitation)."""

    n_paths_used: int | None = None
    """How many synthetic paths the report was computed against."""

    holdout: bool = False
    """True when the validation ran against a customer-supplied
    held-out window (true OOS); False when ran against the training
    data (sanity check)."""

    caveats: list[str] = dataclasses.field(default_factory=list)
    """Human-readable lines, one per metric whose honest ``passed``
    check failed. Empty when every scored metric cleared its tightened
    threshold. Populated by :meth:`__post_init__` after the honesty
    pass against :data:`HONEST_METRIC_THRESHOLDS` — customers reading
    the report can see at a glance which structural facts the model
    did not nail, without re-walking :attr:`metrics`.

    The list does NOT change :attr:`overall` or :attr:`acceptable`;
    those still reflect the platform's weighted-quality verdict. The
    caveats surface what failed *honestly*, regardless of how the
    weighted aggregate landed."""

    def __post_init__(self) -> None:
        """Apply the 1.0.10 honesty pass to :attr:`metrics` and
        populate :attr:`caveats`.

        For every metric in :data:`HONEST_METRIC_THRESHOLDS`:
          - Read its numeric ``value`` from the metric dict.
          - If the value is missing / NaN / not finite, leave the
            metric untouched (we cannot judge honestly).
          - Otherwise, recompute ``passed = value < honest_threshold``
            and overwrite the metric dict's ``passed`` field.
          - When the recompute flips a metric to ``passed=False``,
            append a human-readable caveat line.

        Mutates :attr:`metrics` in place (the dataclass is frozen but
        :attr:`metrics` is a regular dict) and uses
        ``object.__setattr__`` to assign :attr:`caveats` past the
        frozen guard. Metrics not in the honest-threshold registry
        (calibration, dependence, etc.) pass through unchanged.
        """
        _apply_honesty_pass(self)

    @property
    def verdict(self) -> str:
        """Alias for :attr:`overall` — the canonical name across every
        report dataclass (see the module-level 'Report-uniformity
        protocol'). Returns ``'pass'`` / ``'warn'`` / ``'fail'``.
        """
        return self.overall

    @property
    def acceptable(self) -> bool:
        """True when the model is structurally sound enough to act on
        — i.e. ``overall != 'fail'`` AND ``memorization_risk != 'high'``.

        Mirrors the SDK.md guidance:
        ``assert report.overall != 'fail' and report.memorization_risk != 'high'``.
        A ``'warn'`` verdict still counts as acceptable (the customer
        should watch it, but the synthetic distribution hasn't drifted
        past the safe threshold). A ``'high'`` memorization risk
        downgrades any ``overall`` to not-acceptable: the synthetic
        data may leak training samples and the overfit verdicts built
        on top are unreliable.
        """
        if self.overall == "fail":
            return False
        return self.memorization_risk != "high"


# ============================================================================
# 1.0.10 honesty pass — per-metric tightened thresholds + caveat surfacing
# ============================================================================
#
# The persona-simulation surfaced cases where the platform's quality bucket
# said ``acceptable`` (and the per-metric ``passed`` rode along) even when
# the underlying metric value was substantially far from the training
# behaviour — e.g. a sign-flipped ACF on GARCH input. The metrics are right;
# the per-metric ``passed`` flag was the lenient/lying part of the report.
#
# This honesty pass does NOT touch the metric math, NOT touch the overall
# verdict, and NOT gate any downstream method. It only tightens the per-
# metric ``passed`` field so the report surface matches the metric value,
# and surfaces a human-readable :attr:`ValidationReport.caveats` line for
# any metric that now fails honestly.
#
# Thresholds are encoded as "passed iff metric_value < threshold". They are
# strictly tighter than the platform's ``acceptable`` bucket so a metric
# that the platform flagged ``poor`` always stays failed, and metrics in
# the wide ``acceptable`` zone get re-judged honestly.
#
# Metric naming follows the actual registry in
# `the server-side validation module` and the threshold
# tables in ``validation/metrics/core/thresholds.py``:
#
#   - ``tail_heaviness``   — kurtosis-error metric. Excellent/good/acceptable
#     thresholds in the platform are 1.0 / 2.0 / 4.0. The persona task asks
#     for kurtosis within 30% of real; with typical financial excess
#     kurtosis ~5 that is ~1.5 — strictly tighter than the platform's
#     'good' bucket.
#   - ``acf_returns``      — mean ACF error across lags. Persona task asks
#     for magnitude within 0.15 absolute (sign-flip handling falls out:
#     a sign-flipped ACF produces a large absolute error).
#   - ``marginal_ks``      — KS statistic measuring marginal (mean / std /
#     shape) drift. Persona task asks for mean / std recovery within
#     ~10–20% of real std; KS < 0.10 is the established proxy. This is the
#     closest analogue we ship to "mean_recovery / std_recovery".
#   - ``volatility_clustering`` — ACF of squared returns. Same reasoning as
#     acf_returns; tightened to 0.10 (the platform's 'good' bound).
#
# Every threshold here is strictly LESS than (i.e. tighter than) the
# platform's ``acceptable`` bound from
# ``DISTRIBUTION_THRESHOLDS`` / ``TEMPORAL_THRESHOLDS`` so the honesty
# pass cannot make a poor metric look acceptable.
HONEST_METRIC_THRESHOLDS: dict[str, float] = {
    "tail_heaviness":        1.5,   # vs platform 'acceptable'=4.0
    "acf_returns":           0.15,  # vs platform 'acceptable'=0.20
    "volatility_clustering": 0.10,  # vs platform 'acceptable'=0.20
    "marginal_ks":           0.10,  # vs platform 'acceptable'=0.20
}


def _apply_honesty_pass(report: ValidationReport) -> None:
    """Mutate ``report.metrics`` in place: recompute per-metric ``passed``
    against :data:`HONEST_METRIC_THRESHOLDS`, then collect caveats for
    every metric that flips to ``passed=False``. Assigns the resulting
    list to ``report.caveats`` via ``object.__setattr__`` (the dataclass
    is frozen).

    Skips metrics whose value is missing / non-numeric / non-finite —
    we cannot honestly judge what we cannot read.
    """
    caveats: list[str] = []
    metrics = report.metrics if isinstance(report.metrics, dict) else {}
    for metric_name, threshold in HONEST_METRIC_THRESHOLDS.items():
        entry = metrics.get(metric_name)
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        try:
            value_f = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value_f):
            continue
        honest_passed = value_f < threshold
        entry["passed"] = bool(honest_passed)
        if not honest_passed:
            interp = entry.get("interpretation", "")
            caveats.append(
                f"{metric_name}: value={value_f:.4f} exceeds honest "
                f"threshold {threshold:g} (passed=False). {interp}".rstrip()
            )
    # ``caveats`` is a declared field on the frozen dataclass; assign past
    # the frozen guard. The default factory gave us a fresh list per
    # construction, so this only overwrites the empty default.
    object.__setattr__(report, "caveats", caveats)


@dataclass(frozen=True)
class GenerationResult:
    """One batch of synthetic alternative-history paths plus structural
    validation metrics + memorization risk so the customer gets the
    "is this trustworthy" signal in one call.

    Internal cyclical (seasonal) embeddings are stripped from
    ``feature_names``, ``paths_returns``, ``paths_prices``, and
    ``last_prices`` — user-facing arrays only contain the customer's
    own feature columns.
    """

    paths_returns: np.ndarray
    """Shape ``(n_paths, horizon, n_real_features)`` — z-scored transformed
    returns straight from the ODE sampler. Useful for downstream metric
    computation. Calendar dims stripped."""

    paths_prices: np.ndarray
    """Shape ``(n_paths, horizon, n_real_features)`` — converted to
    price/level space via per-feature ``transform_params`` (log_return
    cumulative, difference cumulative, yoy/mom direct, level_std direct).
    This is what the customer's backtest engine consumes."""

    feature_names: list[str]
    """Real feature names in column order (internal cyclical embeddings
    removed)."""

    last_prices: np.ndarray
    """Per-feature anchor used to convert returns → prices. Useful for
    debugging / overlay plots."""

    horizon: int
    n_paths: int
    seed: int | None
    sdk_version: str
    """The CLIENT-side sablier-flow version that received this result.
    From 1.0.8 onward the client overwrites whatever the worker
    serialized so customers always see the SDK they imported (see
    :attr:`FitResult.sdk_version` for the rationale)."""

    # ----- bundled diagnostics --------------------------------------------
    # Both default to None for backwards compatibility with older payloads;
    # everything produced by current TEE workers populates them.

    memorization_risk: str | None = None
    """``'low'`` / ``'medium'`` / ``'high'`` from the NN-distance
    ratio, computed server-side. Customer should not trust the overfit
    verdict if ``'high'`` — see notebook
    ``03_memorization_audit.ipynb``."""

    memorization_nn_distance_ratio: float | None = None
    """Synthetic-to-training NN distance over training-to-training NN
    distance (computed by the server). ``> 0.80`` = low risk;
    ``0.50–0.80`` = medium; ``< 0.50`` = high. Thresholds are
    calibrated for financial flow models, not image diffusion: a
    perfectly calibrated flow samples from the same noisy continuous
    distribution as real returns, so synth landing within the training
    manifold (ratio ``< 1``) is normal — only ratio ``< 0.50`` is a
    genuine signal of literal sample regurgitation (consistent with
    the customer report where ratio ``= 0.84`` coexisted with
    ``coverage_95 = 0.951``)."""

    paths_index: Any = None
    """Optional index for :meth:`as_dataframes` to use as a default.
    Set by :meth:`Client.generate` when called with ``like=df`` so the
    resulting synthetic DataFrames overlay onto the customer's backtest
    window without an extra kwarg. ``None`` otherwise — the method
    falls back to a RangeIndex."""

    def as_dataframes(self, index: Any = None) -> list[Any]:
        """Convert the synthetic paths to a list of ``pd.DataFrame``\\s,
        one per path, each with columns equal to :attr:`feature_names`.

        ``index`` falls back to :attr:`paths_index` (set by ``like=`` on
        :meth:`Client.generate`) when not supplied, then to a
        RangeIndex if neither is set. Promote-from-adapter — equivalent
        to ``sablier_flow.adapters.as_dataframes(self, index=index)``.

        Canonical recipe — feed your backtest the SAME window on both
        the real and the synthetic side::

            import sablier_flow as sf
            backtest_window = df.iloc[-21:]                 # ← the slice you'll evaluate
            fit  = sf.fit(df, features=df.columns.tolist(),
                          data_types={c: 'price' for c in df.columns},
                          horizon=21)
            gen  = sf.generate(fit.model_id, n_paths=100, like=backtest_window)
            synth_results = [my_backtest(d) for d in gen.as_dataframes()]
            verdict = sf.robustness(
                my_backtest(backtest_window),               # ← real Sharpe on the SAME 21-bar window
                synth_results,                              # ← synth Sharpes on 21-bar windows
                primary_metric='sharpe',
            )

        Each ``d`` is a ``pd.DataFrame`` with the same columns / index
        shape as the slice you passed to ``like=``; your existing
        backtest function runs on it unchanged.

        .. warning::
           Pass the SAME window to both sides — ``my_backtest(df)`` over
           the full 3500-bar series compared against ``my_backtest(d)``
           over 21-bar synth windows is asymmetric and mechanically
           produces ``'highly_overfit'`` because the real Sharpe is
           computed on 167× more data than the synth Sharpes.
        """
        from sablier_flow.adapters.dataframe import as_dataframes
        effective = index if index is not None else self.paths_index
        return as_dataframes(self, index=effective)

    def with_paths_index(self, index: Any) -> GenerationResult:
        """Return a copy of this :class:`GenerationResult` with
        :attr:`paths_index` set to ``index``.

        Use this after :meth:`Client.fetch_result` on a
        :meth:`Client.generate_async` handle when you want the synth
        paths to overlay your backtest window — the async path can't
        infer the index itself because the original ``like=`` DataFrame
        isn't carried in the :class:`JobHandle`. The sync
        :meth:`Client.generate` path uses the same method internally
        so wire-shape canonicalisation lives in one place. Equivalent
        to ``dataclasses.replace(result, paths_index=index)``; the
        underlying dataclass is frozen, so this returns a fresh copy
        rather than mutating in place.
        """
        return dataclasses.replace(self, paths_index=index)
