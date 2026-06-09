"""Thin HTTP client for the remote sablier-flow TEE.

The customer's typical entry point::

    import sablier_flow
    client = sablier_flow.Client(api_key='sk_live_...')
    fit    = client.fit(real_data, features=[...], horizon=504)
    report = client.validate(fit.model_id)           # OOS structural + memorization
    paths  = client.generate(fit.model_id, n_paths=1000, like=backtest_window)
    synth_dfs = paths.as_dataframes()                # ready for any backtest engine

The remote path is the v1 product — customer data is encrypted with a
one-shot symmetric key under an attested X25519 envelope, ships to a
TEE running an SDK-pinned image, and the SDK never sees plaintext on
either side. The hosted Sablier service is the only supported path —
this thin client does not ship local-pipeline functions.

The full job lifecycle wired by this Client:

  1. Serialize the DataFrame to Parquet bytes.
  2. POST /v1/jobs with the job kind + params, receive (job_id, attestation_quote).
  3. Verify the quote against the SDK-pinned image digest, extract
     the TEE's ephemeral X25519 pubkey.
  4. Generate a one-shot AES-256-GCM result_key.
  5. Envelope-encrypt {result_key, params_json, data_parquet} to the
     TEE's ephemeral pubkey, PUT /v1/jobs/{id}/data.
  6. Poll GET /v1/jobs/{id} until status == "completed".
  7. GET /v1/jobs/{id}/result, AES-decrypt with the result_key,
     deserialize to a :class:`GenerationResult`.

The transport, attestation, and crypto primitives are pluggable —
:class:`HttpxTransport` is the production default; tests inject
:class:`InMemoryTransport`. :class:`AttestationVerifier` can run in
``mode="fake-for-dev"`` for staging.
"""

from __future__ import annotations

import base64
import io
import os
import time
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from sablier_flow.client.attestation import (
    AttestationQuote,
    AttestationVerifier,
)
from sablier_flow.client.cache import DiskCache, make_cache_key
from sablier_flow.client.crypto import envelope_encrypt
from sablier_flow.client.payload import (
    JobResultPayload,
    JobUploadPayload,
    decrypt_result,
)
from sablier_flow.client.transport import (
    CreateJobRequest,
    HttpxTransport,
    JobSummary,
    RemoteJobError,
    Transport,
)
from sablier_flow.types import (
    FitResult,
    GenerationResult,
    JobHandle,
    Model,
    ValidationReport,
)

if TYPE_CHECKING:
    from sablier_flow.types import CreditsBalance, UsageEvent, UsageSummary

__all__ = [
    "ALLOWED_DATA_TYPES",
    "Client",
    "cancel_job",
    "credits",
    "estimate_cost",
    "fetch_result",
    "fit",
    "fit_async",
    "generate",
    "generate_async",
    "list_jobs",
    "ping",
    "resume_job",
    "usage",
    "usage_summary",
    "validate",
    "validate_async",
    "validate_data",
    "whoami",
]


_TERMINAL_STATUSES = frozenset({"completed", "failed"})

# Sentinel for ``attestation_mode=`` on Client + module-level shortcuts.
# Treated as ``"production"`` downstream, but identity-compared upstream so
# the SDK can tell "caller accepted the default" apart from "caller wrote
# attestation_mode='production' themselves". The distinction matters
# because :class:`AttestationVerifier` emits a one-shot warning when a
# caller *explicitly* opts into production-without-registry — letting a
# wrapper-level literal default trip that warning would mean every
# default-constructed Client emits noise on import. See
# tests/unit/test_attestation_1_0_11.py for the contract.
_ATTESTATION_MODE_DEFAULT: Any = object()

# Canonical data-layer contract. The five column types are the only ones the
# backend transform code knows how to z-score forward and invert back to the
# customer's space without silently producing a wrong synth output.
ALLOWED_DATA_TYPES = frozenset({"price", "level", "return"})
# 1.1.0 — collapsed from the 5-string {price, return, rate, index, volatility}
# vocabulary that conflated semantic kind with a frequency-aware override path.
# The new vocabulary is transform-honest:
#   'price'  → log-return + z-score      (compounding multiplicative series)
#   'level'  → difference + z-score      (additive series; rates, vols, spreads, indices)
#   'return' → identity + z-score        (already-stationary series)
# Stair-step / forward-filled lower-cadence features (CPI, GDP, Fed rate) are
# not supported — aggregate to row cadence first.

# Wire-mapping shim: the SDK exposes the 3-string customer vocabulary
# `{price, level, return}` but the wire protocol still carries the
# pre-1.1 form for back-compat. `'level'` → `'rate'` translates to the
# same difference transform server-side. Collapses to identity once
# the server accepts the new vocab natively.
_WIRE_DATA_TYPE_MAPPING = {
    "price":  "price",
    "level":  "rate",
    "return": "return",
}

# Post-ffill NaN fraction the SDK tolerates before the data is unusable. Above
# this the model would mask out nearly every step for that column, so we surface
# the problem locally rather than wasting a training round-trip.
_MAX_NAN_FRACTION = 0.7


class Client:
    """Remote client for sablier-flow's confidential-compute TEE.

    Parameters
    ----------
    api_key
        Issued by Sablier; identifies the customer's organization and
        gates rate limits + credit accounting.
    endpoint
        Base URL of the TEE service. Defaults to the production
        endpoint when set; override for staging or self-hosted.
    pinned_image_digest
        ``sha256:<hex>`` digest the SDK release pins. Defaults to the
        digest baked into the wheel at build time; override only for
        staging / advanced workflows.
    attestation_mode
        ``"production"`` (default) or ``"fake-for-dev"``. The latter
        skips signature math but still enforces image digest, TEE type,
        hardware, measurements, and freshness — useful for staging
        roundtrips without real TEE-signed quotes.
    transport
        Override the HTTP layer; defaults to :class:`HttpxTransport`
        against the configured ``endpoint``. Tests pass in
        :class:`InMemoryTransport` for end-to-end coverage with no
        live server.
    timeout_s
        Per-HTTP-call timeout (default 60s).
    poll_interval_s
        How often :meth:`generate` polls the job status
        while waiting for the TEE to finish (default 2s).
    poll_timeout_s
        Total budget for a single job (default 30 min). Raises
        :class:`TimeoutError` if exceeded.
    cache_dir
        If set, enables an on-disk cache of generation results keyed by
        ``sha256(input_parquet + json(params))``. Identical follow-up
        requests are served from disk without re-contacting the TEE.
        Pass ``True`` to use the default ``~/.cache/sablier_flow``; a
        ``str``/``Path`` overrides the location; ``None`` (default)
        disables caching.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str | None = None,
        pinned_image_digest: str | None = None,
        attestation_mode: Any = _ATTESTATION_MODE_DEFAULT,
        transport: Transport | None = None,
        timeout_s: float = 60.0,
        poll_interval_s: float = 2.0,
        poll_timeout_s: float = 30 * 60,
        verify: bool | str | None = None,
        cache_dir: str | os.PathLike[str] | bool | None = None,
        profile: str = "default",
    ) -> None:
        """``verify``: TLS verification mode passed through to httpx.

        - ``None`` / ``True`` (default): verify against the system CA bundle
        - ``False``: skip verification (DEMO/DEV ONLY)
        - ``str`` path to a PEM: pin a specific cert / CA bundle —
          used during staging deploys where the TEE serves a
          self-signed cert.

        ``api_key`` is optional: if ``None`` we look at the
        ``SABLIER_FLOW_API_KEY`` env var first, then the credentials
        file (``~/.sablier/credentials``) written by :func:`login`.
        Raises :class:`ValueError` with a pointer to ``sf.login()`` if
        nothing resolves.
        """
        if not api_key:
            api_key = os.environ.get("SABLIER_FLOW_API_KEY")
        creds: dict[str, Any] | None = None
        stored_endpoint_from_creds: str | None = None
        if not api_key:
            # Last resort — read credentials written by sf.login().
            from sablier_flow.client.login import load_credentials
            creds = load_credentials(profile=profile)
            if creds and creds.get("api_key"):
                api_key = str(creds["api_key"])
                stored = creds.get("endpoint")
                if stored:
                    stored_endpoint_from_creds = str(stored)
        if not api_key:
            raise ValueError(
                "api_key is required. Run `sablier_flow.login()` to authenticate "
                "interactively, set SABLIER_FLOW_API_KEY in the env, or pass "
                "api_key='sk_live_...' explicitly. Get a key at "
                "https://sablier.ai → Settings → API Keys."
            )
        if pinned_image_digest is None:
            pinned_image_digest = _default_pinned_digest()
        # Resolution order for endpoint (shared with _build_client via
        # :func:`_resolve_endpoint`):
        #   1) explicit ``endpoint=`` kwarg
        #   2) SABLIER_FLOW_ENDPOINT env var
        #   3) endpoint stored in ~/.sablier/credentials (only if no env)
        #   4) hardcoded default 'https://flow.sablier.ai/v1'
        self.api_key = api_key
        self.endpoint = _resolve_endpoint(
            explicit=endpoint,
            stored=stored_endpoint_from_creds,
        )
        self.pinned_image_digest = pinned_image_digest
        # Resolve attestation_mode for the public attribute + verifier
        # forward. The sentinel signals "caller accepted the default" —
        # downstream the verifier still treats it as production, but we
        # also tell it ``mode_explicit=False`` so the one-shot
        # production-without-registry warning stays silent for callers
        # who never opted in. Any other value (string) is treated as
        # explicit user opt-in and forwarded with ``mode_explicit=True``.
        mode_is_default = attestation_mode is _ATTESTATION_MODE_DEFAULT
        resolved_mode: str = (
            "production" if mode_is_default else str(attestation_mode)
        )
        self.attestation_mode = resolved_mode
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s
        self._transport: Transport = transport or HttpxTransport(
            api_key=api_key,
            endpoint=self.endpoint,
            timeout_s=timeout_s,
            verify=verify,
        )
        self._verifier = AttestationVerifier(
            expected_image_digest=pinned_image_digest,
            mode=resolved_mode,  # type: ignore[arg-type]
            mode_explicit=not mode_is_default,
        )
        self._cache: DiskCache | None
        if cache_dir is None or cache_dir is False:
            self._cache = None
        elif cache_dir is True:
            self._cache = DiskCache()
        else:
            # DiskCache accepts Path | str | None; we've already narrowed
            # cache_dir away from None and bool above. PathLike is a
            # superclass of Path so cast through str() for mypy clarity.
            self._cache = DiskCache(os.fspath(cache_dir))

    # ------------------------------------------------------------------
    # Defining calls: fit / generate / validate
    # ------------------------------------------------------------------

    def fit(
        self,
        real_data: pd.DataFrame,
        *,
        data_types: dict[str, str] | None = None,
        features: Sequence[str] | None = None,
        horizon: int | None = None,
        train_split: float | None = 0.8,
        embargo_days: int = 21,
        seed: int | None = None,
        idempotency_key: str | None = None,
        quiet: bool = False,
    ) -> FitResult:
        """Train a flow model on the customer's history.

        Returns a :class:`FitResult` whose ``model_id`` is the handle for
        subsequent :meth:`generate` and :meth:`validate` calls.

        ``real_data`` must be a :class:`pandas.DataFrame` with a
        :class:`~pandas.DatetimeIndex` (monotonic, ideally no duplicates,
        uniform cadence) and all-numeric columns — typically prices or
        returns indexed by bar timestamp. **Any uniform cadence is
        accepted** in 1.1.0 (daily, intraday 5-min / 1-min, weekly,
        monthly, etc.); the SDK auto-detects the row cadence from the
        index. NaNs are tolerated and passed through to the model (which
        masks them). The SDK rejects only columns whose post-ffill NaN
        fraction exceeds 70%. Need ≥ 200 rows; more is better.

        ``data_types`` (REQUIRED) maps each feature column to one of
        :data:`ALLOWED_DATA_TYPES`:

          - ``'price'``  — compounding multiplicative series (asset prices,
            FX, ratios). Transformed via log-return + z-score. Must be
            strictly positive.
          - ``'level'``  — additive series with meaningful levels (rates,
            volatility indices, spreads, dollar index). Transformed via
            difference + z-score. Can cross zero / be negative.
          - ``'return'`` — already-stationary series (factor returns,
            pre-differenced data). Transformed via identity + z-score.

        Missing or unsupported values raise :class:`TypeError` /
        :class:`ValueError` before the network round-trip. Stair-step
        features (forward-filled lower-cadence data such as monthly CPI in
        a daily DataFrame) are NOT supported in 1.1.0 — aggregate to the
        row cadence before fitting.

        ``features`` is the list of columns the model trains on. All listed
        columns are co-generated jointly — every feature is sampled at
        every horizon step and every column is available to constraints,
        backtests, and downstream analytics. Defaults to every numeric
        column of ``real_data`` when omitted.

        ``train_split`` (default ``0.8``) controls the train/test split:
        the server holds out the last ``1 - train_split`` fraction of
        ``real_data`` as an OOS slice (with ``embargo_days`` of padding
        between train end and OOS start to defend against rolling-stat
        leakage). The OOS slice is persisted encrypted alongside the
        checkpoint; subsequent :meth:`validate` calls without an explicit
        ``holdout_data`` use it automatically.

        Pass ``train_split=None`` to skip the split and train on the full
        DataFrame — useful when you've already done a split externally,
        or when running an offline calibration where OOS is irrelevant.

        ``horizon`` is the number of **time steps (bars)** the model is
        trained against — purely a count, independent of cadence: 504 means
        504 bars whether those are daily (~2y), hourly, or 1-min bars.
        Default: 504 steps, or half the available history if shorter. The
        maximum is data-adaptive: it scales with how many rows you pass,
        allowing a longer window only when the history leaves enough
        non-overlapping spans to train it robustly (floor 504 steps, capped
        at 5000). The generator is horizon-agnostic, so
        ``generate(model_id, horizon=M)`` works for any ``M`` without
        retraining; quality is best near the trained value and degrades
        modestly as you stretch further past it.
        """
        if horizon is not None:
            _check_fit_horizon_bounds(
                horizon, n_rows=len(real_data), train_split=train_split,
            )
        if seed is None:
            import secrets
            seed = secrets.randbelow(2**31 - 1)
        idempotency_key = _ensure_idempotency_key(idempotency_key)
        # Fail fast on bad input before the network round-trip — agents
        # and humans both benefit from a sub-second local error over a
        # 1-2 min queue + start + cryptic Cloud Run failure.
        params = self._prep_fit_params(
            real_data,
            data_types=data_types,
            features=features,
            horizon=horizon,
            train_split=train_split,
            embargo_days=embargo_days,
            seed=seed,
        )
        # 1.0.10 — cost-accounting visibility before the GPU round-trip.
        # 5 of 6 persona simulations asked "how much will this cost?";
        # printing the pre-flight estimate to stderr at zero extra
        # latency makes the SDK honest about spend without forcing
        # callers to thread sf.estimate_cost(...) through every script.
        self._print_estimate_cost_line(
            kind="fit",
            quiet=quiet,
            n_features=len(params.get("target_features") or []) or None,
            n_rows=len(real_data),
            horizon=params.get("horizon"),
        )
        result_bytes = self._run_job(
            kind="fit",
            real_data=real_data,
            params=params,
            idempotency_key=idempotency_key,
        )
        self._print_actual_cost_line(kind="fit", quiet=quiet)
        return _stamp_sdk_version(
            JobResultPayload.from_bytes(result_bytes).to_fit_result()
        )

    def generate(
        self,
        model_id: str,
        *,
        data_types: dict[str, str] | None = None,
        n_paths: int = 1000,
        horizon: int | None = None,
        anchor_data: pd.DataFrame | None = None,
        like: pd.DataFrame | None = None,
        seed: int | None = None,
        idempotency_key: str | None = None,
        quiet: bool = False,
    ) -> GenerationResult:
        """Generate ``n_paths`` synthetic paths from a fitted model.

        Three ways to control the shape of the synthetic paths, in
        increasing order of specificity:

        - **Default** (no kwargs) — paths use the model's trained horizon
          and start from the server-stored training-tail anchor.
        - **Explicit horizon + anchor_data** — pass them separately when
          you want full control.
        - **`like=df`** — the most natural for backtest-augmentation. Pass
          your backtest window and the call derives three things from it:

            1. ``horizon = len(df)`` — synth paths have exactly the same
               number of bars as your window.
            2. ``paths_index = df.index`` — once you call ``.as_dataframes()``
               on the result, each synth path is indexed by the same dates
               as your real window so they overlay directly in plots and
               feed into your backtest function without re-indexing.
            3. **Anchor** — the context window that conditions the
               generation. The server uses the last ``obs_length`` (≈200)
               bars of training data ending immediately before
               ``df.index[0]``, so every synth path starts from the real
               price level at the bar before your backtest window — they
               are continuations of the same history that produced the
               real window, not random starting points.

          ``like=`` overrides both ``horizon`` and ``anchor_data`` when set.

        ``data_types`` (REQUIRED) maps each generated column to its
        canonical type. Must cover every column the customer expects to see
        in the synth output. Allowed values: ``'price'``, ``'return'``,
        ``'rate'``, ``'index'``, ``'volatility'`` (see
        :data:`ALLOWED_DATA_TYPES`). When ``like`` / ``anchor_data`` is
        supplied, ``data_types`` is also checked against those columns.

        Quality trade-off: ``horizon`` (or ``len(like)``) doesn't have to
        match the training horizon — the generator is horizon-agnostic
        — but quality degrades modestly as you stretch further past it.
        """
        if not model_id:
            raise ValueError("model_id is required")
        _check_generate_n_paths_bounds(n_paths)
        if horizon is not None:
            _check_generate_horizon_bounds(horizon)
        if seed is None:
            import secrets
            seed = secrets.randbelow(2**31 - 1)
        idempotency_key = _ensure_idempotency_key(idempotency_key)
        # Validate the optional DataFrame args. Anchor / like are
        # typically short windows (don't enforce 200-row floor).
        if anchor_data is not None:
            _validate_real_data(
                anchor_data, arg_name="anchor_data", require_min_rows=False,
            )
            self._check_columns_cover_model(
                model_id, anchor_data, arg_name="anchor_data",
            )
        if like is not None:
            _validate_real_data(
                like, arg_name="like", require_min_rows=False,
            )
            self._check_columns_cover_model(model_id, like, arg_name="like")

        # data_types ships on the wire as ``feature_data_types`` so
        # training_service / generation_service can pick the correct
        # inverse-transform branch.
        # Only require ``data_types=`` when the customer is
        # supplying fresh data (``like=`` or ``anchor_data=``). Without
        # them the server reuses the model's registered types from fit
        # time and the SDK must not put feature_data_types on the wire
        # (the server-side resolver would otherwise overwrite the
        # checkpoint's registered map with a partial / wrong dict).
        # 1.0.8: relax further — when ``like=`` / ``anchor_data=`` is
        # supplied but ``data_types=`` is NOT, we ALSO skip the require
        # and let the server reuse the model's registered types. The
        # anchoring DataFrame carries the SAME columns the model was
        # fitted on (that invariant is enforced by
        # _check_columns_cover_model above); forcing the customer to
        # re-declare types they already registered at fit time was pure
        # friction. If the customer DOES pass data_types explicitly we
        # still validate + forward it (so overrides keep working).
        ref_df = like if like is not None else anchor_data
        normalized_data_types: dict[str, str] | None
        if ref_df is not None and data_types is not None:
            expected_cols = [
                str(c) for c in ref_df.columns
                if pd.api.types.is_numeric_dtype(ref_df[c])
            ]
            normalized_data_types = _require_data_types(
                data_types,
                expected_columns=expected_cols,
                arg_name="data_types",
            )
            # NaN guard on the supplied DataFrames — same rule as fit.
            _check_nan_fraction(
                ref_df,
                features=expected_cols,
                arg_name="like" if like is not None else "anchor_data",
            )
        elif ref_df is not None:
            # data_types omitted: still NaN-guard the supplied window so
            # garbage-in doesn't burn a paid round-trip. The server
            # reuses the model's registered types from fit time.
            expected_cols = [
                str(c) for c in ref_df.columns
                if pd.api.types.is_numeric_dtype(ref_df[c])
            ]
            _check_nan_fraction(
                ref_df,
                features=expected_cols,
                arg_name="like" if like is not None else "anchor_data",
            )
            normalized_data_types = None
        else:
            normalized_data_types = None
        # 1.0.8: mirror the data_types contract on the wire for
        # ``frequency``. Auto-detect was the source of a noisy
        # late-failure: a short ``like=df.iloc[-21:]`` window that spans
        # a holiday gap (Christmas, Thanksgiving, exchange closures) has
        # a p95 Δt up to ~3.5d which trips the "irregular index" guard
        # 1.1.0 — the `frequency=` kwarg is gone from the public surface;
        # the server reuses the model's registered training frequency on
        # every generate. Like/anchor windows just inherit it.

        # ``like=df`` is the convenient front-end: derive length + index +
        # anchor *price level* from a single DataFrame so the synthetic
        # paths overlay onto the customer's backtest window directly.
        # We extract three things on the client side:
        #
        #   horizon       = len(like)               # how many bars to sample
        #   like_index    = [str(ts) for ts in like.index]   # dates for
        #                   .as_dataframes() alignment
        #   anchor_prices = like.iloc[0].to_dict()  # synth starts here
        #
        # Sending ``anchor_prices`` is what actually makes the synth
        # begin at the real price level on ``like.index[0]``. Without
        # it the server falls back to the checkpoint's stored
        # ``last_prices``, which is anchored at training end (or worse,
        # at 85%-through-training when the pipeline.train_size attribute
        # was missing). That mismatch was the root of the visible bug
        # where synth paths started hundreds of dollars below the real
        # series even with ``like=`` set.
        like_index: list[str] | None = None
        anchor_prices: dict[str, float] | None = None
        if like is not None:
            horizon = len(like)
            try:
                like_index = [str(ts) for ts in like.index]
            except Exception:
                like_index = None
            try:
                first_row = like.iloc[0]
                anchor_prices = {
                    str(col): float(first_row[col])
                    for col in like.columns
                    if pd.api.types.is_numeric_dtype(like[col])
                }
            except Exception:
                anchor_prices = None
        elif anchor_data is not None:
            # Forward-forecast case: the customer's intent is "synth
            # continues from where my real data ends." Anchor at the
            # LAST row of anchor_data ("today"), not the first — and
            # crucially not the checkpoint's stored ``last_prices``
            # which is anchored at training end (or worse, at
            # 85%-through-training when train_size was missing on the
            # pipeline). Same scale-mismatch fix as the ``like=`` branch
            # above, just for the forward-generation use case.
            try:
                last_row = anchor_data.iloc[-1]
                anchor_prices = {
                    str(col): float(last_row[col])
                    for col in anchor_data.columns
                    if pd.api.types.is_numeric_dtype(anchor_data[col])
                }
            except Exception:
                anchor_prices = None

        params: dict[str, Any] = {
            "model_id": model_id,
            "n_paths": int(n_paths),
            "horizon": horizon,
            "seed": int(seed),
        }
        if normalized_data_types is not None:
            params["feature_data_types"] = _to_wire_data_types(normalized_data_types)
        if like_index is not None:
            params["like_index"] = like_index
        if anchor_prices is not None:
            params["anchor_prices"] = anchor_prices
        # 1.0.10 — cost-accounting visibility. Cheap call but still
        # debits credits; persona testing showed callers wanted the
        # spend confirmed without having to ``sf.estimate_cost`` first.
        self._print_estimate_cost_line(
            kind="generate",
            quiet=quiet,
            n_paths=int(n_paths),
            horizon=horizon,
        )
        result_bytes = self._run_job(
            kind="generate",
            real_data=anchor_data,
            params=params,
            idempotency_key=idempotency_key,
        )
        self._print_actual_cost_line(kind="generate", quiet=quiet)
        result = _stamp_sdk_version(
            JobResultPayload.from_bytes(result_bytes).to_generation_result()
        )
        # Stash the customer's requested index on the result so
        # .as_dataframes() can default-align to the backtest window
        # without an extra kwarg. Uses GenerationResult.paths_index
        # (a declared field; no monkey-patching).
        if like is not None:
            result = _with_paths_index(result, like.index)
        return result

    def validate(
        self,
        model_id: str,
        *,
        data_types: dict[str, str] | None = None,
        holdout_data: pd.DataFrame | None = None,
        n_paths: int = 500,
        seed: int | None = None,
        idempotency_key: str | None = None,
        quiet: bool = False,
    ) -> ValidationReport:
        """Run the structural-validation suite on a fitted model.

        Without ``holdout_data`` the server uses the OOS slice persisted
        at fit time (the 20% train-test split with embargo). Pass
        ``holdout_data`` to override with your own slice.

        Cheap: no GPU training, only inference + metric compute.

        ``data_types`` (REQUIRED only when ``holdout_data`` is supplied)
        maps each model column to its canonical type. Allowed values:
        ``'price'``, ``'level'``, ``'return'`` (see
        :data:`ALLOWED_DATA_TYPES`). When ``holdout_data`` is supplied
        the keys are also checked against its column set.

        ``n_paths`` defaults to 500 — empirically the smallest value
        that yields stable structural-metric estimates (KS / ES tail
        statistics jitter visibly below ~500 paths). Generation is
        cheap relative to fit, and 500 matches the family-DSR n_paths
        guidance surfaced in ``FamilyReport.notes``.
        """
        if not model_id:
            raise ValueError("model_id is required")
        _check_validate_n_paths_bounds(n_paths)
        if seed is None:
            import secrets
            seed = secrets.randbelow(2**31 - 1)
        idempotency_key = _ensure_idempotency_key(idempotency_key)
        if holdout_data is not None:
            _validate_real_data(
                holdout_data, arg_name="holdout_data", require_min_rows=False,
            )
        # Only require ``data_types=`` when the customer is supplying a
        # fresh slice (holdout_data). Without holdout_data the server
        # reuses the model's training OOS slice and the data_types it
        # already registered at fit time, so the SDK must not gate the
        # call on data_types= and must not put feature_data_types on the
        # wire (the server-side resolver would otherwise overwrite the
        # checkpoint's registered map with a partial / wrong dict).
        normalized_data_types: dict[str, str] | None
        if holdout_data is not None:
            expected_cols = [
                str(c) for c in holdout_data.columns
                if pd.api.types.is_numeric_dtype(holdout_data[c])
            ]
            normalized_data_types = _require_data_types(
                data_types,
                expected_columns=expected_cols,
                arg_name="data_types",
            )
            _check_nan_fraction(
                holdout_data,
                features=expected_cols,
                arg_name="holdout_data",
            )
        else:
            normalized_data_types = None
        # 1.0.8: mirror the data_types contract on the wire for
        # ``frequency`` (see Client.generate for the rationale). Short
        # holdout windows that span a holiday tripped the
        # "irregular index" detection guard even when the underlying
        # cadence was plain business-daily; the model already knows
        # its training frequency so re-detecting here was pure friction.
        params: dict[str, Any] = {
            "model_id": model_id,
            "n_paths": int(n_paths),
            "seed": int(seed),
            "holdout": holdout_data is not None,
        }
        if normalized_data_types is not None:
            params["feature_data_types"] = _to_wire_data_types(normalized_data_types)
        # 1.0.10 — cost-accounting visibility (see Client.fit /
        # Client.generate for the persona-testing rationale).
        self._print_estimate_cost_line(
            kind="validate",
            quiet=quiet,
            n_paths=int(n_paths),
        )
        result_bytes = self._run_job(
            kind="validate",
            real_data=holdout_data,
            params=params,
            idempotency_key=idempotency_key,
        )
        self._print_actual_cost_line(kind="validate", quiet=quiet)
        report = _stamp_sdk_version(
            JobResultPayload.from_bytes(result_bytes).to_validation_report()
        )
        # 1.0.10 — single-feature models can't meaningfully populate
        # dependence-family metrics (cross-asset correlation, joint
        # distribution, tail dependence) — there is no second feature
        # to be dependent ON. The server still emits those metric rows
        # because the registry is universal; the SDK suppresses them
        # locally so the customer doesn't see a ``fail`` verdict driven
        # by metrics that are not applicable to their model.
        return _suppress_dependence_metrics_when_single_feature(
            report, model_features=self._safe_model_features(model_id),
        )

    # ------------------------------------------------------------------
    # Model management — list / inspect / delete fitted models.
    # ------------------------------------------------------------------

    def list_models(self, *, limit: int = 50) -> list[Model]:
        """List fitted models on the customer's account, most-recently-used
        first.

        Returns a list of :class:`Model` dataclasses with each model's
        metadata (``model_id``, ``features``, ``training_end_date``,
        ``expires_at``, etc.). Use this to pick an existing model to
        reuse — passing its ``model_id`` to :meth:`generate` or
        :meth:`validate` avoids re-fitting and re-paying the training cost.

        Parameters
        ----------
        limit
            Max rows to return. The server caps at 500.

        Examples
        --------
        Pick the most recent model that contains SPY::

            models = client.list_models()
            spy_model = next(m for m in models if 'SPY' in m.features)
            paths = client.generate(spy_model.model_id, like=backtest_window)
        """
        resp = self._transport.list_models(limit=int(limit))
        return [_model_info_to_dataclass(m) for m in resp.models]

    def get_model(self, model_id: str) -> Model:
        """Fetch metadata for a single model. Raises ``ValueError`` if
        the model is not found OR belongs to a different user (same
        error in both cases — no enumeration probe)."""
        from sablier_flow.client.transport import JobNotFoundError

        if not model_id:
            raise ValueError("model_id is required")
        try:
            info = self._transport.get_model(model_id)
        except JobNotFoundError as exc:
            raise ValueError(f"model {model_id} not found") from exc
        return _model_info_to_dataclass(info)

    def delete_model(self, model_id: str) -> None:
        """Mark a model as expired and best-effort remove its encrypted
        blobs from object storage. Idempotent — deleting an already-gone
        model returns silently."""
        if not model_id:
            raise ValueError("model_id is required")
        self._transport.delete_model(model_id)

    # ------------------------------------------------------------------
    # Account / billing / pre-flight — surface what the dashboard shows
    # so customers don't have to leave the Python REPL.
    # ------------------------------------------------------------------

    def ping(self) -> dict[str, Any]:
        """Liveness check. Returns ``{'status': 'ok', 'server_version':
        ..., 'min_sdk_version': ...}``. Useful as a smoke test from CI
        / a notebook to confirm the SDK can reach the server and that
        the installed SDK version is still supported."""
        r = self._transport.health()
        return r.model_dump()

    def whoami(self) -> dict[str, Any]:
        """Return identity of the authenticated caller: user_id, email,
        name, subscription tier, and the prefix/name of the API key
        in use. Useful for "am I on the right account?" debugging."""
        return self._transport.whoami().model_dump()

    def credits(self) -> CreditsBalance:
        """Return the customer's current credit balance as a
        :class:`~sablier_flow.types.CreditsBalance`.

        Fields:

          - ``available``: credits the customer can spend right now
            (monthly_allocation − monthly_used + purchased).
          - ``monthly_allocation``: credits granted per billing period by
            the current subscription tier.
          - ``monthly_used``: credits already used in this billing period.
          - ``purchased``: one-time-purchased credits that carry over.
          - ``tier``: the user's tier string ('free' / 'pro' / etc.).
        """
        from sablier_flow.types import CreditsBalance

        wire = self._transport.credits()
        return CreditsBalance.model_validate(wire.model_dump())

    def usage(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[UsageEvent]:
        """Return the customer's flow-SDK usage history (most recent first)
        as a list of :class:`~sablier_flow.types.UsageEvent`.

        Always restricted to *this* customer's jobs — there is no
        cross-customer view. Scoped to flow operations (fit / generate
        / validate) only; other Sablier-platform charges (if any) are not
        included.

        Parameters
        ----------
        since, until
            ISO date strings ('2026-05-01' or '2026-05-01T00:00:00')
            bounding the creation window. Both inclusive on the lower
            side, exclusive on the upper.
        kind
            Filter to a single job type ('fit' / 'generate' / 'validate').
        limit
            Max rows returned (server caps at 500).
        """
        from sablier_flow.types import UsageEvent

        r = self._transport.usage(since=since, until=until, kind=kind, limit=limit)
        return [UsageEvent.model_validate(item.model_dump()) for item in r.items]

    def usage_summary(self, *, period: str = "month") -> UsageSummary:
        """Aggregate usage by job kind over a window. ``period`` is one
        of ``'month'`` (current calendar month, default), ``'week'``
        (last 7 days), ``'30d'`` (last 30 days), or ``'all'`` (lifetime).

        Returns a :class:`~sablier_flow.types.UsageSummary` with
        ``period_start``, ``period_end``, ``total_credits``, and
        ``by_kind`` (each kind maps to its job count + credits).
        """
        from sablier_flow.types import UsageSummary

        wire = self._transport.usage_summary(period=period)
        return UsageSummary.model_validate(wire.model_dump())

    def estimate_cost(
        self,
        kind: str,
        *,
        real_data: pd.DataFrame | None = None,
        features: Sequence[str] | None = None,
        horizon: int | None = None,
        n_paths: int | None = None,
        n_features: int | None = None,
        n_rows: int | None = None,
    ) -> dict[str, Any]:
        """Pre-flight credit estimate for an upcoming job. No GPU is
        dispatched; the server runs a deterministic formula
        (``n_features × n_rows × horizon × n_paths`` for the relevant
        kind) and returns ``{estimated_credits, low, high, notes}``.

        The estimate is for **credits**, not wall-clock time. Wall-clock
        depends on queue depth, GPU availability and per-job dataset
        shape, so the SDK does not surface a time prediction — poll
        :func:`list_jobs` or block with :func:`fetch_result` for the live
        signal instead.

        Parameters
        ----------
        kind
            ``'fit'`` / ``'generate'`` / ``'validate'``.
        real_data
            For ``kind='fit'``: the DataFrame you'd pass to
            :meth:`fit`. The server uses ``len(real_data)`` and the
            number of numeric columns. Optional — pass ``n_features``
            and ``n_rows`` explicitly if you don't have the DataFrame
            on hand.
        features
            Subset of columns to feature-count if smaller than the
            DataFrame's column set.
        horizon
            Training horizon you intend to use.
        n_paths
            Number of synthetic paths for ``kind='generate'``.
            (Currently a flat cost — n_paths doesn't change credits.)
        n_features
            Explicit feature count. Use this when you don't have
            ``real_data`` on hand (e.g. CI estimating spend before
            pulling a 1 GB parquet). Forwarded directly to the server
            and overridden by the count derived from ``real_data`` if
            both are provided.
        n_rows
            Explicit row count, same semantics as ``n_features`` — pass
            ``len(df)`` directly when you can't ship the DataFrame.
        """
        from sablier_flow.client.transport import EstimateCostRequest

        valid = {"fit", "generate", "validate"}
        if kind not in valid:
            raise ValueError(
                f"kind must be one of {sorted(valid)}; got {kind!r}"
            )

        derived_n_features: int | None = None
        derived_n_rows: int | None = None
        if real_data is not None:
            derived_n_rows = len(real_data)
            if features is not None:
                derived_n_features = len(list(features))
            else:
                derived_n_features = len([
                    c for c in real_data.columns
                    if pd.api.types.is_numeric_dtype(real_data[c])
                ])
        # DataFrame-derived counts take precedence (they're guaranteed
        # accurate); explicit kwargs fill in when no DataFrame was given.
        wire_n_features = (
            derived_n_features if derived_n_features is not None else n_features
        )
        wire_n_rows = derived_n_rows if derived_n_rows is not None else n_rows
        body = EstimateCostRequest(
            kind=kind,
            n_features=wire_n_features,
            n_rows=wire_n_rows,
            horizon=horizon,
            n_paths=n_paths,
        )
        response = self._transport.estimate_cost(body).model_dump()
        # 1.0.20 — strip the server-side `estimated_duration_s` heuristic
        # from the response dict. The credit estimate is deterministic
        # (n_features × n_rows × horizon formula); the duration estimate
        # is a heuristic that runs 4-5× too high in practice (observed:
        # 52 min predicted vs 11 min actual on a 7-feature 14-year fit),
        # and surfacing it to customers — or to AI agents introspecting
        # the response — anchored expectations on a misleading number.
        # Wire field kept on the dataclass for back-compat; just not
        # exposed in the user-facing dict any more.
        response.pop("estimated_duration_s", None)
        return response

    # ------------------------------------------------------------------
    # The full attest-encrypt-poll-decrypt lifecycle. Kept private so
    # the public methods (fit, generate, validate)
    # share a single happy path.
    # ------------------------------------------------------------------

    def _run_job(
        self,
        *,
        kind: str,
        real_data: pd.DataFrame | None,
        params: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> bytes:
        """Open + upload + poll + decrypt — the synchronous round-trip.

        Composes :meth:`_open_and_upload` and :meth:`_wait_and_decrypt`
        so the async surface (``fit_async`` etc.) and the sync one share
        every line of the wire/crypto path. Anything in this function
        beyond the two composed calls is the disk-cache shortcut: cache
        hits skip the entire TEE round-trip including attestation.
        """
        # Cache check — keyed by (kind, input bytes, params). Hits skip
        # the entire TEE round-trip; the security boundary is that cache
        # entries come from previously-attested round-trips on this same
        # SDK build, stored only on the customer's local disk.
        parquet_bytes = _df_to_parquet_bytes(real_data) if real_data is not None else b""
        cache_key: str | None = None
        if self._cache is not None:
            cache_key = make_cache_key(
                parquet_bytes,
                {"kind": kind, **params},
                endpoint=self.endpoint,
                api_key=self.api_key,
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        job_id, result_key = self._open_and_upload(
            kind=kind,
            parquet_bytes=parquet_bytes,
            params=params,
            idempotency_key=idempotency_key,
        )
        # Persist the result_key BEFORE entering the poll loop. Without
        # this, any mid-poll failure (network blip, Ctrl-C, OS reboot,
        # AuthenticationError on a key rotation) means we permanently
        # lose the one-shot AES key the TEE will encrypt the result with
        # — the job runs, the customer is billed, but the ciphertext is
        # undecryptable. :meth:`resume` reads this file back.
        _persist_pending_job(
            job_id=job_id,
            kind=kind,
            result_key=result_key,
            endpoint=self.endpoint,
            api_key=self.api_key,
        )
        try:
            result_bytes = self._wait_and_decrypt(job_id, kind, result_key)
        except BaseException:
            # Leave the pending file in place for :meth:`resume` to find.
            raise
        else:
            _clear_pending_job(job_id)

        if self._cache is not None and cache_key is not None:
            import contextlib as _ctx
            with _ctx.suppress(OSError):
                self._cache.put(cache_key, result_bytes)
        return result_bytes

    def _open_and_upload(
        self,
        *,
        kind: str,
        parquet_bytes: bytes,
        params: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> tuple[str, bytes]:
        """First half of the TEE round-trip: POST → verify → upload.

        Returns ``(job_id, result_key)``. The ``result_key`` is the
        one-shot AES-256-GCM key the TEE will encrypt the result with;
        the caller must hand it back to :meth:`_wait_and_decrypt` to
        retrieve the plaintext.
        """
        # Wire contract: ``feature_data_types`` and ``frequency`` ship
        # as top-level fields on CreateJobRequest (siblings of ``kind`` /
        # ``params``). We pop the keys out of ``params`` here so the
        # wire shape is unambiguous (top-level only — no double-send).
        wire_params = dict(params)
        feature_data_types = wire_params.pop("feature_data_types", None)
        create_resp = self._transport.create_job(
            CreateJobRequest(  # type: ignore[arg-type]
                kind=kind,
                params=wire_params,
                feature_data_types=feature_data_types,
            ),
            idempotency_key=idempotency_key,
        )
        job_id = create_resp.job_id

        # Verify quote against the SDK-pinned image digest. Raises
        # AttestationVerificationError on any mismatch — no plaintext
        # ever leaves the customer's machine on a failed handshake.
        quote = AttestationQuote.from_wire(create_resp.quote_bytes)
        ephemeral_pub = self._verifier.verify(quote)

        upload = JobUploadPayload.build(data_parquet=parquet_bytes, params=params)
        envelope = envelope_encrypt(upload.to_bytes(), ephemeral_pub)
        self._transport.upload_data(job_id, envelope.to_bytes())
        return job_id, upload.result_key

    def _wait_and_decrypt(self, job_id: str, kind: str, result_key: bytes) -> bytes:
        """Second half of the TEE round-trip: poll → download → decrypt.

        Renders a progress bar for ``kind=='fit'`` (the long one) and
        a quiet poll for generate/validate. On a job failure, raises
        :class:`RemoteJobError`; on a poll-timeout-budget overrun, raises
        :class:`TimeoutError` with the last-seen status.
        """
        deadline = time.monotonic() + self.poll_timeout_s
        bar = _ProgressRenderer.create(kind=kind)
        try:
            while True:
                status = self._transport.get_status(job_id)
                bar.update(status.progress, status.status)
                if status.status in _TERMINAL_STATUSES:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"job {job_id} did not finish within {self.poll_timeout_s}s "
                        f"(last status: {status.status!r})"
                    )
                time.sleep(self.poll_interval_s)
        finally:
            bar.close()

        if status.status == "failed":
            raise RemoteJobError(job_id, status.error_message or "unknown failure")

        result_resp = self._transport.get_result(job_id)
        return decrypt_result(result_resp.ciphertext_bytes, result_key)

    def _check_columns_cover_model(
        self,
        model_id: str,
        df: pd.DataFrame,
        *,
        arg_name: str,
    ) -> None:
        """Symmetric to fit's strict-features check, in reverse: when the
        caller hands :meth:`generate` a ``like=`` or ``anchor_data=`` frame,
        require the DataFrame's columns to be a superset of the model's
        trained feature set.

        Silent partial-feature acceptance is the same footgun as silent
        subset fits: the server falls back to its checkpoint anchors for
        any missing column, the customer's prices vector silently differs
        from what they think they're sending, and the synth paths drift
        away from the real window with no error surfaced. Catching it
        client-side saves a paid round-trip on top of the wrong result.
        """
        try:
            model = self.get_model(model_id)
        except Exception:
            # If the model lookup itself fails (network blip, transient
            # 5xx) don't block the customer — the server will surface a
            # clearer error on the create_job call moments later. The
            # strict check is best-effort UX, not a hard contract.
            return
        model_features = list(model.features or [])
        if not model_features:
            return
        df_cols = set(map(str, df.columns))
        missing = [c for c in model_features if c not in df_cols]
        if missing:
            raise ValueError(
                f"{arg_name} is missing columns the model was trained on: "
                f"{missing}. Model {model_id!r} expects features "
                f"{model_features}; {arg_name} has columns "
                f"{sorted(df_cols)}. Add the missing columns to "
                f"{arg_name} (or pick a model whose feature set matches) "
                "— partial-feature generates silently drift from the real "
                "window because the server falls back to checkpoint "
                "anchors for any missing column."
            )

    def _prep_fit_params(
        self,
        real_data: pd.DataFrame,
        *,
        data_types: dict[str, str] | None,
        features: Sequence[str] | None,
        horizon: int | None,
        train_split: float | None,
        embargo_days: int,
        seed: int,
    ) -> dict[str, Any]:
        """Client-side fit preflight — DataFrame validation, strict feature
        coverage check, row-cadence detection + log line, and the params
        dict the server expects. Shared by :meth:`fit` and :meth:`fit_async`.

        Strict feature check (introduced 0.7.1) — raises ``ValueError`` if:

          - any name in ``features`` is missing from ``real_data.columns``
            (silent subset fits were a frequent footgun: the model would
            train on the overlapping columns and the customer wouldn't
            notice until the wrong-universe bias surfaced downstream); or
          - any numeric column in ``real_data.columns`` isn't listed in
            ``features``, when ``features`` is set explicitly (catches the
            reverse footgun: parquet grew columns the customer forgot to
            add to the feature list).

        Pass ``features=None`` to fit on every numeric column (opt-out —
        no coverage check).
        """
        _validate_real_data(real_data, arg_name="real_data")
        if train_split is not None and not 0.0 < float(train_split) < 1.0:
            raise ValueError(
                f"train_split must be in (0, 1) or None; got {train_split}"
            )
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0; got {embargo_days}")

        feature_list: list[str] | None = list(features) if features else None
        if feature_list is not None:
            df_cols = list(real_data.columns)
            df_set = set(df_cols)
            feat_set = set(feature_list)
            missing = [c for c in feature_list if c not in df_set]
            if missing:
                raise ValueError(
                    f"features references columns not in real_data: {missing}. "
                    f"real_data has: {df_cols}. "
                    "Either rename the columns or drop them from features=."
                )
            # Reverse check: only complain about numeric extras (string /
            # date columns are commonly carried for joins and are fine to
            # ignore — _validate_real_data already required numeric for
            # any column passed in).
            extras = [
                c for c in df_cols
                if c not in feat_set and pd.api.types.is_numeric_dtype(real_data[c])
            ]
            if extras:
                raise ValueError(
                    f"real_data has numeric columns not in features: {extras}. "
                    "Either add them to features= so the joint model learns "
                    "them, or drop them from real_data before fitting "
                    "(silent column subset fits were a frequent footgun)."
                )

        # Resolve the effective feature list BEFORE the data_types check so
        # the error message names the actual columns the model will train on.
        effective_features: list[str] = (
            feature_list
            if feature_list is not None
            else [
                str(c) for c in real_data.columns
                if pd.api.types.is_numeric_dtype(real_data[c])
            ]
        )

        # data_types is required (no default) and every feature must
        # have an allowed type. SDK callers ship it on the wire as
        # ``feature_data_types``.
        normalized_data_types = _require_data_types(
            data_types,
            expected_columns=effective_features,
            arg_name="data_types",
        )

        # NaN guard at the SDK boundary — let the masking signal reach the
        # model untouched. The backend used to silently fillna(0) /
        # nan_to_num before training, which destroyed the masking signal
        # before it ever reached the model.
        _check_nan_fraction(
            real_data,
            features=effective_features,
            arg_name="real_data",
        )

        # 1.1.0 — row cadence is auto-detected from the DatetimeIndex and
        # surfaced for the customer's info line; the wire-frequency value
        # sent to the server is collapsed to one of the four backend-known
        # families via _resolve_wire_frequency (intraday → 'daily' on the
        # wire so the legacy FREQUENCY_DATA_TYPE_TRANSFORMS overrides
        # don't accidentally fire on the customer's at-cadence data).
        cadence_label, median_dt = _detect_row_cadence(real_data.index)
        wire_frequency = _resolve_wire_frequency(cadence_label)
        n_cols = len(effective_features)
        print(
            f"sablier-flow: fitting {n_cols} feature(s) over {len(real_data)} bars  "
            f"[row cadence: {cadence_label} (median Δt={median_dt})]"
        )

        # Translate the customer-facing 3-string vocabulary to the wire
        # 5-string vocabulary the current backend speaks. When
        # sablier-backend goes live on AWS this collapses to identity.
        wire_data_types = _to_wire_data_types(normalized_data_types)

        return {
            "horizon": horizon,
            "target_features": feature_list,
            "conditioning_features": [],
            "feature_data_types": wire_data_types,
            "frequency": wire_frequency,
            "train_split": float(train_split) if train_split is not None else None,
            "embargo_days": int(embargo_days),
            "seed": int(seed),
        }

    # ------------------------------------------------------------------
    # Async-job-handle surface — fit_async / generate_async / validate_async,
    # plus list_jobs / cancel_job / fetch_result.
    # ------------------------------------------------------------------
    #
    # The async methods do EVERY client-side preflight the sync versions do
    # (validation, frequency inference, anchor extraction) and run the open
    # + verify + encrypt + upload halves of the round-trip — they return as
    # soon as the TEE has the encrypted upload. The caller then either
    # polls server-side via :meth:`list_jobs` / :meth:`Client.get_job_status`
    # and downloads via :meth:`fetch_result`, or calls :meth:`cancel_job`
    # before the work is done.

    def _async_dispatch(
        self,
        *,
        kind: str,
        real_data: pd.DataFrame | None,
        params: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> JobHandle:
        """Open + upload only — returns the handle without waiting.

        ``kind`` is the canonical SDK vocabulary (``'fit'`` |
        ``'generate'`` | ``'validate'``) and ships on the wire
        unchanged — :meth:`fetch_result` dispatches on the same
        vocabulary the customer sees.

        The local disk cache is bypassed in async mode: a cache hit would
        skip the TEE round-trip entirely and leave us with no job_id to
        track, which is the wrong contract for an async handle. Callers
        who want cache hits should use the sync method.
        """
        parquet_bytes = _df_to_parquet_bytes(real_data) if real_data is not None else b""
        job_id, result_key = self._open_and_upload(
            kind=kind,
            parquet_bytes=parquet_bytes,
            params=params,
            idempotency_key=idempotency_key,
        )
        return JobHandle(
            job_id=job_id,
            kind=kind,
            result_key_b64=base64.b64encode(result_key).decode("ascii"),
        )

    def fit_async(
        self,
        real_data: pd.DataFrame,
        *,
        data_types: dict[str, str] | None = None,
        features: Sequence[str] | None = None,
        horizon: int | None = None,
        train_split: float | None = 0.8,
        embargo_days: int = 21,
        seed: int | None = None,
        idempotency_key: str | None = None,
    ) -> JobHandle:
        """Async variant of :meth:`fit` — opens the job, ships the upload,
        and returns immediately with a :class:`JobHandle`. Call
        :meth:`fetch_result` later to retrieve the :class:`FitResult`.

        Same contract as :meth:`fit`: ``data_types`` is REQUIRED (no
        default), ``frequency`` is optional and auto-detected from
        ``real_data.index`` when omitted. See :meth:`fit` for the full
        kwarg docs.

        Job control + live progress
        ---------------------------
        Once you hold the returned :class:`JobHandle` (or the bare
        ``handle.job_id``), the SDK exposes everything needed to monitor
        and steer the job from the same process or a different one:

        - :meth:`fetch_result` — block on the handle until it finishes
          (polls server-side; default poll budget is generous).
        - :meth:`list_jobs` — list recent jobs, most-recent first; each
          entry includes ``status``, ``progress`` (dict with ``step``,
          ``phase``, ``message``, ``metrics``, ``total_steps``) and
          ``last_progress_at`` so you can render a heartbeat.
        - :meth:`cancel_job` — cancel by handle or by ``job_id``.
        - :func:`sablier_flow.resume_job` — pick up a previously-saved
          handle from disk after a process restart.

        Persistence note:
            The async path does NOT write a ``~/.sablier/pending_jobs/``
            recovery file — only the sync entrypoints (:meth:`fit` /
            :meth:`generate` / :meth:`validate`) do, which is what
            :meth:`resume` / :func:`sablier_flow.resume_job` read back.
            Async callers own the lifecycle of the returned
            :class:`JobHandle`: persist it (``handle.to_dict()`` →
            ``JobHandle.from_dict()``) if the calling process might die
            before the TEE finishes, otherwise :meth:`resume_job` cannot
            recover this job — the one-shot ``result_key`` lives only
            inside the handle."""
        if horizon is not None:
            _check_fit_horizon_bounds(
                horizon, n_rows=len(real_data), train_split=train_split,
            )
        if seed is None:
            import secrets
            seed = secrets.randbelow(2**31 - 1)
        idempotency_key = _ensure_idempotency_key(idempotency_key)
        params = self._prep_fit_params(
            real_data,
            data_types=data_types,
            features=features,
            horizon=horizon,
            train_split=train_split,
            embargo_days=embargo_days,
            seed=seed,
        )
        return self._async_dispatch(
            kind="fit",
            real_data=real_data,
            params=params,
            idempotency_key=idempotency_key,
        )

    def generate_async(
        self,
        model_id: str,
        *,
        data_types: dict[str, str] | None = None,
        n_paths: int = 1000,
        horizon: int | None = None,
        anchor_data: pd.DataFrame | None = None,
        like: pd.DataFrame | None = None,
        seed: int | None = None,
        idempotency_key: str | None = None,
    ) -> JobHandle:
        """Async variant of :meth:`generate`. Returns a :class:`JobHandle`;
        call :meth:`fetch_result` to retrieve the :class:`GenerationResult`.

        Same contract as :meth:`generate`: ``data_types`` is REQUIRED
        only when ``like`` / ``anchor_data`` is supplied, and
        ``frequency`` is optional and auto-detected from the supplied
        ``like`` / ``anchor_data`` index. See :meth:`generate` for the
        full kwarg docs.

        Note: the ``like=`` index alignment that :meth:`generate` does
        client-side after the result lands cannot be replayed inside
        :meth:`fetch_result` (we have no DataFrame at fetch time). If you
        need ``like=``-style indexing, do it yourself after fetch with
        ``result.with_paths_index(your_window.index)``.

        See :meth:`fit_async` for the full job-control surface
        (:meth:`list_jobs` / :meth:`fetch_result` / :meth:`cancel_job` /
        :func:`sablier_flow.resume_job`) and the handle-persistence
        contract — identical here."""
        if not model_id:
            raise ValueError("model_id is required")
        _check_generate_n_paths_bounds(n_paths)
        if horizon is not None:
            _check_generate_horizon_bounds(horizon)
        if seed is None:
            import secrets
            seed = secrets.randbelow(2**31 - 1)
        idempotency_key = _ensure_idempotency_key(idempotency_key)
        if anchor_data is not None:
            _validate_real_data(
                anchor_data, arg_name="anchor_data", require_min_rows=False,
            )
            self._check_columns_cover_model(
                model_id, anchor_data, arg_name="anchor_data",
            )
        if like is not None:
            _validate_real_data(
                like, arg_name="like", require_min_rows=False,
            )
            self._check_columns_cover_model(model_id, like, arg_name="like")
        # Only require ``data_types=`` when the customer is supplying
        # fresh data (``like=`` / ``anchor_data=``). Without them the
        # model's registered types are reused server-side; see
        # Client.generate for the full rationale. We also relax when
        # ``like=`` / ``anchor_data=`` IS supplied but ``data_types=``
        # is not — the anchoring DataFrame carries the same columns the
        # model was fitted on (enforced by _check_columns_cover_model
        # above), so re-declaring types is pure friction. Explicit
        # ``data_types=`` still validates + forwards as before.
        ref_df = like if like is not None else anchor_data
        normalized_data_types: dict[str, str] | None
        if ref_df is not None and data_types is not None:
            expected_cols = [
                str(c) for c in ref_df.columns
                if pd.api.types.is_numeric_dtype(ref_df[c])
            ]
            normalized_data_types = _require_data_types(
                data_types,
                expected_columns=expected_cols,
                arg_name="data_types",
            )
            _check_nan_fraction(
                ref_df,
                features=expected_cols,
                arg_name="like" if like is not None else "anchor_data",
            )
        elif ref_df is not None:
            expected_cols = [
                str(c) for c in ref_df.columns
                if pd.api.types.is_numeric_dtype(ref_df[c])
            ]
            _check_nan_fraction(
                ref_df,
                features=expected_cols,
                arg_name="like" if like is not None else "anchor_data",
            )
            normalized_data_types = None
        else:
            normalized_data_types = None
        # 1.0.8: mirror the data_types contract on the wire for
        # ``frequency`` (see Client.generate for the rationale). Short
        # like/anchor windows that span a holiday tripped the
        # "irregular index" guard even when the underlying cadence was
        # plain business-daily; the model already registered its
        # training frequency so re-detecting was pure friction.
        like_index: list[str] | None = None
        anchor_prices: dict[str, float] | None = None
        if like is not None:
            horizon = len(like)
            try:
                like_index = [str(ts) for ts in like.index]
            except Exception:
                like_index = None
            try:
                first_row = like.iloc[0]
                anchor_prices = {
                    str(col): float(first_row[col])
                    for col in like.columns
                    if pd.api.types.is_numeric_dtype(like[col])
                }
            except Exception:
                anchor_prices = None
        elif anchor_data is not None:
            # Forward-forecast case — mirror the fix in Client.generate.
            # Anchor at the LAST row of anchor_data so synth continues
            # from "today" rather than checkpoint-end / training-end.
            try:
                last_row = anchor_data.iloc[-1]
                anchor_prices = {
                    str(col): float(last_row[col])
                    for col in anchor_data.columns
                    if pd.api.types.is_numeric_dtype(anchor_data[col])
                }
            except Exception:
                anchor_prices = None

        params: dict[str, Any] = {
            "model_id": model_id,
            "n_paths": int(n_paths),
            "horizon": horizon,
            "seed": int(seed),
        }
        if normalized_data_types is not None:
            params["feature_data_types"] = _to_wire_data_types(normalized_data_types)
        if like_index is not None:
            params["like_index"] = like_index
        if anchor_prices is not None:
            params["anchor_prices"] = anchor_prices
        return self._async_dispatch(
            kind="generate",
            real_data=anchor_data,
            params=params,
            idempotency_key=idempotency_key,
        )

    def validate_async(
        self,
        model_id: str,
        *,
        data_types: dict[str, str] | None = None,
        holdout_data: pd.DataFrame | None = None,
        n_paths: int = 500,
        seed: int | None = None,
        idempotency_key: str | None = None,
    ) -> JobHandle:
        """Async variant of :meth:`validate`. Returns a :class:`JobHandle`;
        call :meth:`fetch_result` to retrieve the :class:`ValidationReport`.

        Same contract as :meth:`validate`: ``data_types`` is REQUIRED
        only when ``holdout_data`` is supplied, and ``frequency`` is
        optional and auto-detected from ``holdout_data.index`` when
        supplied. See :meth:`validate` for the full kwarg docs.

        See :meth:`fit_async` for the full job-control surface
        (:meth:`list_jobs` / :meth:`fetch_result` / :meth:`cancel_job` /
        :func:`sablier_flow.resume_job`) and the handle-persistence
        contract — identical here."""
        if not model_id:
            raise ValueError("model_id is required")
        _check_validate_n_paths_bounds(n_paths)
        if seed is None:
            import secrets
            seed = secrets.randbelow(2**31 - 1)
        idempotency_key = _ensure_idempotency_key(idempotency_key)
        if holdout_data is not None:
            _validate_real_data(
                holdout_data, arg_name="holdout_data", require_min_rows=False,
            )
        # Only require ``data_types=`` when the customer is supplying a
        # fresh slice (see Client.validate for the full rationale).
        # Without holdout_data the server reuses the model's registered
        # types from fit time.
        normalized_data_types: dict[str, str] | None
        if holdout_data is not None:
            expected_cols = [
                str(c) for c in holdout_data.columns
                if pd.api.types.is_numeric_dtype(holdout_data[c])
            ]
            normalized_data_types = _require_data_types(
                data_types,
                expected_columns=expected_cols,
                arg_name="data_types",
            )
            _check_nan_fraction(
                holdout_data,
                features=expected_cols,
                arg_name="holdout_data",
            )
        else:
            normalized_data_types = None
        # 1.1.0 — the `frequency=` kwarg is gone from the public surface;
        # the server reuses the model's registered training frequency on
        # validate. Holdout slices inherit it.
        params: dict[str, Any] = {
            "model_id": model_id,
            "n_paths": int(n_paths),
            "seed": int(seed),
            "holdout": holdout_data is not None,
        }
        if normalized_data_types is not None:
            params["feature_data_types"] = _to_wire_data_types(normalized_data_types)
        return self._async_dispatch(
            kind="validate",
            real_data=holdout_data,
            params=params,
            idempotency_key=idempotency_key,
        )

    def fetch_result(
        self, handle: JobHandle,
    ) -> FitResult | GenerationResult | ValidationReport:
        """Block on a previously-opened async job until it finishes, then
        download + decrypt + materialize the result.

        Dispatches on ``handle.kind``:

          - ``'fit'``      → :class:`FitResult`
          - ``'generate'`` → :class:`GenerationResult`
          - ``'validate'`` → :class:`ValidationReport`

        Pairs with :meth:`fit_async` / :meth:`generate_async` /
        :meth:`validate_async`. Subject to the same poll-timeout budget
        as the sync methods (defaults to 30 min per call)."""
        # 1.0.10 — persona testing surfaced the "I have a job_id string
        # from yesterday, can I get the result back?" path landing on a
        # raw AttributeError from ``handle.result_key_b64``. That's
        # exactly the wrong shape for an SDK error: ``fetch_result``
        # takes a JobHandle (which carries the AES key), and pending-job
        # recovery by id is what :meth:`resume` is for. Surface the typed
        # redirect locally so the caller doesn't need to read the source
        # to figure out which method they wanted.
        if isinstance(handle, str):
            raise TypeError(
                f"fetch_result expects a JobHandle, got str. To resume "
                f"by id use sf.resume_job(job_id={handle!r}) instead."
            )
        result_key = base64.b64decode(handle.result_key_b64.encode("ascii"))
        result_bytes = self._wait_and_decrypt(handle.job_id, handle.kind, result_key)
        payload = JobResultPayload.from_bytes(result_bytes)
        if handle.kind == "fit":
            return _stamp_sdk_version(payload.to_fit_result())
        if handle.kind == "generate":
            return _stamp_sdk_version(payload.to_generation_result())
        if handle.kind == "validate":
            return _stamp_sdk_version(payload.to_validation_report())
        raise ValueError(
            f"unknown handle.kind {handle.kind!r}; expected one of "
            f"'fit' | 'generate' | 'validate'"
        )

    def list_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[JobSummary]:
        """List the caller's jobs, most-recent first.

        Pairs with :meth:`fit_async` etc. so the customer can see what's
        in flight without having to track each handle. Filter by
        ``status`` (``'pending'``, ``'running'``, ``'completed'``,
        ``'failed'``) to narrow the view."""
        resp = self._transport.list_jobs(status=status, limit=int(limit))
        return list(resp.jobs)

    def cancel_job(self, handle_or_id: JobHandle | str) -> None:
        """Cancel a queued or running job. Idempotent on already-terminal
        jobs (returns silently). The job's status flips to ``failed`` with
        ``error_message='cancelled by user'``."""
        job_id = handle_or_id.job_id if isinstance(handle_or_id, JobHandle) else str(handle_or_id)
        self._transport.cancel_job(job_id)

    def resume(
        self, job_id: str,
    ) -> FitResult | GenerationResult | ValidationReport:
        """Resume a paid-but-undelivered job by ``job_id``.

        This is the str-id entry point — pair with :func:`resume_job` at
        the module level. :meth:`fetch_result` deliberately takes a
        :class:`JobHandle` only (the AES result_key has to come from
        somewhere) and raises ``TypeError`` redirecting here when handed
        a bare string.

        The sync entrypoints (:meth:`fit` / :meth:`generate` /
        :meth:`validate`) persist the one-shot result key to
        ``~/.sablier/pending_jobs/{job_id}.json`` immediately after the
        TEE accepts the upload, and clear it only on a successful
        decrypt. Anything that kills the poll loop in between —
        Ctrl-C, network blip, OS reboot, transient ``AuthenticationError``
        on a key rotation — leaves the file in place so the customer can
        recover the result they already paid for without re-running the
        job.

        Dispatches on the persisted ``kind`` exactly like
        :meth:`fetch_result`. Removes the pending-job file on a
        successful decrypt; on a transient failure it stays so the
        customer can call :meth:`resume` again.

        Note:
            Sync-path only. Async dispatches (:meth:`fit_async` /
            :meth:`generate_async` / :meth:`validate_async`) deliberately
            skip the pending-job persistence because the caller already
            holds the :class:`JobHandle` (which carries the same
            ``result_key`` in-memory). To recover an async job,
            persist the handle yourself (``handle.to_dict()`` →
            ``JobHandle.from_dict()``) and pass it back to
            :meth:`fetch_result` — :meth:`resume` will raise
            ``ValueError('no pending job ...')`` for job ids that were
            opened through an async entrypoint.
        """
        if not job_id:
            raise ValueError("job_id is required")
        record = _load_pending_job(job_id)
        if record is None:
            raise ValueError(
                f"no pending job {job_id!r} on disk. Either the job already "
                "completed (result was decrypted + the pending file cleared) "
                "or it was never opened from this machine — pending-job "
                "recovery only works for jobs the local SDK started."
            )
        result_key = base64.b64decode(
            record["result_key_b64"].encode("ascii")
        )
        kind = str(record.get("kind") or "")
        # 1.0.21 — clear the pending file on EVERY terminal outcome,
        # not only on success. The previous code only called
        # _clear_pending_job after _wait_and_decrypt returned normally;
        # if the server returned a terminal failure (RemoteJobError —
        # job cancelled / failed / expired / deleted), the exception
        # propagated and the pending file lived on disk indefinitely.
        # Customers ended up with stale .json files in
        # ~/.sablier/pending_jobs/ that resume() would keep trying to
        # poll forever. Wrap the wait+decrypt in try/except and clear
        # the pending file on RemoteJobError too — only the live retryable
        # cases (network hiccups, attestation transient failures) should
        # leave the pending file in place.
        try:
            result_bytes = self._wait_and_decrypt(job_id, kind, result_key)
        except RemoteJobError:
            # Terminal server-side failure — clear the pending record so
            # the next resume() doesn't hit the same dead handle.
            _clear_pending_job(job_id)
            raise
        _clear_pending_job(job_id)
        payload = JobResultPayload.from_bytes(result_bytes)
        if kind == "fit":
            return _stamp_sdk_version(payload.to_fit_result())
        if kind == "generate":
            return _stamp_sdk_version(payload.to_generation_result())
        if kind == "validate":
            return _stamp_sdk_version(payload.to_validation_report())
        # Unknown kind = corrupted pending file; also clear it so resume()
        # doesn't loop forever on a record we can't decode.
        _clear_pending_job(job_id)
        raise ValueError(
            f"pending job {job_id!r} has unknown kind {kind!r}; expected "
            "one of 'fit' | 'generate' | 'validate'"
        )

    # ------------------------------------------------------------------
    # 1.0.10 cost-accounting visibility — surface estimated spend BEFORE
    # the GPU round-trip and the actual spend AFTER it. Persona testing
    # surfaced this as the #1 ergonomic ask: 5 of 6 users wanted "how
    # much did this just cost me?" without having to thread
    # sf.estimate_cost(...) and sf.credits() through every script.
    # ------------------------------------------------------------------

    def _cost_lines_quiet(self, quiet: bool) -> bool:
        """Resolve whether cost-accounting lines should be suppressed.
        Honors the ``quiet=`` kwarg first, then ``SABLIER_FLOW_QUIET=1``
        as a set-once override for CI / log-sensitive notebooks."""
        if quiet:
            return True
        return os.environ.get("SABLIER_FLOW_QUIET") == "1"

    def _print_estimate_cost_line(
        self,
        *,
        kind: str,
        quiet: bool,
        n_features: int | None = None,
        n_rows: int | None = None,
        horizon: int | None = None,
        n_paths: int | None = None,
    ) -> None:
        """Print the pre-flight cost estimate to stderr in one line.
        Best-effort — if the estimate call fails (transient 5xx, fresh
        account with no quota answer) we swallow the error so an
        ergonomic feature can't break the headline call."""
        if self._cost_lines_quiet(quiet):
            return
        import sys
        try:
            from sablier_flow.client.transport import EstimateCostRequest

            body = EstimateCostRequest(
                kind=kind,
                n_features=n_features,
                n_rows=n_rows,
                horizon=horizon,
                n_paths=n_paths,
            )
            est = self._transport.estimate_cost(body)
            n = round(float(est.estimated_credits))
        except Exception:
            return
        print(
            f"sablier-flow: estimated cost {n} credits "
            "(use sf.estimate_cost(...) to preview before charging).",
            file=sys.stderr,
            flush=True,
        )

    def _print_actual_cost_line(self, *, kind: str, quiet: bool) -> None:
        """Print the post-call actual spend + remaining balance to stderr.
        Best-effort — failures here must not surface to the caller (the
        headline result already succeeded; a credits-endpoint blip would
        be the wrong thing to bubble up)."""
        if self._cost_lines_quiet(quiet):
            return
        import sys
        try:
            wire = self._transport.credits()
            balance = int(wire.available)
            # Actual spend isn't surfaced as a sibling field on the
            # credits response, so look it up via the most recent
            # filtered usage row for this kind (best-effort).
            actual = self._lookup_recent_usage_credits(kind)
        except Exception:
            return
        if actual is None:
            # Couldn't resolve a precise actual; emit balance-only.
            print(
                f"sablier-flow: call complete (remaining balance: {balance}).",
                file=sys.stderr,
                flush=True,
            )
            return
        print(
            f"sablier-flow: actual cost {round(actual)} credits "
            f"(remaining balance: {balance}).",
            file=sys.stderr,
            flush=True,
        )

    def _lookup_recent_usage_credits(self, kind: str) -> float | None:
        """Best-effort fetch of the just-charged credits for ``kind``.
        Returns ``None`` if the usage endpoint is unavailable or the
        most-recent matching row can't be resolved."""
        try:
            resp = self._transport.usage(
                since=None, until=None, kind=kind, limit=1,
            )
            items = list(getattr(resp, "items", []) or [])
            if not items:
                return None
            first = items[0]
            return float(first.credits_charged)
        except Exception:
            return None

    def _safe_model_features(self, model_id: str) -> list[str] | None:
        """Return the model's registered feature list, or ``None`` if
        the lookup fails. Used by the single-feature dependence-metric
        suppression path: best-effort, never a hard contract."""
        try:
            model = self.get_model(model_id)
        except Exception:
            return None
        return list(model.features or [])


# ============================================================================
# Helpers
# ============================================================================


class _ProgressRenderer:
    """Render server-emitted progress events to the user.

    Tries tqdm first (the common case in notebooks / CLI). If tqdm
    isn't installed, falls back to a quiet line-printer that only
    shows when the phase or step changes meaningfully. Either way
    this is purely cosmetic — the SDK doesn't depend on progress
    rendering for correctness.

    The server's ``progress`` field is a JSON string; we parse it
    inside :meth:`update` and tolerate any shape mismatch silently
    so a backend change can't break the client.
    """

    @classmethod
    def create(cls, *, kind: str) -> _ProgressRenderer:
        # generate/validate complete in seconds — no bar to avoid flicker.
        # fit is the long one where a bar pays off.
        if kind != "fit":
            return _NullRenderer()
        try:
            from tqdm.auto import tqdm  # type: ignore
            return _TqdmRenderer(tqdm)
        except Exception:
            return _PrintRenderer()

    def update(self, progress_str: str | None, status: str) -> None:
        ...

    def close(self) -> None:
        ...


class _NullRenderer(_ProgressRenderer):
    def update(self, progress_str: str | None, status: str) -> None: return None
    def close(self) -> None: return None


class _TqdmRenderer(_ProgressRenderer):
    """tqdm-backed progress bar."""

    def __init__(self, tqdm_cls: Any) -> None:
        self._tqdm_cls = tqdm_cls
        self._bar: Any = None
        self._last_step = 0
        self._last_phase = ""

    def update(self, progress_str: str | None, status: str) -> None:
        if not progress_str:
            return
        try:
            import json
            event = json.loads(progress_str)
        except Exception:
            return
        phase = str(event.get("phase") or "")
        step = int(event.get("step") or 0)
        total = event.get("total_steps")
        metrics = event.get("metrics") or {}

        # Pre-0.7.1 used ``event.get("message") or phase`` as the bar
        # description. The server's message often baked in a snapshot
        # like ``"epoch 0/500"`` taken at the start of training and never
        # updated, so the bar permanently read ``epoch 0/500: 78%|███|
        # 397/500`` for the whole 13-min run — the right-side counter
        # was the true progress, the left-side label lied. Use the bare
        # phase token (``"training"``) instead; the counter already
        # shows step/total, and the phase changes are surfaced via
        # set_description below when they actually happen.
        desc = phase or "training"

        # Lazy-construct the bar so the first server event sets total.
        if self._bar is None and total is not None:
            self._bar = self._tqdm_cls(total=int(total), desc=desc, leave=True)
            self._last_phase = phase
            self._last_step = 0
        if self._bar is None:
            return

        # If the phase changed (preprocessing → training → completed),
        # reset the bar's description.
        if phase and phase != self._last_phase:
            self._bar.set_description(desc)
            self._last_phase = phase

        delta = max(step - self._last_step, 0)
        if delta:
            self._bar.update(delta)
            self._last_step = step
        if metrics:
            postfix = {k: (f"{v:.4f}" if isinstance(v, float) else v)
                       for k, v in metrics.items() if k != "total_steps"}
            self._bar.set_postfix(postfix, refresh=False)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class _PrintRenderer(_ProgressRenderer):
    """tqdm-less fallback. Prints a single line per phase change."""

    def __init__(self) -> None:
        self._last_message = ""

    def update(self, progress_str: str | None, status: str) -> None:
        if not progress_str:
            return
        try:
            import json
            event = json.loads(progress_str)
        except Exception:
            return
        message = str(event.get("message") or event.get("phase") or "")
        if message and message != self._last_message:
            print(f"[sablier-flow] {message}", flush=True)
            self._last_message = message

    def close(self) -> None:
        return None


def _with_paths_index(result: GenerationResult, index: Any) -> GenerationResult:
    """Return a copy of ``result`` with ``paths_index`` set. Pulled out
    of the inline call site because :class:`GenerationResult` is frozen
    and :func:`dataclasses.replace` is the correct way to substitute
    a field on a frozen dataclass."""
    from dataclasses import replace
    return replace(result, paths_index=index)


# Metric names + categories the SDK marks ``status='not_applicable'``
# when a fit had only one feature. Cross-asset dependence is undefined
# on a 1-D model; the server still emits the rows because the metric
# registry is universal, and pre-1.0.10 a single-feature ``validate()``
# would land on ``overall='fail'`` driven entirely by metrics that
# couldn't possibly pass. Two layers of detection so we catch both
# category-tagged and category-less payloads:
_DEPENDENCE_CATEGORIES = frozenset({"dependence"})
_DEPENDENCE_METRIC_NAMES = frozenset({
    "pearson_correlation",
    "spearman_correlation",
    "cross_correlation",
    "tail_dependence",
    "tail_dependence_upper",
    "tail_dependence_lower",
    "copula_distance",
    "non_elliptical",
})


def _is_dependence_metric(name: str, entry: Any) -> bool:
    """True when ``entry`` is a dependence-family metric — either by its
    ``category`` field or by a well-known name. Robust to entries that
    aren't dicts (older payload shapes / future variants)."""
    if isinstance(entry, dict):
        cat = str(entry.get("category") or "").lower()
        if cat in _DEPENDENCE_CATEGORIES:
            return True
    return name in _DEPENDENCE_METRIC_NAMES


def _suppress_dependence_metrics_when_single_feature(
    report: ValidationReport,
    *,
    model_features: list[str] | None,
) -> ValidationReport:
    """Mark dependence-family metric rows as ``status='not_applicable'``
    when the model was trained on a single feature, and recompute the
    overall verdict ignoring them.

    Why: cross-asset / cross-feature metrics (pearson correlation,
    copula distance, tail dependence, ...) are undefined on a one-feature
    model. The server emits the rows anyway because the metric registry
    is universal; pre-1.0.10 single-feature ``validate()`` calls landed
    on ``overall='fail'`` driven entirely by structurally-inapplicable
    rows. Best-effort: when ``model_features`` isn't known (the
    ``get_model`` lookup failed transiently) we return the report
    unchanged rather than guess.
    """
    if model_features is None:
        return report
    if len(model_features) != 1:
        return report
    if not report.metrics:
        return report

    new_metrics: dict[str, Any] = {}
    suppressed_passed = 0
    suppressed_failed = 0
    for name, entry in report.metrics.items():
        if _is_dependence_metric(name, entry):
            if isinstance(entry, dict):
                # Stamp status=not_applicable so downstream readers can
                # tell why the row no longer drives the verdict, and
                # null out the boolean ``passed`` so weighted aggregates
                # that respect ``passed`` skip it too.
                marked = dict(entry)
                marked["status"] = "not_applicable"
                # Track what the server thought before we suppressed it
                # so a curious customer can still inspect the raw value.
                marked.setdefault("original_passed", entry.get("passed"))
                marked["passed"] = None
                new_metrics[name] = marked
                if entry.get("passed") is False:
                    suppressed_failed += 1
                elif entry.get("passed") is True:
                    suppressed_passed += 1
            else:
                # Non-dict entry — store the not_applicable marker
                # under a wrapper dict so the row isn't lost outright.
                new_metrics[name] = {
                    "status": "not_applicable",
                    "original_value": entry,
                }
        else:
            new_metrics[name] = entry

    # Recompute overall if suppression removed any failures: a fail
    # driven entirely by dependence rows on a 1-feature model is wrong;
    # promote it to the worst-case of the remaining applicable rows.
    new_overall = report.overall
    if suppressed_failed > 0 and report.overall == "fail":
        applicable = [
            v for k, v in new_metrics.items()
            if isinstance(v, dict)
            and v.get("status") != "not_applicable"
            and "passed" in v
        ]
        if applicable:
            any_failed = any(v.get("passed") is False for v in applicable)
            new_overall = "warn" if any_failed else "pass"
        else:
            # All metrics were dependence — without any applicable rows
            # the verdict can't be derived; flip to 'warn' as a neutral
            # signal (don't bless a model we have no signal on).
            new_overall = "warn"

    from dataclasses import replace
    return replace(report, overall=new_overall, metrics=new_metrics)


def _stamp_sdk_version(
    result: FitResult | GenerationResult | ValidationReport,
) -> Any:
    """Overwrite ``result.sdk_version`` with the *client* SDK's version
    string so customers reading ``.sdk_version`` always see the wheel
    that decoded the payload, not the worker's internal pipeline
    version (which used to leak through as e.g. ``'0.5.1'`` — the
    semver of the bundled training pipeline, not anything a customer
    can act on).

    Pre-1.0.8, ``FitResult.sdk_version`` / ``GenerationResult.sdk_version``
    / ``ValidationReport.sdk_version`` held whatever string the worker
    serialized into the payload. From 1.0.8 on the worker's value is
    ignored and the client overwrites it via :func:`dataclasses.replace`
    (the dataclasses are frozen so in-place mutation would fail). If
    the destination dataclass doesn't have an ``sdk_version`` field
    (older payload variants) we return it unchanged — best-effort, not
    a hard contract.
    """
    from dataclasses import fields, replace

    from sablier_flow import __version__ as client_version

    try:
        if not any(f.name == "sdk_version" for f in fields(result)):
            return result
        return replace(result, sdk_version=client_version)
    except Exception:
        # Defensive: replace can raise on misconfigured dataclasses. The
        # stamp is informational; we'd rather hand back the original
        # result than fail the call over a cosmetic field.
        return result


def _model_info_to_dataclass(info: Any) -> Model:
    """Map the wire ``ModelInfoResponse`` pydantic model to the public
    :class:`Model` dataclass — keeps pydantic out of the customer-visible
    type surface."""
    return Model(
        model_id=info.model_id,
        features=list(info.features),
        training_horizon=int(info.training_horizon),
        n_assets=int(info.n_assets),
        status=str(info.status),
        training_start_date=info.training_start_date,
        training_end_date=info.training_end_date,
        holdout_start_date=info.holdout_start_date,
        holdout_end_date=info.holdout_end_date,
        train_split=info.train_split,
        embargo_days=info.embargo_days,
        sdk_version=info.sdk_version,
        training_loss=info.training_loss,
        created_at=info.created_at,
        last_used_at=info.last_used_at,
        expires_at=info.expires_at,
    )


# Local bounds checks so an obviously-bad horizon / n_paths fails
# sub-second on the customer's machine instead of after a 1-2 min queue +
# job-start + cryptic Cloud Run failure. The server enforces an
# architectural backstop independently (the estimate endpoint caps horizon
# at 5000; the denoiser's learned positional-embedding table —
# ``sablier_flow_internal`` ``MAX_HORIZON`` — tops out at 8192).
_FIT_HORIZON_FLOOR = 504        # always-allowed cap (the legacy flat value)
_FIT_HORIZON_CEILING = 5000     # never exceed the server's hard backstop
_FIT_OBS_LENGTH = 200           # encoder history consumed per training window
_FIT_MIN_NONOVERLAP_SPANS = 2   # require >= this many non-overlapping horizon spans
_GENERATE_HORIZON_MAX = 5000
_GENERATE_N_PATHS_MAX = 1_000_000
_VALIDATE_N_PATHS_MAX = 10_000


def _fit_horizon_max(n_rows: int | None, train_split: float | None) -> int:
    """Data-adaptive upper bound on the *training* horizon.

    A flat cap both under-serves data-rich customers (who can train a
    longer window robustly) and rubber-stamps data-poor ones. Training
    horizon ``H`` consumes ``H`` bars per window, and the number of
    *non-overlapping* spans of length ``H`` in the training portion is
    ``(train_rows - obs_length) // H``. Requiring at least
    :data:`_FIT_MIN_NONOVERLAP_SPANS` such spans is the simplest honest
    anti-memorization bound — a long window on thin data leaves too few
    independent spans and the model just memorizes (the server's
    NN-distance audit is the backstop; failing fast here is kinder).

    The result is clamped to ``[_FIT_HORIZON_FLOOR, _FIT_HORIZON_CEILING]``
    so (a) no horizon that passed the old flat-504 check ever newly fails
    and (b) the cap never exceeds the server backstop. Falls back to the
    flat floor when ``n_rows`` is unknown.
    """
    if not n_rows or n_rows <= 0:
        return _FIT_HORIZON_FLOOR
    ts = train_split if (train_split is not None and 0.0 < train_split <= 1.0) else 1.0
    train_rows = int(n_rows * ts)
    usable = max(train_rows - _FIT_OBS_LENGTH, 0)
    data_max = usable // _FIT_MIN_NONOVERLAP_SPANS
    return int(min(_FIT_HORIZON_CEILING, max(_FIT_HORIZON_FLOOR, data_max)))


def _check_fit_horizon_bounds(
    horizon: Any,
    *,
    n_rows: int | None = None,
    train_split: float | None = None,
) -> None:
    """Reject horizon <= 0 or above the data-adaptive cap before any
    network call. The cap scales with the supplied history length (see
    :func:`_fit_horizon_max`); when ``n_rows`` is omitted it falls back to
    the flat :data:`_FIT_HORIZON_FLOOR`, so existing callers keep their
    behavior. Mirrors the server backstop so callers see the same error
    locally that they'd otherwise hit after a paid round-trip."""
    hmax = _fit_horizon_max(n_rows, train_split)
    try:
        h = int(horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"horizon must be an integer in (0, {hmax}]; got {horizon!r}"
        ) from exc
    if h <= 0 or h > hmax:
        ts = train_split if train_split is not None else 1.0
        ctx = f" for {n_rows} rows (train_split={ts:g})" if n_rows else ""
        raise ValueError(
            f"horizon must be in (0, {hmax}]; got {h}. The fit-horizon cap "
            f"is {hmax}{ctx}: a longer training window needs proportionally "
            f"more history to leave enough non-overlapping spans to train on "
            f"(>= {_FIT_MIN_NONOVERLAP_SPANS}). The generator is "
            f"horizon-agnostic — train at the cap and generate any horizon up "
            f"to {_GENERATE_HORIZON_MAX} without retraining."
        )


def _check_generate_n_paths_bounds(n_paths: Any) -> None:
    """Reject n_paths <= 0 or > :data:`_GENERATE_N_PATHS_MAX` before any
    network call."""
    try:
        n = int(n_paths)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"n_paths must be a positive integer <= {_GENERATE_N_PATHS_MAX}; "
            f"got {n_paths!r}"
        ) from exc
    if n <= 0 or n > _GENERATE_N_PATHS_MAX:
        raise ValueError(
            f"n_paths must be in (0, {_GENERATE_N_PATHS_MAX}]; got {n}. "
            "For backtest augmentation 100-1000 is plenty; structural "
            "metrics stabilize around 500."
        )


def _check_generate_horizon_bounds(horizon: Any) -> None:
    """Reject horizon <= 0 or > :data:`_GENERATE_HORIZON_MAX` before any
    network call. Generate horizon cap is higher than fit's because the
    sampler integrates over any window — quality degrades modestly as you
    stretch further past the training horizon but the model still runs."""
    try:
        h = int(horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"horizon must be a positive integer <= {_GENERATE_HORIZON_MAX}; "
            f"got {horizon!r}"
        ) from exc
    if h <= 0 or h > _GENERATE_HORIZON_MAX:
        raise ValueError(
            f"horizon must be in (0, {_GENERATE_HORIZON_MAX}]; got {h}. "
            "Quality degrades modestly as you stretch past the trained "
            "horizon; multi-decade generation is rarely what you want."
        )


def _check_validate_n_paths_bounds(n_paths: Any) -> None:
    """Reject n_paths <= 0 or > :data:`_VALIDATE_N_PATHS_MAX` before any
    network call. Validate caps below generate because structural metrics
    saturate around 500-1000 paths — paying for more just burns budget."""
    try:
        n = int(n_paths)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"n_paths must be a positive integer <= {_VALIDATE_N_PATHS_MAX}; "
            f"got {n_paths!r}"
        ) from exc
    if n <= 0 or n > _VALIDATE_N_PATHS_MAX:
        raise ValueError(
            f"n_paths must be in (0, {_VALIDATE_N_PATHS_MAX}]; got {n}. "
            "Structural-metric variance saturates around 500-1000 paths; "
            "higher values burn budget without improving the verdict."
        )


def _require_data_types(
    data_types: Any,
    *,
    expected_columns: Sequence[str],
    arg_name: str = "data_types",
) -> dict[str, str]:
    """Enforce the data-layer contract: data_types must be a dict
    mapping every expected column to one of :data:`ALLOWED_DATA_TYPES`.

    Raises :class:`TypeError` when ``data_types`` is ``None`` — agents
    and humans benefit from a sub-second local error citing the allowed
    set over a 2-minute queue + train + cryptic decoder failure.

    Raises :class:`ValueError` when keys / values are wrong: missing
    columns, unknown values, or columns named that the caller never said
    they were generating.
    """
    if data_types is None:
        raise TypeError(
            f"{arg_name}= is required (no default). Pass a dict mapping "
            f"each feature column to its canonical type, e.g. "
            f"{arg_name}={{'SPY': 'price', 'VIX': 'level'}}. Allowed "
            f"values: {sorted(ALLOWED_DATA_TYPES)}."
        )
    if not isinstance(data_types, dict):
        raise TypeError(
            f"{arg_name}= must be a dict[str, str]; got "
            f"{type(data_types).__name__}. Allowed values: "
            f"{sorted(ALLOWED_DATA_TYPES)}."
        )
    expected = list(expected_columns)
    missing = [c for c in expected if c not in data_types]
    if missing:
        raise ValueError(
            f"missing data_type for {missing}. Pass {arg_name}={{...}} "
            f"covering every feature: {expected}. Allowed values: "
            f"{sorted(ALLOWED_DATA_TYPES)}."
        )
    # 1.1.0 — friendly migration error for the old 5-string vocabulary.
    # No customers existed before this rename, but the in-wheel demo
    # registry + every public example used the old names; surface a
    # precise pointer if a caller still passes one.
    _LEGACY_ALIASES = {
        "rate": "level",
        "volatility": "level",
        "index": "price",
    }
    bad: list[tuple[str, Any]] = [
        (k, v) for k, v in data_types.items() if v not in ALLOWED_DATA_TYPES
    ]
    if bad:
        legacy_hits = [(k, v, _LEGACY_ALIASES[v]) for k, v, in [
            (k, v) for k, v in bad if v in _LEGACY_ALIASES
        ]]
        if legacy_hits:
            mapping = ", ".join(
                f"{k!r}: {old!r} → {new!r}" for k, old, new in legacy_hits
            )
            raise ValueError(
                f"{arg_name} uses retired data-type name(s) (collapsed in 1.1.0): "
                f"{mapping}. Update to the new vocabulary "
                f"({sorted(ALLOWED_DATA_TYPES)}) — see "
                f"https://docs.sablier.ai/SDK/#data-types"
            )
        bad_pairs = ", ".join(f"{k!r}={v!r}" for k, v in bad)
        raise ValueError(
            f"{arg_name} has unsupported value(s): {bad_pairs}. Allowed "
            f"values: {sorted(ALLOWED_DATA_TYPES)}."
        )
    # Return a normalized copy so callers can stash it without re-checking.
    return {str(k): str(v) for k, v in data_types.items()}


def _to_wire_data_types(customer_data_types: dict[str, str]) -> dict[str, str]:
    """Translate the SDK's customer-facing 3-string vocabulary to the
    legacy 5-string vocabulary the server still speaks.

    Customer-facing → wire:
        'price'  → 'price'    (LOG_RETURN, unchanged)
        'level'  → 'rate'     (DIFFERENCE — same transform server-side; the
                               backend's enum still distinguishes 'rate' /
                               'volatility' / 'index' but they all dispatch
                               to DIFFERENCE at daily, which is what we want.
                               We pick 'rate' as the canonical wire value.)
        'return' → 'return'   (IDENTITY, unchanged)

    Removed when sablier-backend goes live on AWS and natively speaks
    {'price', 'level', 'return'}.
    """
    return {col: _WIRE_DATA_TYPE_MAPPING[t] for col, t in customer_data_types.items()}


def _detect_row_cadence(index: Any) -> tuple[str, pd.Timedelta]:
    """Auto-detect the row cadence of a DatetimeIndex.

    Returns ``(human_label, median_delta)``. ``human_label`` is the
    customer-facing description for the info line ("intraday (5-min)",
    "daily", "monthly", etc.). The actual wire value sent to the server
    is always ``'daily'`` via :func:`_resolve_wire_frequency` for
    backwards compatibility with the server's
    transform pipeline (see ``_WIRE_DATA_TYPE_MAPPING`` rationale near
    the top of this module).

    Raises only on degenerate inputs: non-DatetimeIndex, single-row
    series, or grossly irregular sampling (95th-percentile gap > 3× the
    median, which is almost certainly an event log rather than a bar
    series).
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        raise ValueError(
            "row-cadence auto-detection requires a DatetimeIndex with at "
            "least 2 rows. Pass a DataFrame with a proper DatetimeIndex."
        )
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        raise ValueError(
            "row-cadence auto-detection requires at least 2 distinct "
            "timestamps."
        )
    median = deltas.median()
    p95 = deltas.quantile(0.95)
    if p95 > 3 * median:
        raise ValueError(
            f"irregular index detected (median Δt={median}, p95 Δt={p95}). "
            "The model requires a uniform-cadence DatetimeIndex — resample "
            "your DataFrame to a uniform grid before fitting."
        )
    one_minute = pd.Timedelta(minutes=1)
    one_hour = pd.Timedelta(hours=1)
    one_day = pd.Timedelta(days=1)
    if median < one_minute:
        return (f"intraday ({median.total_seconds():.0f}s)", median)
    if median < one_hour:
        return (f"intraday ({int(median.total_seconds() / 60)}-min)", median)
    if median < one_day:
        return (f"intraday ({median.total_seconds() / 3600:.1f}h)", median)
    if median <= pd.Timedelta(days=3):
        return ("daily", median)
    if median <= pd.Timedelta(days=10):
        return ("weekly", median)
    if median <= pd.Timedelta(days=45):
        return ("monthly", median)
    if median <= pd.Timedelta(days=100):
        return ("quarterly", median)
    raise ValueError(
        f"row cadence median Δt={median} is coarser than quarterly; not "
        "supported by the model. Resample to quarterly or finer."
    )


def _resolve_wire_frequency(cadence_label: str) -> str:
    """Map the SDK's detected row cadence to a wire-frequency string the
    current backend accepts. The backend understands
    ``{'daily', 'weekly', 'monthly', 'quarterly'}`` and uses the value
    to gate ``FREQUENCY_DATA_TYPE_TRANSFORMS`` overrides (YoY / MoM /
    LEVEL_STANDARDIZED for forward-filled lower-cadence features).

    In 1.1.0 we never want those overrides to fire (the SDK no longer
    exposes stair-step support; customer data is assumed at-cadence).
    So we collapse the wire value to ``'daily'`` for everything that
    isn't one of the four canonical families — including intraday
    (which the backend doesn't natively know but treats fine as a
    timestep sequence with day-of-year cyclical embedding).
    """
    if cadence_label in ("daily", "weekly", "monthly", "quarterly"):
        return cadence_label
    # Any intraday cadence label collapses to 'daily' on the wire.
    return "daily"


def _check_nan_fraction(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    arg_name: str = "real_data",
) -> None:
    """Reject only columns whose NaN fraction (post-ffill) exceeds
    :data:`_MAX_NAN_FRACTION`. Pass NaNs through to the server otherwise —
    the model masks them, and the silent ``fillna(0)`` / ``nan_to_num``
    layers the backend used to apply destroyed that masking signal
    before it reached the model.
    """
    bad: list[tuple[str, float]] = []
    for col in features:
        if col not in df.columns:
            continue
        series = df[col]
        # ffill first — a column that opens missing then fills in is
        # fine; what we're catching is "this column is missing for most
        # of history" (e.g., a feature added halfway through training).
        post_ffill = series.ffill()
        nan_frac = float(post_ffill.isna().mean())
        if nan_frac > _MAX_NAN_FRACTION:
            bad.append((col, nan_frac))
    if bad:
        details = ", ".join(f"{col!r} ({frac:.0%} NaN)" for col, frac in bad)
        raise ValueError(
            f"{arg_name} has column(s) with post-ffill NaN fraction over "
            f"{_MAX_NAN_FRACTION:.0%}: {details}. Drop these columns, "
            "extend history so they're populated for >30% of bars, or "
            "load a different dataset. NaNs are otherwise passed through "
            "to the model (which masks them)."
        )

    # Surface smaller gaps that pass the reject threshold but still matter:
    # interior NaNs get forward-filled before the transform server-side, so
    # the affected bars carry no fresh signal. This is the classic symptom of
    # calendar misalignment between feature sources (e.g. an FX/macro column
    # on a different trading calendar than equity columns). Warn, don't fail —
    # passing gaps through is intentional, but the customer should know.
    gappy: list[tuple[str, float]] = []
    for col in features:
        if col not in df.columns:
            continue
        raw_frac = float(df[col].isna().mean())
        if 0.0 < raw_frac <= _MAX_NAN_FRACTION:
            gappy.append((col, raw_frac))
    if gappy:
        details = ", ".join(f"{col!r} ({frac:.1%})" for col, frac in gappy)
        warnings.warn(
            f"{arg_name} has NaN gaps in column(s): {details}. These bars are "
            "forward-filled before generation and carry no fresh signal — "
            "often a sign of calendar misalignment between feature sources. "
            "Align/clean these columns for best generation quality.",
            stacklevel=2,
        )


def _validate_real_data(
    df: Any,
    *,
    arg_name: str,
    min_rows: int = 200,
    require_min_rows: bool = True,
) -> None:
    """Sanity-check a customer DataFrame before the network round-trip.

    Server requirements: monotonic DatetimeIndex, all-numeric columns,
    ≥ ``min_rows`` rows (only enforced when ``require_min_rows`` — anchor
    DataFrames passed to ``generate`` don't need 200 rows). Raising
    locally with a precise message saves the customer ~1-2 min of
    queue-and-job-start before discovering "your data is wrong."
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{arg_name} must be a pandas DataFrame; got {type(df).__name__}. "
            "Load your data with pd.read_parquet / pd.read_csv / your "
            "warehouse client and pass the resulting DataFrame directly."
        )
    if df.empty:
        raise ValueError(f"{arg_name} is empty — nothing to fit on.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"{arg_name}.index must be a pd.DatetimeIndex (one row per "
            f"bar). Got {type(df.index).__name__}. Fix with "
            f"`df.index = pd.to_datetime(df.index)` or "
            f"`df.set_index('date_column')` before passing it in."
        )
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            f"{arg_name}.index must be monotonic increasing. "
            "Fix with `df = df.sort_index()`."
        )
    if df.index.has_duplicates:
        n_dup = int(df.index.duplicated().sum())
        raise ValueError(
            f"{arg_name}.index has {n_dup} duplicate timestamp(s). "
            "Fix with `df = df[~df.index.duplicated(keep='first')]` "
            "or deduplicate upstream."
        )
    # All columns must be numeric — the server's data pipeline z-scores
    # and computes returns, which is undefined on strings / datetimes.
    non_numeric = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise TypeError(
            f"{arg_name} has non-numeric columns: {non_numeric[:5]}"
            f"{'...' if len(non_numeric) > 5 else ''}. Drop / convert "
            "them before fitting (Sablier models numeric features only)."
        )
    if require_min_rows and len(df) < min_rows:
        raise ValueError(
            f"{arg_name} has {len(df)} rows; need at least {min_rows} "
            "to fit a flow model. Use a longer history."
        )


def _df_to_parquet_bytes(df: pd.DataFrame | None) -> bytes:
    """Serialize a DataFrame to Parquet bytes (in-memory, no temp file).

    Returns ``b""`` when ``df`` is None — used by data-less calls such
    as :meth:`Client.generate` with no explicit anchor.
    """
    if df is None:
        return b""
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="zstd", index=True)
    return buf.getvalue()


def _default_pinned_digest() -> str:
    """The image digest the SDK release pins.

    Resolution order:
      1. ``SABLIER_FLOW_PINNED_IMAGE_DIGEST`` env var (staging override).
      2. ``sablier_flow/_pinned_image_digest.txt`` baked into the wheel
         by the release pipeline (.github/workflows/release.yml writes
         this from the tee-image-digest artifact before the wheel is
         built).
      3. placeholder ``sha256:000…`` — only meaningful before
         the first signed release ships.

    The order is deliberate: env var wins so staging deploys can target
    a non-pinned image, but production wheels always have a real digest
    baked in. Customers can re-verify by rebuilding from the
    sablier-flow tag (the build is reproducible).
    """
    import os
    from importlib.resources import files

    env_override = os.environ.get("SABLIER_FLOW_PINNED_IMAGE_DIGEST")
    if env_override:
        return env_override
    try:
        baked = (files("sablier_flow") / "_pinned_image_digest.txt").read_text().strip()
        if baked:
            return baked
    except (FileNotFoundError, ModuleNotFoundError):
        pass
    return "sha256:" + "0" * 64


def _ensure_idempotency_key(explicit: str | None) -> str:
    """Auto-generate an Idempotency-Key for state-changing entrypoints
    when the caller didn't pass one.

    The backend caches ``(user_id, key)`` for 24h, so a transient
    retry — same key, identical body — yields the same job_id + quote
    instead of double-charging the customer. The SDK *always* sends a
    key now (uuid4 hex) so even un-instrumented retry loops are safe;
    customers who want explicit replay-protection can still pass their
    own key.
    """
    if explicit:
        return explicit
    import uuid as _uuid
    return _uuid.uuid4().hex


# ============================================================================
# Pending-job recovery — persist (job_id, result_key) so a mid-poll
# crash doesn't leave a paid-for job permanently undecryptable.
# ============================================================================


def _pending_jobs_dir() -> Any:
    """Resolve the pending-jobs directory, honoring SABLIER_FLOW_PENDING_DIR
    for tests. Creates the dir with mode 0700 if missing.

    Path-typed (returns :class:`pathlib.Path`) but typed loosely so the
    rest of the file can stay free of an extra import on the hot path.
    """
    import contextlib as _ctx
    from pathlib import Path

    override = os.environ.get("SABLIER_FLOW_PENDING_DIR")
    d = Path(override) if override else Path.home() / ".sablier" / "pending_jobs"
    d.mkdir(parents=True, exist_ok=True)
    # Tighten permissions on POSIX. Best-effort: on Windows os.chmod is a
    # near-no-op so we don't gate the rest of the SDK on it succeeding.
    with _ctx.suppress(OSError):
        os.chmod(d, 0o700)
    return d


def _persist_pending_job(
    *,
    job_id: str,
    kind: str,
    result_key: bytes,
    endpoint: str,
    api_key: str,
) -> None:
    """Write the recovery record for an in-flight job.

    Schema is the source of truth for :meth:`Client.resume` — bump
    ``layout_version`` if the fields ever change so old files are
    rejected cleanly rather than silently mis-decoded.
    """
    import datetime as _dt
    import hashlib
    import json

    record = {
        "layout_version": 1,
        "job_id": job_id,
        "kind": kind,
        "result_key_b64": base64.b64encode(result_key).decode("ascii"),
        "endpoint": endpoint,
        # Store only the prefix — enough for the customer to see which
        # account opened the job, never enough to authenticate as them.
        "api_key_prefix": (api_key or "")[:8],
        "api_key_fingerprint": hashlib.sha256(
            (api_key or "").encode("utf-8")
        ).hexdigest()[:16],
        "started_at_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    path = _pending_jobs_dir() / f"{job_id}.json"
    # Atomic-ish write: a partial write would leave the customer with a
    # corrupt JSON file that resume() couldn't parse. The .tmp + rename
    # dance avoids that on POSIX.
    tmp = path.with_suffix(".json.tmp")
    data = json.dumps(record, separators=(",", ":")).encode("utf-8")
    fd = os.open(
        str(tmp),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
    except Exception:
        # If we couldn't even write the temp file, drop it and re-raise
        # so the customer notices — silent failure here would defeat the
        # recovery contract.
        import contextlib as _ctx
        with _ctx.suppress(OSError):
            os.unlink(tmp)
        raise
    os.replace(tmp, path)


def _load_pending_job(job_id: str) -> dict[str, Any] | None:
    """Read back a pending-job record. Returns ``None`` if the file is
    missing (already resumed / never opened on this host)."""
    import json

    path = _pending_jobs_dir() / f"{job_id}.json"
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        return dict(json.loads(raw.decode("utf-8")))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"pending-job file {path} is corrupt: {exc}. Delete it and "
            "re-run the job (the result_key it would have held is no "
            "longer recoverable)."
        ) from exc


def _clear_pending_job(job_id: str) -> None:
    """Remove the pending-job record. Idempotent — missing file is fine."""
    import contextlib as _ctx
    path = _pending_jobs_dir() / f"{job_id}.json"
    with _ctx.suppress(FileNotFoundError):
        os.unlink(path)


# ============================================================================
# Convenience module-level function
# ============================================================================


# Kwargs the Client constructor accepts (vs everything else, which flows to
# Client.fit / Client.generate / Client.validate). Kept here so the convenience shortcut and the
# @augment decorator can split a single **kwargs blob without duplicating the
# list of fields.
_CLIENT_INIT_KWARGS = frozenset({
    "endpoint",
    "pinned_image_digest",
    "attestation_mode",
    "transport",
    "timeout_s",
    "poll_interval_s",
    "poll_timeout_s",
    "verify",
    "cache_dir",
})


# Env vars the convenience shortcut consults when the matching kwarg is None.
# Lets a customer set their config once and call generate(model_id) /
# @augment without re-passing endpoint+cert on every call.
_ENV_FALLBACKS = {
    "endpoint":            "SABLIER_FLOW_ENDPOINT",
    "verify":              "SABLIER_FLOW_CERT",
    "pinned_image_digest": "SABLIER_FLOW_PINNED_IMAGE_DIGEST",
    "attestation_mode":    "SABLIER_FLOW_ATTESTATION_MODE",
}


def _reject_unknown_kwargs(
    kwargs: dict[str, Any], *, allowed_call_kwargs: frozenset[str],
) -> None:
    """Kwarg-typo guard for module-level shortcuts.

    The sync ``sf.fit`` / ``sf.generate`` / ``sf.validate`` shortcuts
    explicitly list every accepted call-kwarg and use a ``**extra``
    catch-all to raise on typos. The bare shortcuts (``sf.whoami``,
    ``sf.credits``, …) historically forwarded **kwargs straight into
    ``_build_client`` which silently dropped anything that wasn't a
    client-init kwarg — so ``sf.whoami(api_kye='…')`` succeeded with
    the env-var fallback and the typo'd value was lost.

    After popping client-init keys + ``profile`` + the shortcut's own
    documented call-kwargs, anything left is a typo and we raise
    :class:`TypeError` listing the offenders (matching the existing
    ``fit/generate/validate`` shortcuts' error message).
    """
    extras = sorted(
        k for k in kwargs
        if k not in _CLIENT_INIT_KWARGS
        and k != "profile"
        and k not in allowed_call_kwargs
    )
    if extras:
        raise TypeError(f"unexpected keyword arguments: {extras}")


def _resolve_endpoint(
    *,
    explicit: str | None,
    stored: str | None,
) -> str:
    """Resolve the final endpoint URL using the canonical priority order.

    Shared by :class:`Client.__init__` and :func:`_build_client` so the
    sync constructor and the module-level shortcuts behave identically:

      1. explicit ``endpoint=`` kwarg (caller's intent always wins)
      2. ``SABLIER_FLOW_ENDPOINT`` env var (set-once config for CI / agents)
      3. ``stored`` endpoint from ~/.sablier/credentials (only when no env)
      4. hardcoded production default

    1.0.8 — once the source is picked, auto-append ``/v1`` when the
    resolved value is a bare host (no path component). Pre-1.0.8
    customers who passed ``endpoint='https://flow.sablier.ai'`` got a
    404 on whoami because the SDK appended ``/whoami`` while the server
    expected ``/v1/whoami``. The auto-append removes that footgun
    without touching values that already carry a path (those stay
    as-is and we log a debug line so power users can spot a mismatch).
    """
    if explicit:
        raw = explicit
        # 1.0.21 — even an explicit kwarg must clear the safety gate.
        # Previously Client(endpoint='http://my.cdn/...', api_key=...)
        # would happily send the api_key in cleartext, and
        # Client(endpoint='https://evil.example.com/...') would ship it
        # off-domain. The kwarg path was the only one bypassing
        # validate_stored_endpoint; tighten it here so the api_key is
        # never transmitted to an unvetted endpoint regardless of how
        # the customer set it.
        _enforce_endpoint_allowlist(raw, source="endpoint= kwarg")
    else:
        env_endpoint = os.environ.get("SABLIER_FLOW_ENDPOINT")
        if env_endpoint:
            # 1.0.21 — same gate for env-var override. CI / agent setups
            # that set SABLIER_FLOW_ENDPOINT must point at an allowlisted
            # host; otherwise the api_key leaks to whatever URL the env
            # var was pointed at.
            _enforce_endpoint_allowlist(env_endpoint, source="SABLIER_FLOW_ENDPOINT env var")
            raw = env_endpoint
        elif stored:
            # `stored` already went through validate_stored_endpoint in
            # load_credentials — keep it as the safe fallback.
            raw = stored
        else:
            # The hardcoded default already includes /v1 — return it directly.
            return "https://flow.sablier.ai/v1"
    return _ensure_v1_suffix(raw)


def _enforce_endpoint_allowlist(url: str, *, source: str) -> None:
    """Reject ``url`` if it doesn't pass the same allowlist
    :func:`validate_stored_endpoint` applies to ~/.sablier/credentials.

    Raises ``ValueError`` rather than warning — for the kwarg + env-var
    paths the api_key would otherwise be transmitted to the rejected
    host, so a hard refusal at construction time is strictly safer than
    a logged warning + silent send."""
    from sablier_flow.client.login import validate_stored_endpoint
    if validate_stored_endpoint(url) is None:
        raise ValueError(
            f"refusing to use endpoint={url!r} from {source}: only "
            f"https:// URLs pointing at sablier.ai / *.sablier.ai (or the "
            f"Cloud Run canonical hostnames) are allowed. The Sablier "
            f"client must not ship your API key to an unvetted host. "
            f"If you need a non-standard endpoint for local development, "
            f"pass it via the credentials file under a named profile "
            f"(the file's stored endpoint goes through the same allowlist "
            f"but produces a warning rather than refusing construction)."
        )


def _ensure_v1_suffix(endpoint: str) -> str:
    """Append ``/v1`` to a customer-supplied endpoint when it has no
    path component, leave it alone otherwise.

    Cases:
      - ``https://host``                → ``https://host/v1`` (auto-append)
      - ``https://host/``               → ``https://host/v1`` (auto-append)
      - ``https://host/v1``             → unchanged (idempotent)
      - ``https://host/v1/``            → unchanged (trailing slash kept)
      - ``https://host/api/v2``         → unchanged + debug log
        (non-default path — the customer almost certainly meant it)
    """
    import logging
    from urllib.parse import urlparse

    try:
        parsed = urlparse(endpoint)
    except Exception:
        return endpoint
    path = (parsed.path or "").rstrip("/")
    if path == "":
        # Bare host — append /v1.
        # urlunparse would normalize the URL in ways that could surprise
        # power users (e.g. lowercasing the host); cheaper to just
        # string-cat onto a stripped trailing slash.
        base = endpoint.rstrip("/")
        result = f"{base}/v1"
        logging.getLogger(__name__).debug(
            "endpoint %r missing /v1 suffix; using %r", endpoint, result,
        )
        return result
    if path == "/v1":
        # Already canonical — no-op.
        return endpoint
    # Non-default path — could be a staging proxy / api gateway. Leave it
    # alone but emit a one-shot debug line so power users can spot
    # mismatches against the v1 wire contract.
    logging.getLogger(__name__).debug(
        "endpoint %r has non-default path %r; leaving as-is "
        "(expected '/v1' for the production wire contract)",
        endpoint, parsed.path,
    )
    return endpoint


def _build_client(api_key: str | None, kwargs: dict[str, Any]) -> tuple[Client, dict[str, Any]]:
    """Split a kwargs blob into Client-constructor vs call-kwargs, fall
    back to env vars on the constructor side, and return a fresh Client
    + the leftover call-kwargs.

    Resolution order for ``api_key``:
      1. explicit kwarg
      2. ``SABLIER_FLOW_API_KEY`` env var (CI / scripted use)
      3. ``~/.sablier/credentials`` written by :func:`sablier_flow.login`
    """
    import os

    # ``profile=`` is a Client-init concept too but it drives the
    # *credential lookup* here, not the constructor kwarg blob — pop it
    # off before we split, then thread it both into load_credentials()
    # AND back into the Client(profile=...) call so customers who set
    # SABLIER_FLOW_API_KEY in env still get their named profile applied
    # for the endpoint-from-creds path.
    profile = kwargs.pop("profile", "default")

    api_key = api_key or os.environ.get("SABLIER_FLOW_API_KEY")
    stored_endpoint: str | None = None
    if not api_key:
        # Match Client.__init__'s fallback so module-level shortcuts
        # (sf.whoami, sf.fit, ...) pick up keys written by sf.login()
        # without forcing the user to also export an env var.
        from sablier_flow.client.login import load_credentials
        creds = load_credentials(profile=profile)
        if creds and creds.get("api_key"):
            api_key = str(creds["api_key"])
            stored_endpoint = creds.get("endpoint")
    if not api_key:
        raise ValueError(
            "api_key is required. Run `sablier_flow.login()` to authenticate "
            "interactively, set SABLIER_FLOW_API_KEY in the env, or pass "
            "api_key='sk_live_...' to this call. Get a key at "
            "https://sablier.ai → Settings → API Keys "
            "(new accounts get 500 free credits)."
        )

    client_kwargs: dict[str, Any] = {}
    call_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in _CLIENT_INIT_KWARGS:
            client_kwargs[key] = value
        else:
            call_kwargs[key] = value

    # Apply env-var fallbacks for *non-endpoint* constructor knobs. The
    # endpoint is resolved via the shared :func:`_resolve_endpoint` helper
    # below so both this call site and Client.__init__ honor the exact
    # same priority order (explicit > env > creds-file > hardcoded).
    for key, env_name in _ENV_FALLBACKS.items():
        if key == "endpoint":
            continue
        if key not in client_kwargs and (val := os.environ.get(env_name)):
            client_kwargs[key] = val

    client_kwargs["endpoint"] = _resolve_endpoint(
        explicit=client_kwargs.get("endpoint"),
        stored=stored_endpoint,
    )

    return Client(api_key=api_key, profile=profile, **client_kwargs), call_kwargs


def fit(
    real_data: pd.DataFrame,
    *,
    api_key: str | None = None,
    data_types: dict[str, str] | None = None,
    features: Sequence[str] | None = None,
    horizon: int | None = None,
    train_split: float | None = 0.8,
    embargo_days: int = 21,
    seed: int | None = None,
    idempotency_key: str | None = None,
    quiet: bool = False,
    endpoint: str | None = None,
    pinned_image_digest: str | None = None,
    attestation_mode: Any = _ATTESTATION_MODE_DEFAULT,
    verify: bool | str | None = None,
    cache_dir: str | os.PathLike[str] | bool | None = None,
    profile: str = "default",
    **extra: Any,
) -> FitResult:
    """One-shot shortcut for :meth:`Client.fit`.

    Same kwargs as :meth:`Client.fit` plus the :class:`Client` constructor
    knobs (``endpoint``, ``verify``, ``cache_dir`` etc.) so a notebook
    can swing the whole call in one line. ``**extra`` surfaces any
    unrecognized kwarg as a clear ``TypeError`` instead of silently
    swallowing it.

    ``data_types`` is REQUIRED (see :data:`ALLOWED_DATA_TYPES`).
    """
    if extra:
        raise TypeError(f"unexpected keyword arguments: {sorted(extra)}")
    # Filter None so unspecified kwargs fall through to _ENV_FALLBACKS
    # (passing key=None blocks the env fallback in _build_client).
    # ``attestation_mode`` defaults to the sentinel; only forward it
    # into client_kwargs when the caller passed a real value, so the
    # Client constructor's own sentinel default (and its silent-on-default
    # contract — see test_attestation_1_0_11) is preserved.
    ctor: dict[str, Any] = {k: v for k, v in {
        "endpoint": endpoint,
        "pinned_image_digest": pinned_image_digest,
        "verify": verify,
        "cache_dir": cache_dir,
        "profile": profile,
    }.items() if v is not None}
    if attestation_mode is not _ATTESTATION_MODE_DEFAULT:
        ctor["attestation_mode"] = attestation_mode
    client, _ = _build_client(api_key, ctor)
    return client.fit(
        real_data,
        data_types=data_types,
        features=features,
        horizon=horizon,
        train_split=train_split,
        embargo_days=embargo_days,
        seed=seed,
        idempotency_key=idempotency_key,
        quiet=quiet,
    )


def generate(
    model_id: str,
    *,
    api_key: str | None = None,
    data_types: dict[str, str] | None = None,
    n_paths: int = 1000,
    horizon: int | None = None,
    anchor_data: pd.DataFrame | None = None,
    like: pd.DataFrame | None = None,
    seed: int | None = None,
    idempotency_key: str | None = None,
    quiet: bool = False,
    endpoint: str | None = None,
    pinned_image_digest: str | None = None,
    attestation_mode: Any = _ATTESTATION_MODE_DEFAULT,
    verify: bool | str | None = None,
    cache_dir: str | os.PathLike[str] | bool | None = None,
    profile: str = "default",
    **extra: Any,
) -> GenerationResult:
    """One-shot shortcut for :meth:`Client.generate`.

    Pass ``like=window`` to derive horizon + index + anchor from a single
    DataFrame (recommended for backtest-augmentation), or
    ``anchor_data=df`` for explicit anchor control.

    ``data_types`` is REQUIRED (see :data:`ALLOWED_DATA_TYPES`)."""
    if extra:
        raise TypeError(f"unexpected keyword arguments: {sorted(extra)}")
    # Filter None so unspecified kwargs fall through to _ENV_FALLBACKS
    # (passing key=None blocks the env fallback in _build_client).
    # ``attestation_mode`` defaults to the sentinel; only forward when
    # the caller passed a real value so the Client's silent-on-default
    # warning contract is preserved (test_attestation_1_0_11).
    ctor: dict[str, Any] = {k: v for k, v in {
        "endpoint": endpoint,
        "pinned_image_digest": pinned_image_digest,
        "verify": verify,
        "cache_dir": cache_dir,
        "profile": profile,
    }.items() if v is not None}
    if attestation_mode is not _ATTESTATION_MODE_DEFAULT:
        ctor["attestation_mode"] = attestation_mode
    client, _ = _build_client(api_key, ctor)
    return client.generate(
        model_id,
        data_types=data_types,
        n_paths=n_paths,
        horizon=horizon,
        anchor_data=anchor_data,
        like=like,
        seed=seed,
        idempotency_key=idempotency_key,
        quiet=quiet,
    )


def validate(
    model_id: str,
    *,
    api_key: str | None = None,
    data_types: dict[str, str] | None = None,
    holdout_data: pd.DataFrame | None = None,
    n_paths: int = 500,
    seed: int | None = None,
    idempotency_key: str | None = None,
    quiet: bool = False,
    endpoint: str | None = None,
    pinned_image_digest: str | None = None,
    attestation_mode: Any = _ATTESTATION_MODE_DEFAULT,
    verify: bool | str | None = None,
    cache_dir: str | os.PathLike[str] | bool | None = None,
    profile: str = "default",
    **extra: Any,
) -> ValidationReport:
    """One-shot shortcut for :meth:`Client.validate`.

    Pass ``holdout_data=df`` for an explicit OOS slice; omit it to let
    the server use the train/test split it persisted at fit time.

    ``data_types`` is REQUIRED (see :data:`ALLOWED_DATA_TYPES`)."""
    if extra:
        raise TypeError(f"unexpected keyword arguments: {sorted(extra)}")
    # Filter None so unspecified kwargs fall through to _ENV_FALLBACKS
    # (passing key=None blocks the env fallback in _build_client).
    # ``attestation_mode`` defaults to the sentinel; only forward when
    # the caller passed a real value so the Client's silent-on-default
    # warning contract is preserved (test_attestation_1_0_11).
    ctor: dict[str, Any] = {k: v for k, v in {
        "endpoint": endpoint,
        "pinned_image_digest": pinned_image_digest,
        "verify": verify,
        "cache_dir": cache_dir,
        "profile": profile,
    }.items() if v is not None}
    if attestation_mode is not _ATTESTATION_MODE_DEFAULT:
        ctor["attestation_mode"] = attestation_mode
    client, _ = _build_client(api_key, ctor)
    return client.validate(
        model_id,
        data_types=data_types,
        holdout_data=holdout_data,
        n_paths=n_paths,
        seed=seed,
        idempotency_key=idempotency_key,
        quiet=quiet,
    )


def list_models(
    *,
    limit: int = 50,
    api_key: str | None = None,
    **kwargs: Any,
) -> list[Model]:
    """Shortcut for :meth:`Client.list_models` using a one-shot Client."""
    _reject_unknown_kwargs(kwargs, allowed_call_kwargs=frozenset())
    client, _ = _build_client(api_key, kwargs)
    return client.list_models(limit=limit)


def get_model(
    model_id: str,
    *,
    api_key: str | None = None,
    **kwargs: Any,
) -> Model:
    """Shortcut for :meth:`Client.get_model` using a one-shot Client."""
    _reject_unknown_kwargs(kwargs, allowed_call_kwargs=frozenset())
    client, _ = _build_client(api_key, kwargs)
    return client.get_model(model_id)


def delete_model(
    model_id: str,
    *,
    api_key: str | None = None,
    **kwargs: Any,
) -> None:
    """Shortcut for :meth:`Client.delete_model` using a one-shot Client."""
    _reject_unknown_kwargs(kwargs, allowed_call_kwargs=frozenset())
    client, _ = _build_client(api_key, kwargs)
    return client.delete_model(model_id)


# ---- Account / billing / pre-flight shortcuts ------------------------------


def ping(*, api_key: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Shortcut for :meth:`Client.ping`. ``api_key`` is optional (the
    /v1/health endpoint is public) but pass it if you want to verify a
    specific key resolves on the way in too."""
    _reject_unknown_kwargs(kwargs, allowed_call_kwargs=frozenset())
    if api_key is None:
        import os
        api_key = os.environ.get("SABLIER_FLOW_API_KEY") or "no-key"
    client, _ = _build_client(api_key, kwargs)
    return client.ping()


def whoami(*, api_key: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Shortcut for :meth:`Client.whoami`."""
    _reject_unknown_kwargs(kwargs, allowed_call_kwargs=frozenset())
    client, _ = _build_client(api_key, kwargs)
    return client.whoami()


def credits(*, api_key: str | None = None, **kwargs: Any) -> CreditsBalance:
    """Shortcut for :meth:`Client.credits`. Returns a
    :class:`~sablier_flow.types.CreditsBalance`."""
    _reject_unknown_kwargs(kwargs, allowed_call_kwargs=frozenset())
    client, _ = _build_client(api_key, kwargs)
    return client.credits()


def usage(
    *,
    since: str | None = None,
    until: str | None = None,
    kind: str | None = None,
    limit: int = 100,
    api_key: str | None = None,
    **kwargs: Any,
) -> list[UsageEvent]:
    """Shortcut for :meth:`Client.usage`. Returns a list of
    :class:`~sablier_flow.types.UsageEvent`."""
    _reject_unknown_kwargs(kwargs, allowed_call_kwargs=frozenset())
    client, _ = _build_client(api_key, kwargs)
    return client.usage(since=since, until=until, kind=kind, limit=limit)


def usage_summary(
    *,
    period: str = "month",
    api_key: str | None = None,
    **kwargs: Any,
) -> UsageSummary:
    """Shortcut for :meth:`Client.usage_summary`. Returns a
    :class:`~sablier_flow.types.UsageSummary`."""
    _reject_unknown_kwargs(kwargs, allowed_call_kwargs=frozenset())
    client, _ = _build_client(api_key, kwargs)
    return client.usage_summary(period=period)


def estimate_cost(
    kind: str,
    *,
    real_data: pd.DataFrame | None = None,
    features: Sequence[str] | None = None,
    horizon: int | None = None,
    n_paths: int | None = None,
    n_features: int | None = None,
    n_rows: int | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Shortcut for :meth:`Client.estimate_cost`.

    ``n_features`` / ``n_rows`` are forwarded straight to the wire body
    so customers without the DataFrame on hand (e.g. CI estimating
    spend before pulling a 1 GB parquet) can still get a quote."""
    _reject_unknown_kwargs(kwargs, allowed_call_kwargs=frozenset())
    client, _ = _build_client(api_key, kwargs)
    return client.estimate_cost(
        kind, real_data=real_data, features=features,
        horizon=horizon, n_paths=n_paths,
        n_features=n_features, n_rows=n_rows,
    )


# ---- Async-job-handle shortcuts --------------------------------------------


def fit_async(
    real_data: pd.DataFrame,
    *,
    api_key: str | None = None,
    **kwargs: Any,
) -> JobHandle:
    """Shortcut for :meth:`Client.fit_async` using a one-shot Client.
    Same env-var fallbacks as :func:`fit`.

    Returns a :class:`JobHandle` immediately. To monitor or finalize the
    job from anywhere (same process or a different one), use:

        sf.list_jobs()              # status + live progress dict
        sf.fetch_result(handle)     # block until done
        sf.cancel_job(handle)       # cancel
        sf.resume_job(job_id)       # rehydrate by id

    See :meth:`Client.fit_async` for the full kwarg list."""
    client, call_kwargs = _build_client(api_key, kwargs)
    return client.fit_async(real_data, **call_kwargs)


def generate_async(
    model_id: str,
    *,
    api_key: str | None = None,
    n_paths: int = 1000,
    **kwargs: Any,
) -> JobHandle:
    """Shortcut for :meth:`Client.generate_async` using a one-shot Client."""
    client, call_kwargs = _build_client(api_key, kwargs)
    return client.generate_async(model_id, n_paths=n_paths, **call_kwargs)


def validate_async(
    model_id: str,
    *,
    api_key: str | None = None,
    holdout_data: pd.DataFrame | None = None,
    **kwargs: Any,
) -> JobHandle:
    """Shortcut for :meth:`Client.validate_async` using a one-shot Client."""
    client, call_kwargs = _build_client(api_key, kwargs)
    return client.validate_async(model_id, holdout_data=holdout_data, **call_kwargs)


def fetch_result(
    handle: JobHandle,
    *,
    api_key: str | None = None,
    **kwargs: Any,
) -> FitResult | GenerationResult | ValidationReport:
    """Shortcut for :meth:`Client.fetch_result` using a one-shot Client.

    Useful for scripts that opened a job in one process and now want to
    pick the result up from another — persist the handle via
    ``handle.to_dict()`` / ``JobHandle.from_dict()`` and call this with
    the same ``api_key`` + ``endpoint``.

    Passing a raw ``job_id`` string here raises :class:`TypeError` with
    a pointer to :func:`resume_job` — pending-job recovery by id is a
    separate code path because the AES result_key comes from disk, not
    from the (missing) handle."""
    if isinstance(handle, str):
        raise TypeError(
            f"fetch_result expects a JobHandle, got str. To resume "
            f"by id use sf.resume_job(job_id={handle!r}) instead."
        )
    _reject_unknown_kwargs(kwargs, allowed_call_kwargs=frozenset())
    client, _ = _build_client(api_key, kwargs)
    return client.fetch_result(handle)


def list_jobs(
    *,
    status: str | None = None,
    limit: int = 50,
    api_key: str | None = None,
    **kwargs: Any,
) -> list[JobSummary]:
    """Shortcut for :meth:`Client.list_jobs`."""
    _reject_unknown_kwargs(kwargs, allowed_call_kwargs=frozenset())
    client, _ = _build_client(api_key, kwargs)
    return client.list_jobs(status=status, limit=limit)


def cancel_job(
    handle_or_id: JobHandle | str,
    *,
    api_key: str | None = None,
    **kwargs: Any,
) -> None:
    """Shortcut for :meth:`Client.cancel_job`."""
    _reject_unknown_kwargs(kwargs, allowed_call_kwargs=frozenset())
    client, _ = _build_client(api_key, kwargs)
    return client.cancel_job(handle_or_id)


def resume_job(
    job_id: str,
    *,
    api_key: str | None = None,
    **kwargs: Any,
) -> FitResult | GenerationResult | ValidationReport:
    """Shortcut for :meth:`Client.resume` using a one-shot Client.

    Recovers the result of a paid-but-undelivered job whose result_key
    is still on disk under ``~/.sablier/pending_jobs/``. Use this after
    a Ctrl-C / network blip / OS reboot interrupted ``fit`` / ``generate``
    / ``validate`` mid-poll — the file is written before polling starts
    so the customer is never left with an undecryptable charged job.

    Note:
        Sync-path only. Jobs opened via :func:`fit_async` /
        :func:`generate_async` / :func:`validate_async` do not write a
        pending-job file because the caller already holds the
        :class:`JobHandle` (which carries the same ``result_key``).
        Recover async jobs by persisting the handle yourself
        (``handle.to_dict()`` → ``JobHandle.from_dict()``) and calling
        :func:`fetch_result` — :func:`resume_job` raises
        ``ValueError('no pending job ...')`` for async-opened job ids.
    """
    _reject_unknown_kwargs(kwargs, allowed_call_kwargs=frozenset())
    client, _ = _build_client(api_key, kwargs)
    return client.resume(job_id)


def validate_data(real_data: pd.DataFrame) -> None:
    """Client-side DataFrame check — same rules :meth:`Client.fit`
    enforces (DatetimeIndex, all-numeric columns, ≥ 200 rows, monotonic,
    no duplicates). Raises a precise ``TypeError`` / ``ValueError`` with
    an actionable message if anything fails.

    Pure local — no network, no API key. Use this in a pipeline or a
    pre-commit hook to catch bad input before paying for the queue +
    job start.
    """
    _validate_real_data(real_data, arg_name="real_data")


