"""HTTP transport contract between the SDK and the TEE service.

This module defines the **wire protocol** the customer's SDK and the
proprietary TEE service speak. Everything is split along three seams:

  1. **Wire models** (Pydantic) — what bytes move between the two parties.
     These are the source of truth that the server-side FastAPI handlers
     in ``server/api/`` must mirror exactly.

  2. **Transport protocol** (typing.Protocol) — the abstract HTTP-ish
     surface the Client depends on. Anything matching this shape can be
     injected. The default is :class:`HttpxTransport` over real HTTPS;
     tests inject :class:`InMemoryTransport` to exercise the whole
     handshake in-process without a server.

  3. **HttpxTransport** — the real-world impl over ``httpx`` with retries
     and the API-key auth header.

Job lifecycle (as POSTed by the Client):

    POST /v1/jobs                       {kind, params}           -> {job_id, attestation_quote}
                                        (kind in {fit, generate, validate})
    PUT  /v1/jobs/{id}/data             {envelope_bytes}         -> 204
    GET  /v1/jobs/{id}                                           -> {status, ...}
    GET  /v1/jobs/{id}/result                                    -> {ciphertext_bytes}

For long-running training the client polls GET /v1/jobs/{id} until
``status`` is ``"completed"`` (or ``"failed"``).

Workstream D real impl will provide a live ``HttpxTransport`` against
the production endpoint. Today this module is the spec the TEE service
implements, and the in-memory fake lets us test the whole protocol now.
"""

from __future__ import annotations

import base64
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

__all__ = [
    "AttestationQuoteResponse",
    "AuthenticationError",
    "CreateJobRequest",
    "HttpxTransport",
    "InMemoryTransport",
    "JobKind",
    "JobNotFoundError",
    "JobStatus",
    "JobStatusResponse",
    "ModelNotFoundError",
    "RemoteJobError",
    "ResultResponse",
    "SablierClientError",
    "Transport",
    "TransportError",
]


JobKind = Literal["fit", "generate", "validate"]
JobStatus = Literal["pending", "running", "completed", "failed"]


# ============================================================================
# Wire models — same schema on both ends of the HTTPS call
# ============================================================================


class CreateJobRequest(BaseModel):
    """POST /v1/jobs body. Sent by the client to kick off a new job."""

    kind: JobKind
    # Free-form parameter dict; semantics depend on `kind`. Examples:
    #   {"target_features": [...], "horizon": 30, "n_paths": 1000, "seed": 42}
    params: dict[str, Any] = Field(default_factory=dict)
    # Data-layer kwargs the backend keys off (``feature_data_types`` +
    # ``frequency``) ship as top-level fields, siblings of ``kind`` and
    # ``params``. The route handler reads them straight off the envelope
    # and persists them onto the job_params it hands the worker.
    feature_data_types: dict[str, str] | None = None
    frequency: str | None = None


class AttestationQuoteResponse(BaseModel):
    """POST /v1/jobs response. The TEE replies with a fresh attestation
    quote so the client can verify it before encrypting + uploading
    plaintext data."""

    job_id: str
    # Standard-base64 string of the attestation quote bytes
    # (themselves a urlsafe-b64-encoded JSON envelope — see
    # client/attestation.py). Two layers because the inner layer is
    # the verbatim TEE-signed payload; the outer is JSON-transport-
    # safe encoding. Use :meth:`quote_bytes` on this model to recover
    # the inner bytes.
    attestation_quote: str

    @property
    def quote_bytes(self) -> bytes:
        """Decoded attestation quote bytes — feed to
        :meth:`AttestationQuote.from_wire`."""
        return base64.b64decode(self.attestation_quote)

    @classmethod
    def from_bytes_quote(cls, *, job_id: str, quote_bytes: bytes) -> AttestationQuoteResponse:
        """Server-side constructor: takes raw quote bytes and applies
        the wire-safe base64 wrapping."""
        return cls(
            job_id=job_id,
            attestation_quote=base64.b64encode(quote_bytes).decode("ascii"),
        )


class JobStatusResponse(BaseModel):
    """GET /v1/jobs/{id} response. Polled by the client during training."""

    job_id: str
    status: JobStatus
    # Progress hint. Historically a human-readable string (e.g. "epoch 12/50");
    # since the async worker refactor the API may instead return a structured
    # heartbeat dict (e.g. {"step": 30, "phase": "training"}). Accept both
    # shapes plus ``None`` so older and newer servers both deserialize cleanly.
    progress: dict | str | None = None
    # ISO-8601 timestamp string of the last progress heartbeat from the
    # worker, or ``None`` if no heartbeat has been emitted yet. Optional
    # so older servers that don't send the field still deserialize.
    last_progress_at: str | None = None
    # Populated when status == "failed". The TEE never leaks customer data
    # into this string — it's always a category code + safe message.
    error_message: str | None = None


class ResultResponse(BaseModel):
    """GET /v1/jobs/{id}/result response. The TEE returns the result
    ciphertext, encrypted to the customer's KMS key."""

    job_id: str
    # Standard-base64 string of the result ciphertext bytes (the
    # AES-GCM-encrypted JobResultPayload). Use :attr:`ciphertext_bytes`
    # to recover the raw bytes on the client side.
    ciphertext: str

    @property
    def ciphertext_bytes(self) -> bytes:
        return base64.b64decode(self.ciphertext)

    @classmethod
    def from_bytes_ciphertext(cls, *, job_id: str, ciphertext_bytes: bytes) -> ResultResponse:
        """Server-side constructor: takes raw ciphertext bytes and applies
        the wire-safe base64 wrapping."""
        return cls(
            job_id=job_id,
            ciphertext=base64.b64encode(ciphertext_bytes).decode("ascii"),
        )


# ----- Model-management wire models (GET /v1/models, etc.) -------------------


class ModelInfoResponse(BaseModel):
    """One row of GET /v1/models or the full body of GET /v1/models/{id}.

    Mirrors the server's ``ModelInfo`` pydantic class verbatim so the
    SDK can deserialize directly. Optional fields default to ``None``
    because not every server row populates every column."""

    model_id: str
    # Unified field shipped by the backend (column order matches fit-time).
    features: list[str]
    training_horizon: int
    training_start_date: str | None = None
    training_end_date: str | None = None
    holdout_start_date: str | None = None
    holdout_end_date: str | None = None
    train_split: float | None = None
    embargo_days: int | None = None
    sdk_version: str | None = None
    training_loss: float | None = None
    n_assets: int
    status: str
    created_at: str | None = None
    last_used_at: str | None = None
    expires_at: str | None = None


class ListModelsResponse(BaseModel):
    """GET /v1/models response."""

    models: list[ModelInfoResponse]
    total_returned: int


# ----- Account / billing / pre-flight wire models (GET /v1/me, /credits, etc.) ----


class WhoAmIResponse(BaseModel):
    """Identity returned by ``GET /v1/me``."""

    user_id: str
    email: str | None = None
    name: str | None = None
    tier: str = "free"
    is_admin: bool = False
    api_key_id: str | None = None
    api_key_prefix: str | None = None
    api_key_name: str | None = None


class CreditsResponse(BaseModel):
    """Current credit balance returned by ``GET /v1/credits``."""

    available: int
    monthly_allocation: int
    monthly_used: int
    purchased: int
    tier: str


class UsageItem(BaseModel):
    """One row of ``GET /v1/usage``."""

    job_id: str
    kind: str
    status: str
    credits_charged: float
    n_assets: int | None = None
    created_at: str
    completed_at: str | None = None
    duration_s: float | None = None


class UsageResponse(BaseModel):
    """``GET /v1/usage`` response — paginated job history."""

    items: list[UsageItem]
    total_returned: int
    total_credits: float


class UsageSummaryResponse(BaseModel):
    """``GET /v1/usage/summary`` response — aggregate over a window."""

    period_start: str
    period_end: str
    total_credits: float
    by_kind: dict[str, dict[str, Any]]


class EstimateCostRequest(BaseModel):
    """Body for ``POST /v1/jobs/estimate`` pre-flight cost check."""

    kind: str
    n_features: int | None = None
    n_rows: int | None = None
    horizon: int | None = None
    train_split: float | None = None
    n_paths: int | None = None


class EstimateCostResponse(BaseModel):
    """``POST /v1/jobs/estimate`` response — heuristic cost band."""

    estimated_credits: float
    low: float
    high: float
    estimated_duration_s: float | None = None
    notes: list[str] = []


class HealthResponse(BaseModel):
    """``GET /v1/health`` response — server liveness + version envelope."""

    status: str = "ok"
    server_version: str
    min_sdk_version: str


class JobSummary(BaseModel):
    """One row of :class:`ListJobsResponse`. Identifies a job by id +
    kind + status without forcing the SDK to fetch each one individually.
    """

    job_id: str
    kind: str
    status: str
    created_at: str
    completed_at: str | None = None
    credits_charged: float | None = None
    error_message: str | None = None


class ListJobsResponse(BaseModel):
    """``GET /v1/jobs`` response — most-recent-first."""

    jobs: list[JobSummary]
    total_returned: int


# ============================================================================
# Errors
# ============================================================================


class TransportError(Exception):
    """Base class for all transport-layer failures.

    Carries the *transport* layer failure modes — network blips, malformed
    JSON bodies, retry-budget exhaustion. 4xx server responses are surfaced
    via :class:`SablierClientError` (and its subclasses) instead.
    """


class SablierClientError(TransportError):
    """A 4xx response from the hosted service that the SDK could not turn
    into a more specific exception.

    Every 4xx the server can return is documented to include a JSON body
    with a ``detail`` field — the customer-facing error message. We
    surface it verbatim so the SDK reads like a thin shim over the
    server's own error vocabulary. The ``__str__`` puts the server
    detail first (this is what shows up in tracebacks and notebooks),
    followed by the originating URL for debuggability.

    Use :class:`JobNotFoundError` / :class:`ModelNotFoundError` to catch
    the 404 cases specifically; the base class catches all other 4xx
    (e.g. 400 validation, 403 forbidden, 422 unprocessable).
    """

    def __init__(self, status: int, detail: str, url: str) -> None:
        super().__init__(f"{detail} (HTTP {status} on {url})")
        self.status = status
        self.detail = detail
        self.url = url


class AuthenticationError(SablierClientError):
    """Raised when the server rejects the SDK's API key with HTTP 401.

    The customer's API key is either missing, malformed, or has been
    revoked (rotated out, disabled by an admin, or expired). The SDK
    surfaces this as a dedicated exception so callers can catch the
    revocation case explicitly instead of digging through raw
    ``httpx.HTTPStatusError`` instances — common in long-running
    notebooks or scheduled jobs that may outlive the key that started
    them. The message includes the server-supplied ``detail`` field when
    the 401 body is JSON, otherwise a generic message.

    Subclasses :class:`SablierClientError` so that ``except
    SablierClientError:`` reliably catches every 4xx the SDK can raise,
    including auth failures. Callers that only want auth handling can
    still catch :class:`AuthenticationError` specifically.
    """

    def __init__(self, detail: str, url: str = "/") -> None:
        super().__init__(401, detail, url)


class JobNotFoundError(SablierClientError):
    """The remote service has no job with that id (HTTP 404 on a
    ``/jobs/...`` path). Also raised when a job exists but is owned by a
    different API key — the server collapses both into the same 404 to
    avoid enumeration leaks."""


class ModelNotFoundError(SablierClientError):
    """The remote service has no model with that id (HTTP 404 on a
    ``/models/...`` path). Like :class:`JobNotFoundError`, the server
    collapses "wrong owner" into "not found" so there's no enumeration
    probe; the SDK passes the server's ``detail`` through verbatim."""


class RemoteJobError(TransportError):
    """The remote job ran but failed inside the TEE (status='failed').
    Carries the safe error_message from the TEE."""

    def __init__(self, job_id: str, message: str) -> None:
        super().__init__(f"remote job {job_id} failed: {message}")
        self.job_id = job_id
        self.message = message


# ============================================================================
# The Transport protocol — what the Client depends on
# ============================================================================


class Transport(Protocol):
    """Abstract HTTP-ish surface. The Client only calls these four methods.

    Anything matching this shape can be swapped in — :class:`HttpxTransport`
    in production, :class:`InMemoryTransport` in tests, a recording
    transport for cassette-style debugging, etc.
    """

    def create_job(
        self,
        request: CreateJobRequest,
        *,
        idempotency_key: str | None = None,
    ) -> AttestationQuoteResponse:
        """POST /v1/jobs. Returns the job id and an attestation quote
        the client must verify before encrypting + uploading data.

        When ``idempotency_key`` is set, the server caches the response
        keyed by ``(org_id, idempotency_key)`` for 24h; a retry with the
        same key + body returns the same job_id + quote."""
        ...

    def upload_data(self, job_id: str, ciphertext: bytes) -> None:
        """PUT /v1/jobs/{id}/data. Hands the envelope-encrypted input
        ciphertext to the TEE. After this returns, the job moves from
        ``pending`` to ``running``."""
        ...

    def get_status(self, job_id: str) -> JobStatusResponse:
        """GET /v1/jobs/{id}. Polled until ``status`` is terminal."""
        ...

    def get_result(self, job_id: str) -> ResultResponse:
        """GET /v1/jobs/{id}/result. Returns the result ciphertext (only
        valid after status == 'completed')."""
        ...

    # ---- model management ---------------------------------------------------

    def list_models(self, *, limit: int = 50) -> ListModelsResponse:
        """GET /v1/models. Returns the caller's fitted models, most-recently-
        used first. Scope is per-API-key user."""
        ...

    def get_model(self, model_id: str) -> ModelInfoResponse:
        """GET /v1/models/{model_id}. Single model metadata. Raises
        :class:`JobNotFoundError` on 404 (also returned for models owned
        by a different user — no enumeration leak)."""
        ...

    def delete_model(self, model_id: str) -> None:
        """DELETE /v1/models/{model_id}. Idempotent — already-gone returns
        success."""
        ...

    # ---- account / billing / pre-flight ------------------------------------

    def health(self) -> HealthResponse:
        """GET /v1/health. Public — does not require auth. Returns server
        version + the minimum SDK version the server still serves."""
        ...

    def whoami(self) -> WhoAmIResponse:
        """GET /v1/me. Identity of the authenticated caller."""
        ...

    def credits(self) -> CreditsResponse:
        """GET /v1/credits. Current credit balance + tier."""
        ...

    def usage(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> UsageResponse:
        """GET /v1/usage. Flow-SDK-scoped usage history. Filters compose."""
        ...

    def usage_summary(self, *, period: str = "month") -> UsageSummaryResponse:
        """GET /v1/usage/summary. Aggregate usage over a window."""
        ...

    def estimate_cost(self, body: EstimateCostRequest) -> EstimateCostResponse:
        """POST /v1/jobs/estimate. Heuristic credit estimate for a job."""
        ...

    # ---- async-job-handle surface ------------------------------------------

    def list_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> ListJobsResponse:
        """GET /v1/jobs. List the caller's jobs, most-recent first. Use
        with :meth:`Client.list_jobs` to see what's in flight after firing
        off ``fit_async()`` / ``generate_async()`` / ``validate_async()``."""
        ...

    def cancel_job(self, job_id: str) -> None:
        """DELETE /v1/jobs/{id}. Cancels a queued or running job. Idempotent
        on already-terminal jobs (returns silently). The job's status will
        surface as ``failed`` with ``error_message='cancelled by user'``."""
        ...


# ============================================================================
# Real-world transport over httpx
# ============================================================================


@dataclass
class HttpxTransport:
    """Production transport — speaks HTTPS to the TEE service.

    Built on httpx so it works seamlessly with both sync + async code.
    Authenticates with the customer's API key as a bearer token; retries
    transient (5xx, network) errors with exponential backoff.
    """

    api_key: str
    endpoint: str = "https://flow.sablier.ai/v1"
    timeout_s: float = 60.0
    # Total wall-clock retry budget (per-call). Production fits are 15-25
    # minutes of polling against the worker's status endpoint. A 30s
    # budget was tight enough that a single ConnectTimeout (httpx's
    # default is 60s) could exhaust it in one attempt, killing the
    # customer's job even though the worker was still training fine.
    # 300s tolerates one full-minute network blip (and a couple of
    # backoff retries on top) without the deadline bursting. The total
    # job timeout is still capped by ``Client.poll_timeout_s`` (default
    # 30 min) so a genuinely dead connection still surfaces an error.
    retry_budget_s: float = 300.0
    # TLS verification override. Accepts:
    #   - None / True (default): verify against the system CA bundle
    #   - False: skip verification (DEMO/DEV ONLY — never in production)
    #   - str (path to a PEM file): pin a specific cert / CA bundle.
    # Used by self-hosted / staging deploys where the TEE is using a
    # self-signed cert until a real domain + Let's Encrypt cert is set up.
    verify: bool | str | None = None

    def _client(self) -> Any:
        """Build a fresh httpx.Client. Imported lazily so the [client]
        extra isn't pulled in if a customer only uses the local pipeline."""
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "httpx is required for HttpxTransport. "
                "Install with: pip install 'sablier-flow[client]'"
            ) from exc
        client_kwargs: dict[str, Any] = {
            "base_url": self.endpoint,
            "headers": {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "sablier-flow-sdk",
            },
            "timeout": self.timeout_s,
        }
        if self.verify is not None:
            client_kwargs["verify"] = self.verify
        return httpx.Client(**client_kwargs)

    def create_job(
        self,
        request: CreateJobRequest,
        *,
        idempotency_key: str | None = None,
    ) -> AttestationQuoteResponse:
        headers: dict[str, str] = {}
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        url = "/jobs"
        with self._client() as c:
            r = self._retry(
                lambda: c.post(url, json=request.model_dump(), headers=headers)
            )
            self._raise_for_status(r, url)
            return AttestationQuoteResponse.model_validate(self._safe_json(r, url))

    def upload_data(self, job_id: str, ciphertext: bytes) -> None:
        url = f"/jobs/{job_id}/data"
        with self._client() as c:
            r = self._retry(
                lambda: c.put(
                    url,
                    content=ciphertext,
                    headers={"Content-Type": "application/octet-stream"},
                )
            )
            # _raise_for_status maps 404 on /jobs/... to JobNotFoundError
            # and surfaces the server's `detail` verbatim.
            self._raise_for_status(r, url)

    def get_status(self, job_id: str) -> JobStatusResponse:
        url = f"/jobs/{job_id}"
        with self._client() as c:
            r = self._retry(lambda: c.get(url))
            self._raise_for_status(r, url)
            return JobStatusResponse.model_validate(self._safe_json(r, url))

    def get_result(self, job_id: str) -> ResultResponse:
        url = f"/jobs/{job_id}/result"
        with self._client() as c:
            r = self._retry(lambda: c.get(url))
            if r.status_code == 409:
                # Job exists but isn't completed — the server refuses
                # to return a result. Surface as RemoteJobError so
                # callers see the in-flight / failed semantics, not a
                # generic SablierClientError. The detail extraction
                # mirrors _raise_for_status so a missing/non-JSON body
                # still produces a sensible message.
                msg = self._extract_detail(r, fallback="job not in completed state")
                raise RemoteJobError(job_id, str(msg))
            self._raise_for_status(r, url)
            return ResultResponse.model_validate(self._safe_json(r, url))

    # ---- model management ---------------------------------------------------

    def list_models(self, *, limit: int = 50) -> ListModelsResponse:
        url = "/models"
        with self._client() as c:
            r = self._retry(lambda: c.get(url, params={"limit": int(limit)}))
            self._raise_for_status(r, url)
            return ListModelsResponse.model_validate(self._safe_json(r, url))

    def get_model(self, model_id: str) -> ModelInfoResponse:
        url = f"/models/{model_id}"
        with self._client() as c:
            r = self._retry(lambda: c.get(url))
            # _raise_for_status maps 404 on /models/... to ModelNotFoundError
            # (with the server's `detail` field surfaced verbatim). Client.py
            # catches that and rewraps as ValueError for the public API.
            self._raise_for_status(r, url)
            return ModelInfoResponse.model_validate(self._safe_json(r, url))

    def delete_model(self, model_id: str) -> None:
        url = f"/models/{model_id}"
        with self._client() as c:
            r = self._retry(lambda: c.delete(url))
            # 204 (deleted) or 404 (already gone) both treated as success
            if r.status_code in (204, 404):
                return
            self._raise_for_status(r, url)

    # ---- account / billing / pre-flight ------------------------------------

    def health(self) -> HealthResponse:
        url = "/health"
        with self._client() as c:
            r = self._retry(lambda: c.get(url))
            self._raise_for_status(r, url)
            return HealthResponse.model_validate(self._safe_json(r, url))

    def whoami(self) -> WhoAmIResponse:
        url = "/me"
        with self._client() as c:
            r = self._retry(lambda: c.get(url))
            self._raise_for_status(r, url)
            return WhoAmIResponse.model_validate(self._safe_json(r, url))

    def credits(self) -> CreditsResponse:
        url = "/credits"
        with self._client() as c:
            r = self._retry(lambda: c.get(url))
            self._raise_for_status(r, url)
            return CreditsResponse.model_validate(self._safe_json(r, url))

    def usage(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> UsageResponse:
        params: dict[str, Any] = {"limit": int(limit)}
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if kind:
            params["kind"] = kind
        url = "/usage"
        with self._client() as c:
            r = self._retry(lambda: c.get(url, params=params))
            self._raise_for_status(r, url)
            return UsageResponse.model_validate(self._safe_json(r, url))

    def usage_summary(self, *, period: str = "month") -> UsageSummaryResponse:
        url = "/usage/summary"
        with self._client() as c:
            r = self._retry(lambda: c.get(url, params={"period": period}))
            self._raise_for_status(r, url)
            return UsageSummaryResponse.model_validate(self._safe_json(r, url))

    def estimate_cost(self, body: EstimateCostRequest) -> EstimateCostResponse:
        url = "/jobs/estimate"
        with self._client() as c:
            r = self._retry(
                lambda: c.post(url, json=body.model_dump(exclude_none=True))
            )
            self._raise_for_status(r, url)
            return EstimateCostResponse.model_validate(self._safe_json(r, url))

    # ---- async-job-handle surface ------------------------------------------

    def list_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> ListJobsResponse:
        params: dict[str, Any] = {"limit": int(limit)}
        if status:
            params["status"] = status
        url = "/jobs"
        with self._client() as c:
            r = self._retry(lambda: c.get(url, params=params))
            self._raise_for_status(r, url)
            return ListJobsResponse.model_validate(self._safe_json(r, url))

    def cancel_job(self, job_id: str) -> None:
        url = f"/jobs/{job_id}"
        with self._client() as c:
            r = self._retry(lambda: c.delete(url))
            # 204 (cancelled), 404 (gone), 409 (already terminal) all treated
            # as "you don't need to cancel anymore" — idempotent surface.
            if r.status_code in (204, 404, 409):
                return
            self._raise_for_status(r, url)

    # Backoff schedule constants (module-level-style, kept on the class so
    # tests can monkey-patch ``HttpxTransport._SLEEP`` to a no-op).
    _BACKOFF_BASE_S: float = 0.5
    _BACKOFF_CAP_S: float = 30.0
    _RETRY_AFTER_CAP_S: float = 60.0

    @staticmethod
    def _SLEEP(seconds: float) -> None:  # intentional UPPER for monkey-patching
        """Indirection over :func:`time.sleep` so tests can null it out."""
        time.sleep(seconds)

    @classmethod
    def _parse_retry_after(cls, value: str | None) -> float | None:
        """Parse a ``Retry-After`` header value into seconds.

        RFC 7231 allows either an integer number of seconds OR an
        HTTP-date. Returns ``None`` if absent or unparseable. The result
        is clamped to ``_RETRY_AFTER_CAP_S`` to keep a buggy server from
        wedging the SDK for hours.
        """
        if not value:
            return None
        value = value.strip()
        # Delta-seconds form
        try:
            secs = float(value)
        except ValueError:
            secs = None  # type: ignore[assignment]
        if secs is None:
            # HTTP-date form
            try:
                target = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            if target is None:
                return None
            import datetime as _dt

            now = _dt.datetime.now(tz=target.tzinfo) if target.tzinfo else _dt.datetime.utcnow()
            secs = max(0.0, (target - now).total_seconds())
        # Clamp to a sane upper bound
        return max(0.0, min(float(secs), cls._RETRY_AFTER_CAP_S))

    def _retry(self, op: Any) -> Any:
        """Backoff on transient errors. Caller passes a zero-arg callable
        returning an httpx.Response.

        Retry triggers:
          * network/timeout exceptions raised by httpx
          * HTTP 5xx responses
          * HTTP 429 Too Many Requests
          * HTTP 503 Service Unavailable (also a 5xx, called out for clarity)

        Backoff:
          * If the response carries a ``Retry-After`` header (delta-seconds
            or HTTP-date), wait exactly that long (clamped to 60s).
          * Otherwise, exponential backoff with jitter:
            ``sleep = base * 2^attempt * uniform(0.5, 1.5)``, capped at 30s.

        Terminal:
          * HTTP 401 -> raise :class:`AuthenticationError` immediately (no retry).
          * Any other non-retryable status -> return the response so the
            caller can ``raise_for_status()`` / handle 404/409/etc.
        """
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ImportError("httpx is required") from exc

        deadline = time.monotonic() + self.retry_budget_s
        attempt = 0
        last_exc: Exception | None = None
        while True:
            retry_after_s: float | None = None
            try:
                resp = op()
                # 401 is terminal — never retry a revoked / bad API key.
                if resp.status_code == 401:
                    raise self._auth_error_from(resp)
                # 429 + 503 + any other 5xx are transient.
                if resp.status_code == 429 or resp.status_code >= 500:
                    retry_after_s = self._parse_retry_after(
                        resp.headers.get("Retry-After")
                    )
                    last_exc = None
                else:
                    return resp
            except AuthenticationError:
                raise
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc

            # Decide whether we can afford another attempt.
            if retry_after_s is not None:
                sleep_s = retry_after_s
            else:
                # Jittered exponential backoff. ``uniform(0.5, 1.5)`` gives
                # +/-50% jitter, which is the conventional "decorrelated"
                # spread for transient-retry storms.
                sleep_s = min(
                    self._BACKOFF_BASE_S * (2 ** attempt) * random.uniform(0.5, 1.5),
                    self._BACKOFF_CAP_S,
                )
            if time.monotonic() + sleep_s > deadline:
                if last_exc is not None:
                    raise TransportError(
                        f"retry budget exhausted (last error: {last_exc!r})"
                    ) from last_exc
                raise TransportError("retry budget exhausted on transient responses")
            self._SLEEP(sleep_s)
            attempt += 1

    @staticmethod
    def _auth_error_from(resp: Any) -> AuthenticationError:
        """Build an :class:`AuthenticationError` from a 401 response,
        pulling the ``detail`` field out of the body when it's JSON."""
        detail: str = "API key rejected (HTTP 401)"
        try:
            body = resp.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            for key in ("detail", "message", "error"):
                val = body.get(key)
                if isinstance(val, str) and val:
                    detail = val
                    break
        return AuthenticationError(detail)

    @staticmethod
    def _extract_detail(resp: Any, fallback: str) -> str:
        """Pull the server's ``detail`` string out of a 4xx response body.

        Every server-side 4xx ships a JSON body shaped like
        ``{"detail": "..."}`` (FastAPI's default). Older or stranger
        servers may use ``message`` or ``error``; we accept those too.
        Falls back to ``fallback`` when the body isn't JSON or doesn't
        carry any known field, so the customer still gets *something*
        descriptive in the exception message.
        """
        try:
            body = resp.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            for key in ("detail", "message", "error"):
                val = body.get(key)
                if isinstance(val, str) and val:
                    return val
                # FastAPI's pydantic-validation 422 puts a list of dicts
                # under ``detail`` — flatten the first ``msg`` for the
                # customer rather than dumping the raw structure.
                if isinstance(val, list) and val:
                    first = val[0]
                    if isinstance(first, dict):
                        msg = first.get("msg")
                        if isinstance(msg, str) and msg:
                            return msg
        return fallback

    @classmethod
    def _raise_for_status(cls, resp: Any, url: str) -> None:
        """Translate a non-2xx response into a typed SDK exception.

        Replaces every bare ``r.raise_for_status()`` site. The exception
        hierarchy mirrors the server's error surface:

          * 2xx                       — return (no-op)
          * 401                       — :class:`AuthenticationError`
          * 404 on a ``/jobs/`` path  — :class:`JobNotFoundError`
          * 404 on a ``/models/`` path — :class:`ModelNotFoundError`
          * 404 otherwise            — :class:`SablierClientError`
          * any other 4xx            — :class:`SablierClientError`
          * 5xx / 429                — left to :meth:`_retry` (this
            helper is invoked AFTER ``_retry`` returns, so by the time
            we see a 5xx the retry budget is already exhausted; we still
            surface it as a generic ``SablierClientError`` rather than
            leaking ``httpx.HTTPStatusError``).
        """
        status = resp.status_code
        if 200 <= status < 300:
            return
        if status == 401:
            raise cls._auth_error_from(resp)
        if status == 404:
            # Look at the URL to pick the more specific exception. The
            # ``url`` arg the caller passes here is the path the SDK hit
            # (e.g. ``/jobs/abc123``); we sniff the segment, not the
            # whole URL, so the same logic works for absolute and
            # relative forms.
            path = str(url)
            detail = cls._extract_detail(resp, fallback="not found")
            lowered = detail.lower()
            if "/models" in path or path.startswith("models") or (
                status == 404 and "model" in lowered
            ):
                raise ModelNotFoundError(status, detail, path)
            if "/jobs" in path or path.startswith("jobs"):
                raise JobNotFoundError(status, detail, path)
            raise SablierClientError(status, detail, path)
        if 400 <= status < 500:
            detail = cls._extract_detail(
                resp, fallback=f"client error (HTTP {status})"
            )
            raise SablierClientError(status, detail, str(url))
        # 5xx fell through retry. Surface as a typed client-side error
        # carrying the server's detail rather than leaking httpx exceptions.
        detail = cls._extract_detail(
            resp, fallback=f"server error (HTTP {status})"
        )
        raise SablierClientError(status, detail, str(url))

    @staticmethod
    def _safe_json(resp: Any, url: str) -> Any:
        """Decode ``resp.json()`` and re-raise JSONDecodeError as
        :class:`TransportError`.

        A 200 response with an HTML body (e.g. a misrouted load-balancer
        intercept page) used to leak raw :class:`json.JSONDecodeError`
        out of the SDK. Wrapping it gives callers something they can
        ``except TransportError`` cleanly.
        """
        try:
            return resp.json()
        except Exception as exc:  # JSONDecodeError, ValueError, etc.
            raise TransportError(
                f"server returned non-JSON body on {url}: {exc!r}"
            ) from exc


# ============================================================================
# In-process fake — tests + dev
# ============================================================================


def _fingerprint(request: CreateJobRequest) -> str:
    """Stable hash of a CreateJobRequest — used for in-process
    idempotency collision detection in InMemoryTransport."""
    import hashlib

    return hashlib.sha256(request.model_dump_json().encode("utf-8")).hexdigest()


@dataclass
class _FakeJob:
    job_id: str
    kind: JobKind
    params: dict[str, Any]
    ephemeral_pubkey: bytes
    status: JobStatus = "pending"
    input_ciphertext: bytes | None = None
    result_ciphertext: bytes | None = None
    error_message: str | None = None


class InMemoryTransport:
    """In-process fake transport for tests + local dev.

    Drop in instead of :class:`HttpxTransport` to exercise the full
    Client flow with no server running. The caller wires up:

      - a callable that *runs the job* (mock TEE), producing the result
        ciphertext (and possibly the attestation quote)
      - the TEE's per-boot ephemeral keypair (so the fake can produce a
        quote the client verifier accepts)

    The fake keeps each job in a dict and returns the status the caller
    set on it. See ``tests/integration/test_client_end_to_end.py`` for
    the canonical wiring.
    """

    def __init__(
        self,
        *,
        run_job: Any,
        quote_generator: Any,
        tee_keys: Any,
    ) -> None:
        """
        Parameters
        ----------
        run_job
            Callable ``(_FakeJob) -> bytes`` that "runs the job" inside
            the fake TEE and returns the encrypted result. Invoked
            synchronously the moment data is uploaded.
        quote_generator
            Callable ``(ephemeral_pubkey_bytes) -> bytes`` matching
            :func:`server.tee.attestation.generate_attestation_quote`.
        tee_keys
            Object with a ``public_key_bytes`` attribute (e.g.
            :class:`server.tee.crypto.TEEKeyState`).
        """
        self._run_job = run_job
        self._quote_generator = quote_generator
        self._tee_keys = tee_keys
        self._jobs: dict[str, _FakeJob] = {}
        self._counter = 0
        self._idempotency: dict[tuple[str, str], AttestationQuoteResponse] = {}

    def create_job(
        self,
        request: CreateJobRequest,
        *,
        idempotency_key: str | None = None,
    ) -> AttestationQuoteResponse:
        # The in-memory fake honors idempotency keys at the
        # (request_fingerprint, key) granularity so tests can exercise
        # the full Client retry path without spinning up a real server.
        if idempotency_key is not None:
            cache_key = (idempotency_key, _fingerprint(request))
            cached = self._idempotency.get(cache_key)
            if cached is not None:
                return cached

        self._counter += 1
        job_id = f"job-{self._counter:06d}"
        pubkey = self._tee_keys.public_key_bytes
        job = _FakeJob(
            job_id=job_id,
            kind=request.kind,
            params=dict(request.params),
            ephemeral_pubkey=pubkey,
        )
        self._jobs[job_id] = job
        quote = self._quote_generator(pubkey)
        resp = AttestationQuoteResponse.from_bytes_quote(job_id=job_id, quote_bytes=quote)
        if idempotency_key is not None:
            cache_key = (idempotency_key, _fingerprint(request))
            self._idempotency[cache_key] = resp
        return resp

    def upload_data(self, job_id: str, ciphertext: bytes) -> None:
        job = self._lookup(job_id)
        job.input_ciphertext = ciphertext
        job.status = "running"
        try:
            job.result_ciphertext = self._run_job(job)
            job.status = "completed"
        except Exception as exc:
            job.status = "failed"
            job.error_message = type(exc).__name__

    def get_status(self, job_id: str) -> JobStatusResponse:
        job = self._lookup(job_id)
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            error_message=job.error_message,
        )

    def get_result(self, job_id: str) -> ResultResponse:
        job = self._lookup(job_id)
        if job.status != "completed" or job.result_ciphertext is None:
            raise RemoteJobError(job_id, job.error_message or "no result")
        return ResultResponse.from_bytes_ciphertext(
            job_id=job_id, ciphertext_bytes=job.result_ciphertext
        )

    # ---- model management (in-memory: no real model registry) ---------------
    # InMemoryTransport's model surface is intentionally empty so tests
    # that don't exercise it pass; tests that DO can patch these methods.

    def list_models(self, *, limit: int = 50) -> ListModelsResponse:
        return ListModelsResponse(models=[], total_returned=0)

    def get_model(self, model_id: str) -> ModelInfoResponse:
        raise ModelNotFoundError(
            404, f"model {model_id} not found", f"/models/{model_id}"
        )

    def delete_model(self, model_id: str) -> None:
        return None

    # ---- account / billing / pre-flight (in-memory) ------------------------

    def health(self) -> HealthResponse:
        return HealthResponse(server_version="in-memory", min_sdk_version="0.0.0")

    def whoami(self) -> WhoAmIResponse:
        return WhoAmIResponse(user_id="in-memory-user", tier="free")

    def credits(self) -> CreditsResponse:
        return CreditsResponse(
            available=10**9, monthly_allocation=10**9, monthly_used=0,
            purchased=0, tier="free",
        )

    def usage(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> UsageResponse:
        return UsageResponse(items=[], total_returned=0, total_credits=0.0)

    def usage_summary(self, *, period: str = "month") -> UsageSummaryResponse:
        from datetime import datetime as _dt
        now = _dt.utcnow().isoformat()
        return UsageSummaryResponse(
            period_start=now, period_end=now, total_credits=0.0, by_kind={},
        )

    def estimate_cost(self, body: EstimateCostRequest) -> EstimateCostResponse:
        return EstimateCostResponse(
            estimated_credits=0.0, low=0.0, high=0.0,
            estimated_duration_s=0.0, notes=["in-memory transport"],
        )

    # ---- async-job-handle surface (in-memory) ------------------------------

    def list_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> ListJobsResponse:
        # Most-recent-first means iteration order in the dict reversed, since
        # _FakeJob isn't timestamped — close enough for tests.
        summaries = []
        for job in reversed(list(self._jobs.values())):
            if status is not None and job.status != status:
                continue
            summaries.append(
                JobSummary(
                    job_id=job.job_id,
                    kind=job.kind,
                    status=job.status,
                    # No real clock in the fake — emit a stable sentinel so
                    # the wire model validates and tests can assert on it.
                    created_at="1970-01-01T00:00:00Z",
                    completed_at=None,
                    credits_charged=None,
                    error_message=job.error_message,
                )
            )
            if len(summaries) >= int(limit):
                break
        return ListJobsResponse(jobs=summaries, total_returned=len(summaries))

    def cancel_job(self, job_id: str) -> None:
        job = self._lookup(job_id)
        if job.status in ("completed", "failed"):
            return  # already terminal — idempotent
        job.status = "failed"
        job.error_message = "cancelled by user"

    def _lookup(self, job_id: str) -> _FakeJob:
        if job_id not in self._jobs:
            raise JobNotFoundError(
                404, f"job {job_id} not found", f"/jobs/{job_id}"
            )
        return self._jobs[job_id]

    def __iter__(self) -> Iterator[_FakeJob]:
        """Allow tests to enumerate jobs created so far for assertions."""
        return iter(self._jobs.values())
