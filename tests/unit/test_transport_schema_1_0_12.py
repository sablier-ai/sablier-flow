"""Schema regression tests for the 1.0.12 fix.

Background: starting with the async worker refactor, the backend's
``GET /v1/jobs/{id}`` endpoint started returning ``progress`` as a
structured heartbeat dict (e.g. ``{"step": 30, "phase": "training"}``)
rather than the original human-readable string. SDK 1.0.11 still typed
``JobStatusResponse.progress`` as ``Optional[str]``, which caused
``sf.fetch_result()`` and ``sf.list_jobs()`` to raise
``pydantic.ValidationError`` on the convenience wrappers — even though
the job itself completed successfully.

These tests pin the widened schema so the regression can't sneak back:

  * ``progress`` accepts a heartbeat ``dict``
  * ``progress`` accepts ``None`` (no heartbeat yet)
  * ``progress`` still accepts the legacy ``str`` shape (backwards-compat
    for any caller — including older self-hosted servers — that still
    emits the string form)
  * ``last_progress_at`` accepts an ISO-8601 datetime string
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sablier_flow.client.transport import JobStatusResponse


def test_JobStatusResponse_accepts_dict_progress() -> None:
    """The async worker emits structured heartbeats; the client model
    must deserialize them without raising."""
    model = JobStatusResponse(
        job_id="job_abc123",
        status="running",
        progress={"step": 30, "phase": "training"},
    )
    assert model.progress == {"step": 30, "phase": "training"}
    assert model.status == "running"


def test_JobStatusResponse_accepts_null_progress() -> None:
    """Before the first heartbeat lands, ``progress`` is ``None``."""
    model = JobStatusResponse(
        job_id="job_abc123",
        status="pending",
        progress=None,
    )
    assert model.progress is None


def test_JobStatusResponse_accepts_null_progress_when_omitted() -> None:
    """Omitting ``progress`` entirely is equivalent to ``None`` — older
    servers that don't ship the field at all must still deserialize."""
    model = JobStatusResponse(job_id="job_abc123", status="pending")
    assert model.progress is None
    assert model.last_progress_at is None


def test_JobStatusResponse_still_accepts_str_progress() -> None:
    """Backwards-compat: any caller still sending the legacy string form
    (e.g. older self-hosted server, hand-rolled fixture) must keep
    working — no breaking change for existing integrations."""
    model = JobStatusResponse(
        job_id="job_abc123",
        status="running",
        progress="epoch 12/50",
    )
    assert model.progress == "epoch 12/50"


def test_JobStatusResponse_accepts_last_progress_at_iso_string() -> None:
    """``last_progress_at`` is wire-shaped as an ISO-8601 datetime string
    (or ``None``); make sure the client accepts the string form."""
    model = JobStatusResponse(
        job_id="job_abc123",
        status="running",
        progress={"step": 30, "phase": "training"},
        last_progress_at="2026-05-31T12:34:56.789Z",
    )
    assert model.last_progress_at == "2026-05-31T12:34:56.789Z"


def test_JobStatusResponse_accepts_last_progress_at_none() -> None:
    """``last_progress_at`` is also nullable (no heartbeat yet)."""
    model = JobStatusResponse(
        job_id="job_abc123",
        status="pending",
        last_progress_at=None,
    )
    assert model.last_progress_at is None


def test_JobStatusResponse_rejects_invalid_progress_type() -> None:
    """Sanity: integers and other unrelated types should still be
    rejected — we widened to ``dict | str | None``, not ``Any``."""
    with pytest.raises(ValidationError):
        JobStatusResponse(
            job_id="job_abc123",
            status="running",
            progress=12345,  # type: ignore[arg-type]
        )
