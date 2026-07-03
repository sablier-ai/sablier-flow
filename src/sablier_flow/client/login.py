"""``sablier_flow.login()`` / ``logout()`` — OAuth-style device flow.

Lets a customer in a Jupyter notebook or SSH'd box authenticate without
copy-pasting an API key from the web dashboard:

    >>> import sablier_flow as sf
    >>> sf.login()                                  # opens browser, prints code
    On any device, open https://sablier.ai/auth/device
    and enter the code: ABCD-EFGH
    Waiting for approval...
    Logged in as team@sablier.ai.

    >>> client = sf.Client()                         # auto-picks up stored key

The flow is RFC 8628 Device Authorization Grant adapted to mint a
sablier API key (rather than an OAuth bearer). State lives in
``~/.sablier/credentials`` (a plain JSON file with mode 0600):

    {
        "default": {
            "api_key": "sk_live_...",
            "key_id": "...",
            "key_prefix": "sk_live_xxxx",
            "endpoint": "https://flow.sablier.ai/v1",
            "logged_in_at": "2026-05-29T12:34:56Z"
        }
    }

``sf.Client()`` (with no api_key kwarg) reads this file; the env var
``SABLIER_FLOW_API_KEY`` still takes precedence so CI can keep its
existing pattern.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import re
import socket
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_ENDPOINT = "https://flow.sablier.ai/v1"
DEFAULT_PROFILE = "default"

_LOG = logging.getLogger(__name__)

# Allowed hosts for a stored endpoint. We intentionally restrict to the
# sablier.ai apex (and any subdomain) plus the canonical Cloud Run service
# hostname pattern. Anything else — including bare http:// — is rejected so
# a tampered-with credentials file can't redirect the SDK to an attacker
# endpoint that would happily collect the API key on the next request.
_ALLOWED_HOST_EXACT = {"sablier.ai"}
_ALLOWED_HOST_SUFFIX = (".sablier.ai",)
# Cloud Run serves the same service under TWO hostname forms:
#   * canonical long form:  sablier[-flow]-api-<hash>.<region>.run.app
#   * short alias:          sablier[-flow]-api-<hash>-<region>.a.run.app
# Both are valid and routable — the short form is what Cloud Run
# actually shows in the console and what our deploy scripts use as the
# canonical prod URL (see scripts/e2e_smoke.py). Both must be on the
# allowlist or a credentials file pointing at the canonical prod URL
# silently falls back to DEFAULT_ENDPOINT on next load. Hash is hex
# (no dashes) on the short form — that's the Cloud Run convention.
# The optional `flow-` segment covers the backend-v2 service name
# (`sablier-flow-api`) alongside the legacy monolith (`sablier-api`).
_ALLOWED_HOST_REGEX = (
    re.compile(r"^sablier-(flow-)?api-[a-z0-9-]+\.us-central1\.run\.app$"),
    re.compile(r"^sablier-(flow-)?api-[a-z0-9]+-uc\.a\.run\.app$"),
)


def validate_stored_endpoint(url: str | None) -> str | None:
    """Return ``url`` if it's a safe Sablier endpoint, else ``None``.

    Used as a tripwire on both read and write of the credentials file —
    a stored endpoint is trusted by ``Client()`` to send the API key to,
    so we constrain it to:

      * scheme MUST be ``https://`` (no plaintext, no ``file://``, etc.)
      * host MUST be ``sablier.ai``, ``*.sablier.ai``, the Cloud Run
        long-form hostname ``sablier-api-*.<region>.run.app``,
        or its short alias ``sablier-api-*-<region>.a.run.app``

    On reject we log a clear warning and return ``None`` rather than
    raising — callers fall back to the default endpoint, which is the
    safe behavior. Raising would break ``Client()`` construction for a
    user who only wanted to use the api_key portion of the file."""
    if not url or not isinstance(url, str):
        return None
    try:
        parsed = urlparse(url)
    except Exception:
        _LOG.warning("ignoring stored endpoint %r: not a parseable URL", url)
        return None
    if parsed.scheme != "https":
        _LOG.warning(
            "ignoring stored endpoint %r: only https:// is accepted (got scheme %r)",
            url,
            parsed.scheme,
        )
        return None
    host = (parsed.hostname or "").lower()
    if not host:
        _LOG.warning("ignoring stored endpoint %r: missing host", url)
        return None
    if host in _ALLOWED_HOST_EXACT:
        return url
    if any(host.endswith(suffix) for suffix in _ALLOWED_HOST_SUFFIX):
        return url
    if any(rx.match(host) for rx in _ALLOWED_HOST_REGEX):
        return url
    _LOG.warning(
        "ignoring stored endpoint %r: host %r is not an allowed Sablier endpoint",
        url,
        host,
    )
    return None


def _credentials_path() -> Path:
    """``~/.sablier/credentials`` — overrideable via SABLIER_FLOW_CREDENTIALS
    for tests + CI."""
    override = os.environ.get("SABLIER_FLOW_CREDENTIALS")
    if override:
        return Path(override)
    return Path.home() / ".sablier" / "credentials"


def _client_name() -> str:
    """Cosmetic label shown on the approval page so the user can see
    what they're authorizing. We surface the hostname + Python version
    so a quant approving a login from a remote box knows it's their
    actual session, not some other process."""
    host = socket.gethostname()
    py = f"py{sys.version_info.major}.{sys.version_info.minor}"
    plat = platform.system().lower()
    return f"sablier-flow on {host} ({plat}, {py})"


def _read_credentials_blob() -> dict[str, dict[str, Any]]:
    path = _credentials_path()
    if not path.exists():
        return {}
    try:
        with path.open("r") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_credentials_blob(data: dict[str, dict[str, Any]]) -> None:
    path = _credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically: temp + rename so a half-written file can never
    # leave the loader confused. mode 0600 because the api_key lives
    # here in cleartext.
    #
    # 1.0.21 — open the tmp file with O_CREAT | O_EXCL and mode 0o600
    # BEFORE any bytes are written. The previous flow created the file
    # at default umask (typically 0o644 = world-readable) and chmod'd
    # it AFTER the json.dump, so the cleartext api_key was readable by
    # any local user for the duration of the write window. The
    # tmp-then-rename atomicity guarantee is preserved.
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Remove any leftover tmp from a previous crashed write so the
    # O_EXCL doesn't trip.
    with contextlib.suppress(FileNotFoundError):
        os.unlink(tmp)
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
    except Exception:
        # If json.dump raises mid-write, leave the empty tmp file
        # behind so the next attempt's O_EXCL doesn't silently overwrite
        # a partial write that might somehow still be useful for debugging.
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    tmp.replace(path)


def load_credentials(profile: str = DEFAULT_PROFILE) -> dict[str, Any] | None:
    """Read the saved credentials for ``profile``. Used by
    :class:`sablier_flow.Client` to pick up the post-login key.

    Returns ``None`` if no credentials are stored (the SDK falls back
    to the ``SABLIER_FLOW_API_KEY`` env var, then errors).

    Any stored ``endpoint`` is run through :func:`validate_stored_endpoint`;
    if it fails (non-https, off-domain, malformed) we strip it to ``None``
    so the caller falls back to the default endpoint instead of silently
    leaking the api_key to whatever URL a tampered-with credentials file
    might point at."""
    blob = _read_credentials_blob()
    entry = blob.get(profile)
    if not entry:
        return None
    entry = dict(entry)
    if "endpoint" in entry:
        entry["endpoint"] = validate_stored_endpoint(entry.get("endpoint"))
    return entry


def save_credentials(
    *,
    api_key: str,
    key_id: str | None = None,
    key_prefix: str | None = None,
    endpoint: str | None = None,
    profile: str = DEFAULT_PROFILE,
) -> None:
    """Write a credentials entry. Called by :func:`login`; exposed
    for advanced workflows (e.g. wiring up a service account).

    The ``endpoint`` is validated via :func:`validate_stored_endpoint`
    before it lands on disk — we don't want to write a bad endpoint that
    we'd then refuse to read back. If a caller passes an invalid endpoint
    we fall back to :data:`DEFAULT_ENDPOINT` and log a warning so the
    write still succeeds (the api_key is the load-bearing part)."""
    from datetime import datetime, timezone
    candidate_endpoint = endpoint or DEFAULT_ENDPOINT
    safe_endpoint = validate_stored_endpoint(candidate_endpoint)
    if safe_endpoint is None:
        _LOG.warning(
            "refusing to write stored endpoint %r; using default %r instead",
            candidate_endpoint,
            DEFAULT_ENDPOINT,
        )
        safe_endpoint = DEFAULT_ENDPOINT
    blob = _read_credentials_blob()
    blob[profile] = {
        "api_key": api_key,
        "key_id": key_id,
        "key_prefix": key_prefix,
        "endpoint": safe_endpoint,
        "logged_in_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_credentials_blob(blob)


def clear_credentials(profile: str = DEFAULT_PROFILE) -> bool:
    """Drop the credentials entry for ``profile``. Returns True if
    something was removed, False if the entry didn't exist."""
    blob = _read_credentials_blob()
    if profile not in blob:
        return False
    blob.pop(profile)
    if blob:
        _write_credentials_blob(blob)
    else:
        # Last entry — remove the file entirely.
        import contextlib
        with contextlib.suppress(FileNotFoundError):
            _credentials_path().unlink()
    return True


# ============================================================================
# login() / logout()
# ============================================================================


@dataclass(frozen=True)
class LoginResult:
    """Returned by :func:`login` — what got stored on disk.

    Custom ``__repr__`` truncates ``api_key`` to the canonical 12-char
    prefix so ``r = sf.login(); r`` in a notebook does NOT persist the
    full ``sk_live_`` secret into the ``.ipynb`` on disk (1.0.21
    follow-up to the print-side truncation already in :func:`login`).
    """

    api_key: str
    key_id: str | None
    key_prefix: str | None
    endpoint: str
    profile: str

    def __repr__(self) -> str:
        # Mirror the 12-char truncation used in the login() print line;
        # never let the full api_key out via repr / logger.info('%r', ...) /
        # traceback frame-locals / Jupyter cell-output persistence.
        safe = (self.api_key or "")[:12] + "..."
        return (
            f"LoginResult(api_key={safe!r}, key_id={self.key_id!r}, "
            f"key_prefix={self.key_prefix!r}, endpoint={self.endpoint!r}, "
            f"profile={self.profile!r})"
        )


def login(
    *,
    endpoint: str | None = None,
    profile: str = DEFAULT_PROFILE,
    open_browser: bool = True,
    poll_timeout_s: float = 600.0,
    verify: bool | str | None = None,
) -> LoginResult:
    """Run the device-auth flow and save the resulting API key.

    Prints the user_code + verification URL, optionally opens the
    browser, then polls the backend until the user approves on any
    device they're signed in on. On success, writes the api_key to
    ``~/.sablier/credentials`` and returns a :class:`LoginResult`.

    Raises :class:`TimeoutError` if ``poll_timeout_s`` elapses with no
    approval. Raises :class:`RuntimeError` on backend-side denial /
    expiry."""
    endpoint = endpoint or os.environ.get("SABLIER_FLOW_ENDPOINT") or DEFAULT_ENDPOINT
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "httpx is required for login(). Install with: pip install sablier-flow"
        ) from exc

    client_kwargs: dict[str, Any] = {
        "base_url": endpoint.rstrip("/"),
        "timeout": 30.0,
        "headers": {"User-Agent": "sablier-flow-sdk"},
    }
    if verify is not None:
        client_kwargs["verify"] = verify

    with httpx.Client(**client_kwargs) as c:
        # 1. Open the device authorization.
        start = c.post(
            "/auth/device/start",
            json={"client_name": _client_name()},
        )
        start.raise_for_status()
        start_data = start.json()
        device_code = start_data["device_code"]
        user_code = start_data["user_code"]
        verification_uri = start_data["verification_uri"]
        verification_uri_complete = start_data.get(
            "verification_uri_complete", verification_uri,
        )
        interval = float(start_data.get("interval", 5))
        expires_in = float(start_data.get("expires_in", poll_timeout_s))

        # 2. Tell the user. The block formatting puts the code in the
        # easiest place for a copy-paste; we also print the URL twice so
        # the SSH terminal user can manually open it.
        print()
        print("To authenticate, open this URL on any device where you're signed in:")
        print(f"    {verification_uri}")
        print()
        print("and enter this code:")
        print(f"    {user_code}")
        print()
        print(f"(Or open the pre-filled link: {verification_uri_complete})")
        print()
        if open_browser:
            import contextlib
            with contextlib.suppress(Exception):
                webbrowser.open(verification_uri_complete, new=2)

        # 3. Poll. RFC 8628 says respect the interval but back off on
        # 'slow_down'. We just hold the constant interval — backend
        # caps polling rate-limit at 60/min which is well above 5s.
        deadline = time.monotonic() + min(poll_timeout_s, expires_in)
        print("Waiting for approval...", flush=True)
        while True:
            time.sleep(interval)
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"login timed out after {poll_timeout_s:.0f}s — "
                    f"no approval received for code {user_code}"
                )
            poll = c.post("/auth/device/token", json={"device_code": device_code})
            if poll.status_code == 200:
                token_data = poll.json()
                break
            # OAuth-style error body — decode best-effort.
            try:
                detail = poll.json().get("detail", "")
            except Exception:
                detail = poll.text
            if detail == "authorization_pending":
                continue
            if detail == "slow_down":
                interval *= 1.5
                continue
            if detail == "access_denied":
                raise RuntimeError("login denied by user — aborting")
            if detail == "expired_token":
                raise RuntimeError(
                    "the user_code expired before approval — please retry sf.login()"
                )
            # Any other 4xx / 5xx — surface to the caller.
            poll.raise_for_status()

    api_key = token_data["api_key"]
    key_id = token_data.get("key_id")
    key_prefix = token_data.get("key_prefix")
    # approver_email arrives on the device approve response when the
    # backend ships it; if it's missing we fall back to a whoami() lookup
    # so the field is always populated for the audit echo.
    approver_email = token_data.get("approver_email")
    if not approver_email:
        try:
            with httpx.Client(**client_kwargs) as c2:
                resp = c2.get(
                    "/whoami",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code == 200:
                    body = resp.json()
                    approver_email = body.get("email") or body.get("user_email")
        except Exception:
            approver_email = None
    # Anti-phishing ordering: PRINT before PERSIST. If anything between
    # the print() calls and save_credentials() trips — Ctrl-C, full disk,
    # PermissionError on ~/.sablier — the user MUST have seen which
    # account approved BEFORE the api_key lands on disk. Otherwise a
    # stored key with no terminal echo of the approver is exactly the
    # phishing surface we're trying to close (user thinks "did that even
    # work?", retries with a different account, ends up with a key bound
    # to an account they never saw confirmed). Flush stdout explicitly
    # so a terminal that's line-buffering still shows the line.
    if approver_email:
        print(f"Logged in as {approver_email}.")
    # Defensive truncation: we trust the server to ship a short
    # `key_prefix` (~12 chars), but if a server bug or older deploy ever
    # ships the full secret here, we MUST NOT leak it to terminal
    # scrollback. Truncate to the canonical 12 chars regardless of what
    # the server sent.
    safe_prefix = (key_prefix or "sk_live_")[:12]
    print(f"API key prefix: {safe_prefix}...")
    print(f"Endpoint: {endpoint}")
    if endpoint != DEFAULT_ENDPOINT:
        print(
            f"WARNING: non-default endpoint in use ({endpoint}); "
            f"default is {DEFAULT_ENDPOINT}.",
        )
    sys.stdout.flush()
    save_credentials(
        api_key=api_key,
        key_id=key_id,
        key_prefix=key_prefix,
        endpoint=endpoint,
        profile=profile,
    )
    return LoginResult(
        api_key=api_key,
        key_id=key_id,
        key_prefix=key_prefix,
        endpoint=endpoint,
        profile=profile,
    )


def logout(*, profile: str = DEFAULT_PROFILE) -> bool:
    """Remove the saved credentials for ``profile``. Returns True if a
    credential was removed, False if there was nothing to remove.

    Local-only — does NOT revoke the API key on the server. Use the web
    dashboard to revoke if you suspect the key is leaked. We don't
    surface a revoke endpoint from here because that would require the
    SDK to be authenticated to call its own deletion, which loops back
    to the device flow."""
    return clear_credentials(profile=profile)
