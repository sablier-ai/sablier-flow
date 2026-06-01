"""On-disk cache for TEE-side generation results.

A generation call costs real money (TEE GPU time). Re-running the same
notebook should not pay it twice. This module is a tiny content-addressed
store keyed by the inputs the customer supplied:

    cache_key = sha256(parquet_bytes ++ json(params, sort_keys=True))

The cache lives under ``cache_dir`` (default ``~/.cache/sablier_flow``).
Each hit returns the same plaintext result bytes the SDK would have
decrypted from the TEE — downstream code in
:meth:`Client._run_job` doesn't care whether they came from a cache hit
or a fresh round-trip.

What we cache: only the decrypted result. We do NOT cache:
  - Attestation quotes (always fetch fresh; pinning is a security boundary)
  - Encrypted envelopes (they're symmetric to one-shot keys, useless once decrypted)
  - Anything that would be re-served plaintext to a different API key

What we don't validate at cache hit time: the SDK version. Cache entries
include the ``sdk_version`` they were produced under so the caller can
choose to invalidate on upgrade; today the Client accepts any hit
regardless. Bump the cache layout via ``_CACHE_LAYOUT_VERSION`` if the
on-disk format ever changes incompatibly.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path

__all__ = ["DiskCache", "default_cache_dir", "make_cache_key"]


# Keys mix endpoint + sha256(api_key)[:16] to prevent cross-environment
# (staging vs prod) and cross-user (shared machine, multiple keys)
# collisions.
_CACHE_LAYOUT_VERSION = 2


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/sablier_flow`` or ``~/.cache/sablier_flow``."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "sablier_flow"
    return Path.home() / ".cache" / "sablier_flow"


def make_cache_key(
    parquet_bytes: bytes,
    params: dict,
    *,
    endpoint: str,
    api_key: str,
) -> str:
    """Deterministic key for one logical generation request.

    Hash inputs:
      - Cache layout version (lets us invalidate the world if the on-disk
        format changes)
      - Endpoint (so staging and prod never collide on the same machine)
      - sha256(api_key)[:16] (so two users sharing a cache dir don't see
        each other's plaintext results, and so a re-issued key invalidates
        previous entries — never the raw api_key, which would write a
        credential-shaped fingerprint to disk)
      - Params dict serialized with sorted keys (so {"a":1,"b":2} and
        {"b":2,"a":1} collide as intended)
      - Parquet bytes (whole payload; includes the user's data + schema)
    """
    api_key_fp = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    h = hashlib.sha256()
    h.update(f"v{_CACHE_LAYOUT_VERSION}\n".encode())
    h.update(endpoint.encode("utf-8"))
    h.update(b"\n")
    h.update(api_key_fp.encode("ascii"))
    h.update(b"\n")
    h.update(json.dumps(params, sort_keys=True, default=str).encode("utf-8"))
    h.update(b"\n")
    h.update(parquet_bytes)
    return h.hexdigest()


class DiskCache:
    """Tiny content-addressed bytes store rooted at ``cache_dir``.

    Files are written atomically (write to a tempfile in the same
    directory, then ``os.replace``) so a crashing process can never leave
    partial entries.

    The cache is not bounded. Operators that need bounds run
    ``rm -rf $CACHE_DIR`` or a cron sweep — keeping the cache itself
    dumb is the whole point.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        # 0o700: the cache holds plaintext TEE results derived from this
        # user's api_key. On a shared host (CI runner, jump box), 0o755
        # would let any local account read them. mkdir() ignores `mode`
        # when the dir already exists, so we chmod unconditionally
        # afterwards. File entries are written 0o600 by tempfile.mkstemp.
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.suppress(OSError):
            os.chmod(self.cache_dir, 0o700)

    def _path_for(self, key: str) -> Path:
        # Fan out into 2-char shards so a single directory doesn't accrete
        # tens of thousands of files (which slows down `ls` and ext4 lookups).
        return self.cache_dir / key[:2] / key

    def get(self, key: str) -> bytes | None:
        """Return cached bytes for ``key``, or None if absent."""
        p = self._path_for(key)
        if not p.is_file():
            return None
        try:
            return p.read_bytes()
        except OSError:
            return None

    def put(self, key: str, value: bytes) -> None:
        """Store ``value`` under ``key``. Atomic — if the process is
        killed mid-write, the cache is unchanged."""
        target = self._path_for(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=key + ".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(value)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
