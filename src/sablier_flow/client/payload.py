"""Serialization for the SDK ↔ TEE wire payloads.

Two flat binary structs cross the wire each job:

  - :class:`JobUploadPayload` — what the customer envelope-encrypts and
    uploads. Carries the input DataFrame (as Parquet bytes), the job
    parameters (JSON), and a one-shot symmetric key (32-byte AES-256-GCM)
    the TEE will use to encrypt the response back to the customer.

  - :class:`JobResultPayload` — what the TEE encrypts with the upload's
    result_key and returns. Carries the GenerationResult's arrays + a
    JSON metadata blob.

The "result key" is a simple shared secret tucked inside the
envelope-encrypted upload. It never traverses the network in cleartext.
For v1 production this is sufficient: the result_key is bound to a
single job, the TEE wipes it on shutdown, and the customer holds the
only copy outside the enclave. The follow-up (Workstream D real impl)
swaps this for a customer-controlled KMS key when long-lived
re-encryption is needed.

Wire format (all integers big-endian, lengths are uint32):

JobUploadPayload::

    [4]   magic = b'SFup'
    [4]   version = 1
    [4]   result_key length (= 32)
    [N]   result_key bytes
    [4]   params_json length
    [N]   params_json bytes (UTF-8)
    [4]   data_parquet length
    [N]   data_parquet bytes

JobResultPayload::

    [4]   magic = b'SFre'
    [4]   version = 1
    [4]   metadata_json length
    [N]   metadata_json bytes (UTF-8)
    [4]   paths_returns_npy length (may be 0 for fit/validate results)
    [N]   paths_returns numpy .npy bytes
    [4]   paths_prices_npy length (may be 0 for fit/validate results)
    [N]   paths_prices numpy .npy bytes
    [4]   last_prices_npy length (may be 0 for fit/validate results)
    [N]   last_prices numpy .npy bytes

A single payload type covers all three job kinds. The metadata JSON
carries a ``"kind"`` field — ``"generate"``, ``"fit"``, or
``"validate"`` — and the array slots are empty for fit/validate (which
return only a model_id or a report dict).

The simple length-prefixed framing keeps parsing trivial and avoids
pulling in protobuf/msgpack/etc.
"""

from __future__ import annotations

import io
import json
import os
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from sablier_flow.client.crypto import GCM_TAG_LENGTH, NONCE_LENGTH
from sablier_flow.types import FitResult, GenerationResult, ValidationReport

__all__ = [
    "GCM_TAG_LENGTH",
    "NONCE_LENGTH",
    "JobResultPayload",
    "JobUploadPayload",
    "decrypt_result",
    "encrypt_result",
    "new_result_key",
]


_UPLOAD_MAGIC = b"SFup"
_RESULT_MAGIC = b"SFre"
_WIRE_VERSION = 1
RESULT_KEY_LENGTH = 32  # AES-256-GCM


def _read_lp(buf: bytes, offset: int) -> tuple[bytes, int]:
    """Read a length-prefixed bytes blob. Returns (blob, new_offset)."""
    (length,) = struct.unpack(">I", buf[offset : offset + 4])
    offset += 4
    return buf[offset : offset + length], offset + length


# ============================================================================
# Upload payload (customer -> TEE, envelope-encrypted)
# ============================================================================


@dataclass(frozen=True)
class JobUploadPayload:
    """The plaintext bytes that go inside the envelope encryption sent
    to the TEE."""

    result_key: bytes  # 32 bytes
    params_json: bytes  # UTF-8 encoded JSON
    data_parquet: bytes  # input DataFrame as Parquet

    def to_bytes(self) -> bytes:
        if len(self.result_key) != RESULT_KEY_LENGTH:
            raise ValueError(
                f"result_key must be {RESULT_KEY_LENGTH} bytes; got {len(self.result_key)}"
            )
        return (
            _UPLOAD_MAGIC
            + struct.pack(">I", _WIRE_VERSION)
            + struct.pack(">I", len(self.result_key)) + self.result_key
            + struct.pack(">I", len(self.params_json)) + self.params_json
            + struct.pack(">I", len(self.data_parquet)) + self.data_parquet
        )

    @classmethod
    def from_bytes(cls, buf: bytes) -> JobUploadPayload:
        if buf[:4] != _UPLOAD_MAGIC:
            raise ValueError(f"bad magic for JobUploadPayload: {buf[:4]!r}")
        (version,) = struct.unpack(">I", buf[4:8])
        if version != _WIRE_VERSION:
            raise ValueError(f"unsupported JobUploadPayload version: {version}")
        offset = 8
        result_key, offset = _read_lp(buf, offset)
        params_json, offset = _read_lp(buf, offset)
        data_parquet, offset = _read_lp(buf, offset)
        return cls(result_key=result_key, params_json=params_json, data_parquet=data_parquet)

    @classmethod
    def build(
        cls,
        *,
        data_parquet: bytes,
        params: dict[str, Any],
        result_key: bytes | None = None,
    ) -> JobUploadPayload:
        """Convenience constructor: pack a params dict + Parquet bytes
        into a payload, generating a fresh result_key by default."""
        if result_key is None:
            result_key = new_result_key()
        return cls(
            result_key=result_key,
            params_json=json.dumps(params, default=str, sort_keys=True).encode("utf-8"),
            data_parquet=data_parquet,
        )

    @property
    def params(self) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self.params_json.decode("utf-8"))
        return result


# ============================================================================
# Result payload (TEE -> customer, AES-GCM encrypted with the upload's result_key)
# ============================================================================


@dataclass(frozen=True)
class JobResultPayload:
    """The plaintext bytes that the TEE encrypts with the result_key
    and returns to the customer.

    The same struct serves all three job kinds — ``generate`` populates
    every field, ``fit`` and ``validate`` populate only ``metadata_json``
    (the arrays carry empty bytes). The receiver picks the right
    ``to_*_result()`` based on the ``kind`` field inside the metadata.
    """

    metadata_json: bytes
    paths_returns_npy: bytes = b""
    paths_prices_npy: bytes = b""
    last_prices_npy: bytes = b""

    def to_bytes(self) -> bytes:
        return (
            _RESULT_MAGIC
            + struct.pack(">I", _WIRE_VERSION)
            + struct.pack(">I", len(self.metadata_json)) + self.metadata_json
            + struct.pack(">I", len(self.paths_returns_npy)) + self.paths_returns_npy
            + struct.pack(">I", len(self.paths_prices_npy)) + self.paths_prices_npy
            + struct.pack(">I", len(self.last_prices_npy)) + self.last_prices_npy
        )

    @classmethod
    def from_bytes(cls, buf: bytes) -> JobResultPayload:
        if buf[:4] != _RESULT_MAGIC:
            raise ValueError(f"bad magic for JobResultPayload: {buf[:4]!r}")
        (version,) = struct.unpack(">I", buf[4:8])
        if version != _WIRE_VERSION:
            raise ValueError(f"unsupported JobResultPayload version: {version}")
        offset = 8
        metadata_json, offset = _read_lp(buf, offset)
        paths_returns_npy, offset = _read_lp(buf, offset)
        paths_prices_npy, offset = _read_lp(buf, offset)
        last_prices_npy, offset = _read_lp(buf, offset)
        return cls(
            metadata_json=metadata_json,
            paths_returns_npy=paths_returns_npy,
            paths_prices_npy=paths_prices_npy,
            last_prices_npy=last_prices_npy,
        )

    # ----- bridge to/from GenerationResult ---------------------------------

    @classmethod
    def from_generation_result(cls, r: GenerationResult) -> JobResultPayload:
        meta: dict[str, Any] = {
            "kind": "generate",
            "feature_names": r.feature_names,
            "horizon": int(r.horizon),
            "n_paths": int(r.n_paths),
            "seed": r.seed,
            "sdk_version": r.sdk_version,
        }
        # Optional diagnostics — included only if the TEE populated them.
        # validation_overall / validation_metrics were dropped in 0.3.0;
        # the structural-validation suite now lives exclusively in
        # Client.validate (separate billed call). Older server payloads
        # may still ship those keys, but :meth:`to_generation_result`
        # ignores them.
        if r.memorization_risk is not None:
            meta["memorization_risk"] = r.memorization_risk
            meta["memorization_nn_distance_ratio"] = r.memorization_nn_distance_ratio
        return cls(
            metadata_json=json.dumps(meta, sort_keys=True, default=str).encode("utf-8"),
            paths_returns_npy=_arr_to_npy(np.asarray(r.paths_returns)),
            paths_prices_npy=_arr_to_npy(np.asarray(r.paths_prices)),
            last_prices_npy=_arr_to_npy(np.asarray(r.last_prices)),
        )

    def to_generation_result(self) -> GenerationResult:
        meta = json.loads(self.metadata_json.decode("utf-8"))
        return GenerationResult(
            paths_returns=_npy_to_arr(self.paths_returns_npy),
            paths_prices=_npy_to_arr(self.paths_prices_npy),
            feature_names=list(meta["feature_names"]),
            last_prices=_npy_to_arr(self.last_prices_npy),
            horizon=int(meta["horizon"]),
            n_paths=int(meta["n_paths"]),
            seed=meta["seed"],
            sdk_version=str(meta["sdk_version"]),
            memorization_risk=meta.get("memorization_risk"),
            memorization_nn_distance_ratio=meta.get("memorization_nn_distance_ratio"),
        )

    # ----- bridge to/from FitResult ---------------------------------------

    @classmethod
    def from_fit_result(cls, r: FitResult) -> JobResultPayload:
        meta: dict[str, Any] = {
            "kind": "fit",
            "model_id": r.model_id,
            "features": list(r.features),
            "training_horizon": int(r.training_horizon),
            "training_end_date": r.training_end_date,
            "sdk_version": r.sdk_version,
            "expires_at": r.expires_at,
            "training_loss": r.training_loss,
            "loss_source": r.loss_source,
            "training_start_date": r.training_start_date,
            "holdout_start_date": r.holdout_start_date,
            "holdout_end_date": r.holdout_end_date,
        }
        return cls(
            metadata_json=json.dumps(meta, sort_keys=True, default=str).encode("utf-8"),
        )

    def to_fit_result(self) -> FitResult:
        meta = json.loads(self.metadata_json.decode("utf-8"))
        return FitResult(
            model_id=str(meta["model_id"]),
            features=list(meta["features"]),
            training_horizon=int(meta["training_horizon"]),
            training_end_date=meta.get("training_end_date"),
            sdk_version=str(meta["sdk_version"]),
            expires_at=meta.get("expires_at"),
            training_loss=meta.get("training_loss"),
            loss_source=meta.get("loss_source"),
            training_start_date=meta.get("training_start_date"),
            holdout_start_date=meta.get("holdout_start_date"),
            holdout_end_date=meta.get("holdout_end_date"),
        )

    # ----- bridge to/from ValidationReport --------------------------------

    @classmethod
    def from_validation_report(cls, r: ValidationReport) -> JobResultPayload:
        meta: dict[str, Any] = {
            "kind": "validate",
            "overall": r.overall,
            "metrics": r.metrics,
            "memorization_risk": r.memorization_risk,
            "memorization_nn_distance_ratio": r.memorization_nn_distance_ratio,
            "n_paths_used": r.n_paths_used,
            "holdout": r.holdout,
        }
        return cls(
            metadata_json=json.dumps(meta, sort_keys=True, default=str).encode("utf-8"),
        )

    def to_validation_report(self) -> ValidationReport:
        meta = json.loads(self.metadata_json.decode("utf-8"))
        return ValidationReport(
            overall=str(meta["overall"]),
            metrics=dict(meta.get("metrics", {})),
            memorization_risk=meta.get("memorization_risk"),
            memorization_nn_distance_ratio=meta.get("memorization_nn_distance_ratio"),
            n_paths_used=meta.get("n_paths_used"),
            holdout=bool(meta.get("holdout", False)),
        )

    @property
    def kind(self) -> str:
        """The job kind that produced this payload (``generate`` / ``fit`` /
        ``validate``). Older payloads without an explicit kind default to
        ``generate`` for backwards compat."""
        try:
            meta = json.loads(self.metadata_json.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return "generate"
        return str(meta.get("kind", "generate"))


def _arr_to_npy(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to its standard .npy byte format."""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _npy_to_arr(blob: bytes) -> np.ndarray:
    """Inverse of :func:`_arr_to_npy`."""
    arr: np.ndarray = np.load(io.BytesIO(blob), allow_pickle=False)
    return arr


# ============================================================================
# Symmetric AES-256-GCM helpers for the return-trip ciphertext
# ============================================================================


def new_result_key() -> bytes:
    """Generate a fresh random 256-bit key."""
    return os.urandom(RESULT_KEY_LENGTH)


def encrypt_result(
    plaintext: bytes,
    result_key: bytes,
    *,
    associated_data: bytes = b"sablier-flow-result-v1",
) -> bytes:
    """Encrypt a JobResultPayload's bytes with AES-256-GCM.

    Returns ``nonce || ciphertext_with_tag``. The 12-byte nonce is
    generated fresh per call and prefixed so the recipient can recover
    it without an out-of-band channel.
    """
    if len(result_key) != RESULT_KEY_LENGTH:
        raise ValueError(
            f"result_key must be {RESULT_KEY_LENGTH} bytes; got {len(result_key)}"
        )
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(NONCE_LENGTH)
    ct = AESGCM(result_key).encrypt(nonce, plaintext, associated_data)
    return nonce + ct


def decrypt_result(
    ciphertext: bytes,
    result_key: bytes,
    *,
    associated_data: bytes = b"sablier-flow-result-v1",
) -> bytes:
    """Inverse of :func:`encrypt_result`."""
    if len(result_key) != RESULT_KEY_LENGTH:
        raise ValueError(
            f"result_key must be {RESULT_KEY_LENGTH} bytes; got {len(result_key)}"
        )
    if len(ciphertext) < NONCE_LENGTH + GCM_TAG_LENGTH:
        raise ValueError(
            f"ciphertext too short to contain nonce + GCM tag (need at least "
            f"{NONCE_LENGTH + GCM_TAG_LENGTH} bytes; got {len(ciphertext)})"
        )
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = ciphertext[:NONCE_LENGTH]
    body = ciphertext[NONCE_LENGTH:]
    return AESGCM(result_key).decrypt(nonce, body, associated_data)
