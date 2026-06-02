"""Attestation-quote signing primitives.

The current trust narrative requires every quote to be signed by a set
of pinned roots (in production: Google Attestation Service + NVIDIA
NRAS + AMD SEV-SNP VCEK chain). the current release ships:

  - A canonical JSON serialization the signer + verifier both agree on.
  - An Ed25519 signature primitive (real math via the ``cryptography``
    library) that exercises the full sign-then-verify flow end-to-end.
  - A pluggable :class:`RootKeyRegistry` Protocol so Future iteration's
    impl can replace the in-memory registry with the actual AMD ARK/ASK
    + NVIDIA RIM + Google AS pinned keys.

Production attestation chains are ECDSA-P256, not Ed25519, but the
seam (canonicalization + per-issuer keypair lookup + detached
signature verification) is identical. Swapping the algorithm is a
one-line change in :func:`verify_signature` / :func:`sign_payload`.

Signature layout: detached. The signatures live inside the JSON
envelope under ``signatures``, but the *payload* each signature covers
is the JSON envelope with the ``signatures`` field stripped (and then
serialized via :func:`canonicalize_payload`). This avoids the
chicken-and-egg of self-referential signature blobs.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

__all__ = [
    "InMemoryRootKeyRegistry",
    "Issuer",
    "RootKeyRegistry",
    "SignatureBlob",
    "SignatureVerificationError",
    "Signer",
    "SignerSet",
    "canonicalize_payload",
    "sign_payload",
    "verify_signature",
]


# Issuers the SDK + server recognize. Production adds 'amd' for the
# SEV-SNP VCEK chain; the current release carries Google + NVIDIA which is what the
# existing AttestationVerifier required.
Issuer = str  # 'google' | 'nvidia' | 'amd' | ...


class SignatureVerificationError(Exception):
    """Raised when a quote signature fails verification."""


# ---------------------------------------------------------------------------
# Wire layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignatureBlob:
    """One signature entry in the quote JSON's ``signatures`` array."""

    issuer: Issuer
    alg: str                # "Ed25519" for early releases; "ES256" in production
    value: str              # base64-standard-encoded signature bytes

    def value_bytes(self) -> bytes:
        return base64.b64decode(self.value)

    @classmethod
    def from_json_dict(cls, d: dict[str, str]) -> SignatureBlob:
        return cls(
            issuer=str(d["issuer"]).lower(),
            alg=str(d.get("alg", "")),
            value=str(d["value"]),
        )

    def to_json_dict(self) -> dict[str, str]:
        return {"issuer": self.issuer, "alg": self.alg, "value": self.value}


def canonicalize_payload(payload: dict[str, Any]) -> bytes:
    """Stable JSON serialization of the attestation payload, used as
    the "message" each signature covers.

    Strips the ``signatures`` field if present (detached layout),
    serializes with sorted keys + compact separators so the bytes are
    deterministic for a given dict.
    """
    cleaned = {k: v for k, v in payload.items() if k != "signatures"}
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Pluggable root-key registry
# ---------------------------------------------------------------------------


class RootKeyRegistry(Protocol):
    """Lookup the pinned root public key for an issuer.

    Production registry returns the pinned AMD ARK/ASK / NVIDIA RIM /
    Google AS public keys baked into the SDK release. Tests inject an
    in-memory registry keyed by issuer name.
    """

    def public_key(self, issuer: Issuer) -> Ed25519PublicKey | None:
        ...


class InMemoryRootKeyRegistry:
    """Dev / test registry — a plain dict from issuer to public key.

    Production swaps for a registry that reads the pinned key bytes
    from the SDK wheel (so a customer airgap-pin stays valid until the
    next SDK release).
    """

    def __init__(self, keys: dict[Issuer, Ed25519PublicKey] | None = None) -> None:
        self._keys: dict[Issuer, Ed25519PublicKey] = dict(keys or {})

    def register(self, issuer: Issuer, key: Ed25519PublicKey) -> None:
        self._keys[issuer.lower()] = key

    def register_raw(self, issuer: Issuer, raw_bytes: bytes) -> None:
        """Convenience: load a 32-byte raw Ed25519 pubkey."""
        self._keys[issuer.lower()] = Ed25519PublicKey.from_public_bytes(raw_bytes)

    def public_key(self, issuer: Issuer) -> Ed25519PublicKey | None:
        return self._keys.get(issuer.lower())


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def verify_signature(
    payload_bytes: bytes,
    signature_value: bytes,
    public_key: Ed25519PublicKey,
) -> None:
    """Verify a single detached signature. Raises on failure."""
    try:
        public_key.verify(signature_value, payload_bytes)
    except InvalidSignature as exc:
        raise SignatureVerificationError("invalid signature") from exc


# ---------------------------------------------------------------------------
# Signer (TEE side)
# ---------------------------------------------------------------------------


@dataclass
class Signer:
    """One issuer's private key + algorithm tag. Held inside the TEE.

    Production wires this to per-issuer attestation hardware (AMD VCEK
    handle, NVIDIA NRAS endpoint, Google AS metadata). For the current release the
    private key bytes live inside the in-enclave container only.
    """

    issuer: Issuer
    private_key: Ed25519PrivateKey
    alg: str = "Ed25519"

    @classmethod
    def generate(cls, issuer: Issuer) -> Signer:
        return cls(issuer=issuer.lower(), private_key=Ed25519PrivateKey.generate())

    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()

    def public_key_bytes(self) -> bytes:
        return self.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def sign(self, payload_bytes: bytes) -> SignatureBlob:
        sig = self.private_key.sign(payload_bytes)
        return SignatureBlob(
            issuer=self.issuer,
            alg=self.alg,
            value=base64.b64encode(sig).decode("ascii"),
        )


class SignerSet:
    """A collection of :class:`Signer` instances by issuer.

    The TEE side wires one of these into :func:`sign_payload` to mint
    quotes. ``required_issuers`` is the set the client expects; missing
    any of these from a sign call raises.
    """

    def __init__(self, signers: list[Signer]) -> None:
        self._by_issuer: dict[Issuer, Signer] = {s.issuer: s for s in signers}

    @property
    def issuers(self) -> list[Issuer]:
        return list(self._by_issuer.keys())

    def signer(self, issuer: Issuer) -> Signer:
        return self._by_issuer[issuer.lower()]

    def to_registry(self) -> InMemoryRootKeyRegistry:
        """Build a RootKeyRegistry containing each signer's public key —
        the client-side equivalent of "trust these roots"."""
        reg = InMemoryRootKeyRegistry()
        for signer in self._by_issuer.values():
            reg.register(signer.issuer, signer.public_key())
        return reg


def sign_payload(
    payload: dict[str, Any],
    signer_set: SignerSet,
    *,
    required_issuers: list[Issuer] | None = None,
) -> list[SignatureBlob]:
    """Produce one detached signature per required issuer over the
    canonicalized payload.

    Caller is responsible for inserting the returned list under the
    ``signatures`` key of the JSON envelope. Raises ``KeyError`` if
    the signer_set doesn't cover every required_issuers entry.
    """
    if required_issuers is None:
        required_issuers = ["google", "nvidia"]
    payload_bytes = canonicalize_payload(payload)
    return [signer_set.signer(issuer).sign(payload_bytes) for issuer in required_issuers]
