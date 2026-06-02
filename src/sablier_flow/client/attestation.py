"""Client-side attestation verifier.

Before the SDK releases the customer's encryption key into a TEE envelope,
it verifies the TEE's attestation quote against a **pinned digest**
baked into this SDK release. The trust narrative:

  1. The SDK ships with a single, hash-pinned TEE image digest.
  2. Sablier publishes the corresponding source — customers can rebuild
     and verify the hash if they want full transparency.
  3. Before any data leaves the customer's machine, the SDK demands a
     fresh attestation quote.
  4. The verifier checks:
       - quote is signed by the expected root (Google + NVIDIA + AMD)
       - image_digest in the quote MATCHES the SDK's pinned digest
       - the quote includes a fresh ephemeral X25519 pubkey
       - VM measurements (firmware, kernel, boot config) match expected
  5. Only if all checks pass does the client envelope-encrypt to that
     ephemeral key.

This module ships **v1 of the verification protocol**. The signature
verification primitives are pure-Python via the cryptography library;
the policy (which roots / which measurements) is pluggable for the
production deploy.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "AttestationQuote",
    "AttestationVerificationError",
    "AttestationVerifier",
    "_reset_attestation_warning",
]

_logger = logging.getLogger(__name__)

# Module-level dedupe flag so we emit the "production mode without registry"
# warning exactly once per process, no matter how many verifiers a client
# instantiates over the lifetime of the SDK.
_PROD_NO_REGISTRY_WARNED = False

# Canonical default attestation mode. Lives here (not as a literal in the
# signature) so the constructor can distinguish "caller passed mode=" from
# "caller accepted the default". Pre-H100 / pre-TEE that distinction is the
# whole reason the noisy-warning fix exists: every default-constructed
# verifier is otherwise indistinguishable from a customer who deliberately
# asked for production-mode verification.
_DEFAULT_ATTESTATION_MODE = "production"

# Sentinel sigil for the mode kwarg. Identity-compared against the bound
# default to detect default-derived construction without changing the public
# type (callers still see ``mode: _TEEMode``).
_MODE_DEFAULT_SENTINEL = object()

# Opt-out env var. Set to "1" / "true" / "yes" / "on" (case-insensitive) to
# silence the production-without-registry warning even when the caller
# explicitly opts into production mode. Useful in CI/scripts that already
# know the gap and don't want log noise.
_SUPPRESS_WARNING_ENV = "SABLIER_FLOW_SUPPRESS_ATTESTATION_WARNING"


def _warning_suppressed_by_env() -> bool:
    """True iff the suppression env var is set to an affirmative value."""
    val = os.environ.get(_SUPPRESS_WARNING_ENV, "")
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _reset_attestation_warning() -> None:
    """Clear the module-level one-shot dedupe flag.

    Public test helper: the production-without-registry warning fires at most
    once per process. Tests that need to observe whether the warning fires
    across multiple ``AttestationVerifier`` constructions should call this
    between cases (or fixture-wrap it) so each construction starts from a
    clean slate. Not intended for application code.
    """
    global _PROD_NO_REGISTRY_WARNED
    _PROD_NO_REGISTRY_WARNED = False

# ============================================================================
# Quote structure (canonical JSON envelope)
# ============================================================================


@dataclass(frozen=True)
class AttestationQuote:
    """Parsed attestation quote returned by the TEE.

    Wire format: a base64-url-encoded JSON envelope from the server, with
    the following fields:

      protocol_version   int       (currently 1)
      image_digest       str       sha256:<hex>
      tee_type           str       "CONFIDENTIAL_SPACE_A3_H100" | similar
      hardware           str       e.g. "NVIDIA_H100"
      nonce              str       base64
      ephemeral_pubkey   str       base64 (32 bytes raw X25519)
      vm_measurements    dict      sub-fields per TEE provider
      issued_at          str       ISO-8601 UTC
      expires_at         str       ISO-8601 UTC
      signatures         list[dict]   one entry per signing root
        each: { "issuer": str, "alg": str, "value": str (base64) }

    The verifier checks signatures + measurements + pinned digest before
    trusting the ephemeral_pubkey.
    """

    protocol_version: int
    image_digest: str
    tee_type: str
    hardware: str
    nonce: str
    ephemeral_pubkey: str
    vm_measurements: dict[str, str]
    issued_at: str
    expires_at: str
    signatures: list[dict[str, str]]
    raw: bytes

    @classmethod
    def from_wire(cls, wire_bytes: bytes) -> AttestationQuote:
        """Decode the base64-url JSON envelope back into a dataclass."""
        try:
            decoded = base64.urlsafe_b64decode(wire_bytes)
            payload = json.loads(decoded)
        except (ValueError, json.JSONDecodeError) as exc:
            raise AttestationVerificationError(
                f"could not decode attestation quote: {exc}"
            ) from exc
        return cls(
            protocol_version=int(payload["protocol_version"]),
            image_digest=str(payload["image_digest"]),
            tee_type=str(payload["tee_type"]),
            hardware=str(payload["hardware"]),
            nonce=str(payload["nonce"]),
            ephemeral_pubkey=str(payload["ephemeral_pubkey"]),
            vm_measurements=dict(payload.get("vm_measurements", {})),
            issued_at=str(payload["issued_at"]),
            expires_at=str(payload["expires_at"]),
            signatures=list(payload.get("signatures", [])),
            raw=wire_bytes,
        )

    @property
    def ephemeral_pubkey_bytes(self) -> bytes:
        return base64.b64decode(self.ephemeral_pubkey)


# ============================================================================
# Verifier
# ============================================================================


class AttestationVerificationError(Exception):
    """Raised when an attestation quote fails any verification check."""


_TEEMode = Literal["production", "fake-for-dev"]


class AttestationVerifier:
    """Stateless verifier for sablier-flow attestation quotes.

    Default behavior: production mode. Structural checks (pinned image
    digest, TEE type / hardware, measurements, freshness, signature
    *presence*) are always strict. **Cryptographic signature
    verification only happens when ``root_key_registry`` is provided.**
    When the registry is ``None``, signature math is skipped (issuers
    must still be present, but the bytes are not validated against any
    pinned key). Future iteration will land the strict path that makes the
    registry mandatory and turns the missing-registry case into a hard
    error.

    Inspect :attr:`strict_verification_active` to tell at runtime
    whether full crypto verification is wired up.

    Typical client usage::

        verifier = AttestationVerifier(
            expected_image_digest='sha256:abc123...',
            expected_tee_type='CONFIDENTIAL_SPACE_A3_H100',
            expected_hardware='NVIDIA_H100',
            root_key_registry=my_pinned_keys,  # enables strict mode
        )
        quote = AttestationQuote.from_wire(server_response_bytes)
        verifier.verify(quote)
        # If we get here, the TEE is what the SDK release pinned and we
        # can trust quote.ephemeral_pubkey_bytes.

    For local dev / testing, instantiate with ``mode='fake-for-dev'``:
    signature checks are skipped and any quote with a matching image
    digest passes. NEVER use fake mode in a production wheel — the
    pinned digest enforcement is the entire trust story.
    """

    def __init__(
        self,
        *,
        expected_image_digest: str,
        expected_tee_type: str = "CONFIDENTIAL_SPACE_A3_H100",
        expected_hardware: str = "NVIDIA_H100",
        expected_measurements: dict[str, str] | None = None,
        mode: _TEEMode = _MODE_DEFAULT_SENTINEL,  # type: ignore[assignment]
        mode_explicit: bool | None = None,
        root_key_registry: Any = None,
        required_issuers: tuple[str, ...] = ("google", "nvidia"),
    ) -> None:
        if not expected_image_digest.startswith("sha256:"):
            raise ValueError(
                f"expected_image_digest must start with 'sha256:' "
                f"(got {expected_image_digest!r})"
            )
        # Detect whether the caller explicitly passed ``mode=`` vs took the
        # default. Identity comparison against the sentinel — value-equality
        # would false-positive any caller that happens to pass the same
        # string as the default.
        #
        # The ``mode_explicit`` kwarg lets upstream wrappers (notably
        # :class:`sablier_flow.Client`, which has its own user-facing
        # ``attestation_mode=`` default) override this inference: those
        # wrappers know whether the *user* passed a value and pass that
        # signal through directly so a wrapper-level default doesn't get
        # mistaken for an explicit user opt-in. When omitted (``None``)
        # we fall back to the sentinel-identity check.
        if mode_explicit is None:
            mode_explicit = mode is not _MODE_DEFAULT_SENTINEL
        if mode is _MODE_DEFAULT_SENTINEL:
            mode = _DEFAULT_ATTESTATION_MODE  # type: ignore[assignment]

        self.expected_image_digest = expected_image_digest
        self.expected_tee_type = expected_tee_type
        self.expected_hardware = expected_hardware
        self.expected_measurements = expected_measurements or {}
        self.mode = mode
        self.root_key_registry = root_key_registry
        self.required_issuers = tuple(i.lower() for i in required_issuers)

        # Emit a one-shot warning when a caller *explicitly* asks for
        # production-mode verification but hasn't wired any pinned root
        # keys. Without a registry the signature math is silently skipped
        # (see _check_signatures), which is decidedly *not* what the name
        # "production" implies. Future iteration will turn this into a hard
        # error; until then, surfacing the gap once per process is the
        # least we can do.
        #
        # Pre-H100 / pre-TEE, default-derived production mode is the
        # expected state for every customer who has no need for attestation
        # — emitting a warning there just creates noise on every
        # fit/generate/validate cycle. The warning only fires when the
        # caller passes ``mode="production"`` explicitly (signalling intent
        # to verify) or can be suppressed entirely via the env var below.
        if (
            self.mode == "production"
            and self.root_key_registry is None
            and mode_explicit
            and not _warning_suppressed_by_env()
        ):
            global _PROD_NO_REGISTRY_WARNED
            if not _PROD_NO_REGISTRY_WARNED:
                _PROD_NO_REGISTRY_WARNED = True
                _logger.warning(
                    "Production mode without a root_key_registry: signature "
                    "verification is disabled. Wire "
                    "AttestationVerifier(root_key_registry=...) to enable "
                    "strict checks. See https://docs.sablier.ai/attestation"
                )

    # ------------------------------------------------------------------
    # Public introspection
    # ------------------------------------------------------------------

    @property
    def strict_verification_active(self) -> bool:
        """True iff this verifier will run full cryptographic signature
        verification on quotes it inspects.

        Strict verification requires production mode AND a non-None
        ``root_key_registry``. In fake-for-dev mode, or in production
        mode with no registry, this returns False.
        """
        return self.mode == "production" and self.root_key_registry is not None

    # ------------------------------------------------------------------
    # The single public entrypoint
    # ------------------------------------------------------------------

    def verify(self, quote: AttestationQuote) -> bytes:
        """Run all checks; raise on any failure. Return the ephemeral
        public key (32 raw X25519 bytes) on success."""
        self._check_protocol_version(quote)
        self._check_pinned_image_digest(quote)
        self._check_tee_type_and_hardware(quote)
        self._check_measurements(quote)
        self._check_freshness(quote)
        self._check_signatures(quote)

        pubkey = quote.ephemeral_pubkey_bytes
        if len(pubkey) != 32:
            raise AttestationVerificationError(
                f"ephemeral_pubkey must be 32 bytes (raw X25519); got {len(pubkey)}"
            )
        return pubkey

    # ------------------------------------------------------------------
    # Individual checks (broken out for testability)
    # ------------------------------------------------------------------

    def _check_protocol_version(self, quote: AttestationQuote) -> None:
        if quote.protocol_version != 1:
            raise AttestationVerificationError(
                f"unsupported quote protocol version {quote.protocol_version}; "
                "this SDK release speaks version 1"
            )

    def _check_pinned_image_digest(self, quote: AttestationQuote) -> None:
        if quote.image_digest != self.expected_image_digest:
            raise AttestationVerificationError(
                f"image digest mismatch: TEE is running {quote.image_digest!r} "
                f"but this SDK release pins {self.expected_image_digest!r}. "
                "Either upgrade the SDK (the TEE was updated) or downgrade the "
                "TEE deployment (the SDK is older than the running image)."
            )

    def _check_tee_type_and_hardware(self, quote: AttestationQuote) -> None:
        if quote.tee_type != self.expected_tee_type:
            raise AttestationVerificationError(
                f"TEE type mismatch: got {quote.tee_type!r}, "
                f"expected {self.expected_tee_type!r}"
            )
        if quote.hardware != self.expected_hardware:
            raise AttestationVerificationError(
                f"hardware mismatch: got {quote.hardware!r}, "
                f"expected {self.expected_hardware!r}"
            )

    def _check_measurements(self, quote: AttestationQuote) -> None:
        for key, expected in self.expected_measurements.items():
            actual = quote.vm_measurements.get(key)
            if actual != expected:
                raise AttestationVerificationError(
                    f"measurement {key!r} mismatch: got {actual!r}, "
                    f"expected {expected!r}"
                )

    def _check_freshness(self, quote: AttestationQuote) -> None:
        """Reject quotes that are issued too far in the future or already expired."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        try:
            issued = datetime.fromisoformat(quote.issued_at.replace("Z", "+00:00"))
            expires = datetime.fromisoformat(quote.expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise AttestationVerificationError(
                f"could not parse issued_at/expires_at timestamps: {exc}"
            ) from exc

        if expires < now:
            raise AttestationVerificationError(
                f"attestation quote expired at {quote.expires_at}"
            )
        # Reject clocks more than 5 min in the future to bound clock-skew exploits
        if (issued - now).total_seconds() > 300:
            raise AttestationVerificationError(
                f"quote issued_at is more than 5 minutes in the future "
                f"(now={now.isoformat()}, issued_at={quote.issued_at})"
            )

    def _check_signatures(self, quote: AttestationQuote) -> None:
        """Verify the signature chain.

        Production mode requires at least the Google + NVIDIA roots to
        sign the quote. fake-for-dev mode skips this entirely so unit
        tests + local CI can exercise the protocol without real keys.

        Two behaviors in production mode:

          1. **With a root key registry**: each required issuer's
             signature is verified against the pinned root key via
             :func:`signing.verify_signature`. Tampered payloads or
             unknown issuers are rejected.
          2. **Without a registry**: we fall back to the
             structure-only check — issuers must be *present* but the
             signature math is not validated. A logger warning makes
             the gap visible. Future iteration makes the
             registry mandatory.
        """
        if self.mode == "fake-for-dev":
            return
        if not quote.signatures:
            raise AttestationVerificationError(
                "no signatures present in attestation quote; cannot verify"
            )

        present_issuers = {sig.get("issuer", "").lower() for sig in quote.signatures}
        required = set(self.required_issuers)
        missing = required - present_issuers
        if missing:
            raise AttestationVerificationError(
                f"missing required signature issuers: {sorted(missing)}; "
                f"present: {sorted(present_issuers)}"
            )

        if self.root_key_registry is None:
            # Legacy fallback — structure-only check. See docstring.
            # The gap (no signature math) is intentional for v1 and
            # documented in the alpha-onboarding guide; the dual-protection
            # envelope is what's actually defending plaintext today. No
            # user-facing log line because it added noise to every
            # fit/generate/validate cycle without giving customers an
            # action to take. Future iteration will flip the registry to
            # required, which will produce a hard error here instead.
            return

        # Full crypto verification — each required issuer's signature
        # must validate against its pinned root key, and the bytes
        # being signed are the canonical JSON envelope sans the
        # signatures field.
        import json

        from sablier_flow.client.signing import (
            SignatureBlob,
            SignatureVerificationError,
            canonicalize_payload,
            verify_signature,
        )

        try:
            decoded = base64.urlsafe_b64decode(quote.raw)
            payload = json.loads(decoded)
        except (ValueError, json.JSONDecodeError) as exc:
            raise AttestationVerificationError(
                f"could not re-decode quote payload for signature verification: {exc}"
            ) from exc

        payload_bytes = canonicalize_payload(payload)
        sigs_by_issuer = {s.get("issuer", "").lower(): s for s in quote.signatures}

        for issuer in self.required_issuers:
            pub = self.root_key_registry.public_key(issuer)
            if pub is None:
                raise AttestationVerificationError(
                    f"no pinned root key for required issuer {issuer!r}"
                )
            blob = SignatureBlob.from_json_dict(sigs_by_issuer[issuer])
            try:
                verify_signature(payload_bytes, blob.value_bytes(), pub)
            except SignatureVerificationError as exc:
                raise AttestationVerificationError(
                    f"signature for issuer {issuer!r} failed verification"
                ) from exc
