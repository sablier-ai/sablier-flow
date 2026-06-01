"""Thin HTTP client for the remote sablier-flow hosted service.

  - :class:`Client` — main entry point, locked public API
  - :func:`fit`, :func:`generate`, :func:`validate` — module-level shortcuts
  - :class:`AttestationVerifier`, :class:`AttestationQuote` — verify the
    TEE's attestation quote against the SDK-pinned image digest before any
    plaintext leaves the customer's machine.
  - :func:`envelope_encrypt`, :func:`envelope_decrypt`, :class:`EnvelopeEncrypted` —
    the X25519 + AES-256-GCM envelope-encryption protocol used to ship
    customer data into the TEE.
"""

from sablier_flow.client.attestation import (
    AttestationQuote,
    AttestationVerificationError,
    AttestationVerifier,
)
from sablier_flow.client.client import Client, fit, generate, validate
from sablier_flow.client.crypto import (
    EnvelopeEncrypted,
    envelope_decrypt,
    envelope_encrypt,
    load_tee_public_key,
)
from sablier_flow.client.transport import (
    AuthenticationError,
    JobNotFoundError,
    ModelNotFoundError,
    RemoteJobError,
    SablierClientError,
    TransportError,
)

__all__ = [
    "AttestationQuote",
    "AttestationVerificationError",
    "AttestationVerifier",
    "AuthenticationError",
    "Client",
    "EnvelopeEncrypted",
    "JobNotFoundError",
    "ModelNotFoundError",
    "RemoteJobError",
    "SablierClientError",
    "TransportError",
    "envelope_decrypt",
    "envelope_encrypt",
    "fit",
    "generate",
    "load_tee_public_key",
    "validate",
]
