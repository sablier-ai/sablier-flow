"""Envelope encryption — the wire protocol between customer + TEE.

The customer's data is encrypted client-side before it ever leaves their
machine. The actual encryption is hybrid:

  1. Generate a fresh AES-256-GCM data-encryption key (DEK) for each job.
  2. Encrypt the payload (Parquet bytes) with the DEK using AES-GCM.
  3. Encrypt the DEK with the TEE's ephemeral X25519 public key from
     the attestation quote.
  4. Ship ``{nonce, encrypted_dek, ciphertext, tag}`` to the TEE.

The TEE has the X25519 private key in attested memory; only it can
decrypt the DEK and then decrypt the payload. The customer's KMS-managed
master key is OPTIONAL — used for at-rest re-encryption of the result
on the way back, so the customer can hold their own decryption key for
the synthetic-paths Parquet.

Strict requirements:
  - AES-256-GCM (authenticated encryption) — never CBC or unauthenticated
  - X25519 for the ephemeral handshake (NIST-recommended curve, fast)
  - 12-byte random nonce per envelope, never reused
  - All bytes through the wire; no JSON-wrapped base64 hex strings
    (smaller, more reproducible)

Wire format (all multi-byte fields big-endian; lengths are uint32):

    [4 bytes]   protocol version (0x00000001 for v1)
    [32 bytes]  customer ephemeral X25519 public key (returned to TEE)
    [4 bytes]   encrypted_dek length
    [N bytes]   encrypted_dek
    [4 bytes]   nonce length (always 12 for AES-GCM)
    [12 bytes]  nonce
    [4 bytes]   ciphertext length
    [M bytes]   ciphertext (Parquet bytes)
    [16 bytes]  GCM tag (appended to ciphertext per cryptography lib API)

the current release ships the client-side encryption pure-Python and unit-tested.
The TEE-side decryption is symmetric and lives in `server/tee/crypto.py`.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey,
        X25519PublicKey,
    )
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
except ImportError as _exc:
    raise ImportError(
        "cryptography is required. Install with: pip install cryptography>=42.0"
    ) from _exc


PROTOCOL_VERSION = 1
DEK_LENGTH = 32       # AES-256
NONCE_LENGTH = 12     # GCM standard
GCM_TAG_LENGTH = 16   # GCM standard
X25519_KEY_LENGTH = 32

__all__ = [
    "DEK_LENGTH",
    "GCM_TAG_LENGTH",
    "NONCE_LENGTH",
    "PROTOCOL_VERSION",
    "X25519_KEY_LENGTH",
    "EnvelopeEncrypted",
    "envelope_decrypt",
    "envelope_encrypt",
    "load_tee_public_key",
]


# ============================================================================
# Output type
# ============================================================================


@dataclass(frozen=True)
class EnvelopeEncrypted:
    """The result of :func:`envelope_encrypt`.

    Use :meth:`to_bytes` to serialize for the wire and :meth:`from_bytes`
    to round-trip back.
    """

    protocol_version: int
    customer_pubkey: bytes  # 32 bytes — customer's ephemeral X25519 pubkey
    encrypted_dek: bytes    # AES-GCM-encrypted DEK (the TEE-pubkey was used to derive the wrap key)
    nonce: bytes            # 12 bytes
    ciphertext: bytes       # Parquet bytes + 16-byte GCM tag appended

    def to_bytes(self) -> bytes:
        """Serialize to the wire format documented at module top."""
        return (
            struct.pack(">I", self.protocol_version)
            + self.customer_pubkey
            + struct.pack(">I", len(self.encrypted_dek)) + self.encrypted_dek
            + struct.pack(">I", len(self.nonce)) + self.nonce
            + struct.pack(">I", len(self.ciphertext)) + self.ciphertext
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> EnvelopeEncrypted:
        """Inverse of :meth:`to_bytes`."""
        offset = 0
        (version,) = struct.unpack(">I", payload[offset : offset + 4])
        offset += 4
        if version != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol version {version}; expected {PROTOCOL_VERSION}"
            )
        customer_pubkey = payload[offset : offset + X25519_KEY_LENGTH]
        offset += X25519_KEY_LENGTH
        (dek_len,) = struct.unpack(">I", payload[offset : offset + 4])
        offset += 4
        encrypted_dek = payload[offset : offset + dek_len]
        offset += dek_len
        (nonce_len,) = struct.unpack(">I", payload[offset : offset + 4])
        offset += 4
        nonce = payload[offset : offset + nonce_len]
        offset += nonce_len
        (ct_len,) = struct.unpack(">I", payload[offset : offset + 4])
        offset += 4
        ciphertext = payload[offset : offset + ct_len]
        return cls(
            protocol_version=version,
            customer_pubkey=customer_pubkey,
            encrypted_dek=encrypted_dek,
            nonce=nonce,
            ciphertext=ciphertext,
        )


# ============================================================================
# Public functions
# ============================================================================


def envelope_encrypt(
    plaintext: bytes,
    tee_public_key: X25519PublicKey | bytes,
    *,
    associated_data: bytes = b"sablier-flow-v1",
) -> EnvelopeEncrypted:
    """Encrypt ``plaintext`` so only the TEE's matching private key can decrypt.

    Steps:
      1. Generate a fresh customer X25519 ephemeral keypair
      2. Derive a wrap key via X25519(customer_priv, tee_pub) → HKDF-SHA256
      3. Generate a random 256-bit DEK
      4. Encrypt the DEK with AES-GCM(wrap_key, nonce=tee_pub[:12])
      5. Encrypt the payload with AES-GCM(DEK, random_nonce)

    The customer publishes their ephemeral pubkey alongside the encrypted
    DEK; the TEE recovers the wrap key via X25519(tee_priv, customer_pub).

    Parameters
    ----------
    plaintext
        The payload to encrypt (typically Parquet bytes).
    tee_public_key
        The ephemeral X25519 public key extracted from the TEE's
        attestation quote. Pass either an :class:`X25519PublicKey` or
        the raw 32 bytes.
    associated_data
        Bound into the AES-GCM tag for authenticated context. Defaults
        to ``b"sablier-flow-v1"``.

    Returns
    -------
    EnvelopeEncrypted
    """
    if isinstance(tee_public_key, bytes):
        if len(tee_public_key) != X25519_KEY_LENGTH:
            raise ValueError(
                f"TEE public key must be {X25519_KEY_LENGTH} bytes; got {len(tee_public_key)}"
            )
        tee_pub_obj = X25519PublicKey.from_public_bytes(tee_public_key)
        tee_pub_bytes = tee_public_key
    else:
        tee_pub_obj = tee_public_key
        tee_pub_bytes = tee_pub_obj.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    # 1. Ephemeral customer keypair
    customer_priv = X25519PrivateKey.generate()
    customer_pub_bytes = customer_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    # 2. X25519 ECDH + HKDF to derive a 32-byte wrap key
    shared_secret = customer_priv.exchange(tee_pub_obj)
    wrap_key = HKDF(
        algorithm=hashes.SHA256(),
        length=DEK_LENGTH,
        salt=None,
        info=b"sablier-flow envelope wrap v1",
    ).derive(shared_secret)

    # 3. Random DEK
    dek = os.urandom(DEK_LENGTH)

    # 4. Wrap the DEK using AES-GCM with the wrap key.
    # Use a derived deterministic nonce so the wire format doesn't need
    # to carry it (the wrap key is single-use anyway — fresh per envelope).
    wrap_nonce = tee_pub_bytes[:NONCE_LENGTH]
    encrypted_dek = AESGCM(wrap_key).encrypt(wrap_nonce, dek, associated_data)

    # 5. Encrypt the payload
    payload_nonce = os.urandom(NONCE_LENGTH)
    ciphertext = AESGCM(dek).encrypt(payload_nonce, plaintext, associated_data)

    return EnvelopeEncrypted(
        protocol_version=PROTOCOL_VERSION,
        customer_pubkey=customer_pub_bytes,
        encrypted_dek=encrypted_dek,
        nonce=payload_nonce,
        ciphertext=ciphertext,
    )


def envelope_decrypt(
    envelope: EnvelopeEncrypted,
    tee_private_key: X25519PrivateKey,
    *,
    associated_data: bytes = b"sablier-flow-v1",
) -> bytes:
    """Inverse of :func:`envelope_encrypt`. Runs inside the TEE.

    This function is in the client crypto module so the protocol can be
    unit-tested end-to-end without spinning up a TEE — the TEE-side code
    just calls this function with the in-enclave private key.
    """
    tee_pub_obj = tee_private_key.public_key()
    tee_pub_bytes = tee_pub_obj.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    customer_pub_obj = X25519PublicKey.from_public_bytes(envelope.customer_pubkey)
    shared_secret = tee_private_key.exchange(customer_pub_obj)
    wrap_key = HKDF(
        algorithm=hashes.SHA256(),
        length=DEK_LENGTH,
        salt=None,
        info=b"sablier-flow envelope wrap v1",
    ).derive(shared_secret)

    wrap_nonce = tee_pub_bytes[:NONCE_LENGTH]
    dek = AESGCM(wrap_key).decrypt(wrap_nonce, envelope.encrypted_dek, associated_data)

    return AESGCM(dek).decrypt(envelope.nonce, envelope.ciphertext, associated_data)


def load_tee_public_key(raw_bytes: bytes) -> X25519PublicKey:
    """Construct an X25519PublicKey from the 32 raw bytes extracted from
    a TEE attestation quote."""
    if len(raw_bytes) != X25519_KEY_LENGTH:
        raise ValueError(
            f"TEE public key must be {X25519_KEY_LENGTH} bytes; got {len(raw_bytes)}"
        )
    return X25519PublicKey.from_public_bytes(raw_bytes)
