"""Client-side AMD SEV-SNP attestation verifier.

Validates a SEV-SNP attestation report end-to-end against AMD's
publicly-published root keys, baked into the SDK at
``sablier_flow/_amd_roots/``.

Chain of trust:

    AMD ARK (root, self-signed)
       ↓ signs
    AMD ASK (signing key for the CPU family)
       ↓ signs
    VCEK (Versioned Chip Endorsement Key — per-CPU)
       ↓ signs
    SEV-SNP attestation report
       └─ contains REPORT_DATA bound to TEE's ephemeral X25519 pubkey

If every link verifies AND the REPORT_DATA matches the ephemeral pubkey
the client received in the attestation quote, then:

  1. The report was minted by a real AMD CPU (ARK is the AMD root).
  2. The CPU was running in SEV-SNP mode at the moment of attestation.
  3. The TEE's ephemeral pubkey was generated INSIDE that same SEV-SNP
     enclave (the REPORT_DATA field binds them cryptographically).
  4. Therefore: encrypting customer data to that pubkey ships it into
     real confidential memory, not impostor cleartext memory.

Spec references:
  - AMD SEV-SNP ABI 56860 chapter 7 (report format)
  - https://www.amd.com/en/developer/sev.html (root cert chain)
"""

from __future__ import annotations

import importlib.resources
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "SEV_REPORT_SIZE",
    "SevSnpReport",
    "SevSnpVerificationError",
    "verify_sev_snp_report",
]


SEV_REPORT_SIZE = 1184      # bytes — fixed by AMD spec
SEV_USER_DATA_OFFSET = 80   # offset of REPORT_DATA inside the report
SEV_USER_DATA_SIZE = 64
SEV_SIG_OFFSET = SEV_REPORT_SIZE - 512  # last 512 bytes are the signature blob


class SevSnpVerificationError(Exception):
    """Raised when an AMD SEV-SNP attestation report fails verification."""


@dataclass(frozen=True)
class SevSnpReport:
    """Parsed AMD SEV-SNP attestation report.

    The raw report is the 1184-byte binary structure from AMD firmware.
    ``user_data`` is the customer-readable REPORT_DATA field — used to
    bind the report to the TEE's ephemeral X25519 pubkey.
    """

    version: int
    raw: bytes
    user_data: bytes        # REPORT_DATA, 64 bytes
    measurement: bytes      # MEASUREMENT field (48 bytes), launch measurement of the guest
    policy: int             # POLICY, uint64

    @classmethod
    def from_bytes(cls, raw: bytes) -> SevSnpReport:
        if len(raw) != SEV_REPORT_SIZE:
            raise SevSnpVerificationError(
                f"report size {len(raw)} != expected {SEV_REPORT_SIZE}"
            )
        (version,) = struct.unpack_from("<I", raw, 0)
        if version not in (2, 5):
            # v2 = older AMD spec, v5 = current. Other versions we don't
            # promise to understand; reject conservatively.
            raise SevSnpVerificationError(
                f"unsupported SEV-SNP report version {version} (expected 2 or 5)"
            )
        (policy,) = struct.unpack_from("<Q", raw, 8)
        user_data = raw[SEV_USER_DATA_OFFSET : SEV_USER_DATA_OFFSET + SEV_USER_DATA_SIZE]
        # MEASUREMENT field: 48 bytes at a version-dependent offset.
        # For both v2 and v5 it sits at offset 144 per AMD spec 56860 §7.3.
        measurement = raw[144 : 144 + 48]
        return cls(
            version=version,
            raw=raw,
            user_data=user_data,
            measurement=measurement,
            policy=policy,
        )


# ============================================================================
# Verifier
# ============================================================================


def verify_sev_snp_report(
    *,
    report_bytes: bytes,
    cert_chain: bytes,
    expected_user_data: bytes,
    require_amd_root: bool = True,
) -> SevSnpReport:
    """Verify a SEV-SNP attestation report against AMD's root chain.

    Parameters
    ----------
    report_bytes
        The 1184-byte binary report from the TEE.
    cert_chain
        Concatenated DER/PEM-encoded VCEK + ASK + ARK certs the TEE
        included alongside the report. Production TEEs put this in the
        ``auxblob`` from ``/sys/kernel/config/tsm/report/<name>/auxblob``.
    expected_user_data
        Up to 64 bytes the client expects to be pinned in REPORT_DATA.
        By convention this is the TEE's ephemeral X25519 public key
        bytes (left-padded to 64 bytes). If this doesn't match, the
        report doesn't belong to this handshake — reject.
    require_amd_root
        If True (production), the VCEK→ASK→ARK chain must terminate at
        the AMD-published Milan ARK we ship with the SDK. If False
        (debugging only), the chain is parsed but not cryptographically
        anchored to AMD's root — useful for staging where the TEE may
        be using a self-signed test chain.

    Returns
    -------
    SevSnpReport
        Parsed report. The fact that this returns at all means every
        link in the chain verified.

    Raises
    ------
    SevSnpVerificationError
        If any link fails (size mismatch, version unsupported,
        REPORT_DATA mismatch, VCEK signature invalid, VCEK not chained
        to AMD root, etc.).
    """
    # --- 1. Parse the report --------------------------------------------
    report = SevSnpReport.from_bytes(report_bytes)

    # --- 2. Bind: REPORT_DATA must match what the client expects --------
    padded_expected = expected_user_data + b"\x00" * (
        SEV_USER_DATA_SIZE - len(expected_user_data)
    )
    if report.user_data != padded_expected:
        raise SevSnpVerificationError(
            "REPORT_DATA in the SEV-SNP report does not match the expected "
            "ephemeral pubkey — the report doesn't belong to this handshake. "
            f"expected={padded_expected[:16].hex()}..., "
            f"got={report.user_data[:16].hex()}..."
        )

    # --- 3. Parse the cert chain (VCEK, ASK, ARK) ----------------------
    vcek_cert, ask_cert, ark_cert = _parse_cert_chain(cert_chain)

    # --- 4. Verify VCEK signed the report ------------------------------
    _verify_report_signature(report_bytes, vcek_cert)

    # --- 5. Verify ASK signed the VCEK ---------------------------------
    _verify_cert_signed_by(vcek_cert, ask_cert)

    # --- 6. Verify ARK signed the ASK ----------------------------------
    _verify_cert_signed_by(ask_cert, ark_cert)

    # --- 7. Pin ARK to AMD's published Milan root ----------------------
    if require_amd_root:
        amd_ark = _load_amd_milan_ark()
        # Compare by public-key bytes (cert serials can drift on rotation).
        if _pubkey_der(ark_cert) != _pubkey_der(amd_ark):
            raise SevSnpVerificationError(
                "ARK in the attestation chain does not match the AMD Milan root "
                "key baked into this SDK release. Either the TEE is using an "
                "unknown CPU family (we only pin Milan today) or the chain has "
                "been tampered with."
            )

    return report


# ============================================================================
# Internals — cert parsing + signature verification
# ============================================================================


# AMD's published GUIDs identifying each cert kind in the auxblob.
# Mixed-endian per RFC 4122 § 4.1.2.
_GUID_VCEK_BYTES = b"\x63\xda\x75\x8d\xe6\x64\x45\x64\xad\xc5\xf4\xb9\x3b\xe8\xac\xcd"
_GUID_ASK_BYTES = b"\x4a\xb7\xb3\x79\xbb\xac\x4f\xe4\xa0\x2f\x05\xae\xf3\x27\xc7\x82"
_GUID_ARK_BYTES = b"\xc0\xb4\x06\xa4\xa8\x03\x49\x52\x97\x43\x3f\xb6\x01\x4c\xd0\xae"


def _parse_cert_chain(blob: bytes) -> tuple[Any, Any, Any]:
    """Split the auxblob into (VCEK, ASK, ARK) certificates.

    The Linux TSM ``auxblob`` is in AMD's GUID-table format, not a flat
    cert concatenation. Layout::

        [Entry 1]  16-byte GUID │ 4-byte offset │ 4-byte length
        [Entry 2]  16-byte GUID │ 4-byte offset │ 4-byte length
        [Entry N]   ...                    ...
        [Padding]  zeros up to first cert
        [Cert 1]   DER bytes (from Entry-1 offset, Entry-1 length)
        [Cert 2]   DER bytes
        ...

    Each entry's GUID identifies what kind of cert it points to —
    VCEK, ASK, or ARK. We index by GUID rather than guessing by
    position or CN, since the order and presence of optional
    entries varies by firmware version.

    Falls back to the legacy "PEM-bundle" format if the first bytes
    don't look like a GUID table.
    """
    from cryptography import x509

    # Fast path: legacy PEM-bundle format
    if b"BEGIN CERTIFICATE" in blob:
        pem_certs: list[Any] = []
        for m in re.findall(
            rb"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
            blob,
            re.DOTALL,
        ):
            pem_certs.append(x509.load_pem_x509_certificate(m))
        return _identify_by_cn_or_chain(pem_certs)

    # AMD GUID-table format
    by_guid: dict[bytes, Any] = {}
    offset = 0
    while offset + 24 <= len(blob):
        guid = blob[offset : offset + 16]
        # All-zero GUID = end of table
        if guid == b"\x00" * 16:
            break
        cert_offset = int.from_bytes(blob[offset + 16 : offset + 20], "little")
        cert_length = int.from_bytes(blob[offset + 20 : offset + 24], "little")
        if cert_offset > 0 and cert_length > 0 and cert_offset + cert_length <= len(blob):
            try:
                cert = x509.load_der_x509_certificate(
                    blob[cert_offset : cert_offset + cert_length]
                )
                by_guid[guid] = cert
            except Exception:
                pass  # malformed entry — skip
        offset += 24

    vcek = by_guid.get(_GUID_VCEK_BYTES)
    ask = by_guid.get(_GUID_ASK_BYTES)
    ark = by_guid.get(_GUID_ARK_BYTES)

    if vcek is None or ask is None or ark is None:
        raise SevSnpVerificationError(
            f"auxblob missing one or more certs (VCEK={vcek is not None}, "
            f"ASK={ask is not None}, ARK={ark is not None}). GUIDs seen: "
            f"{[g.hex() for g in by_guid]}"
        )
    return vcek, ask, ark


def _identify_by_cn_or_chain(certs: list[Any]) -> tuple[Any, Any, Any]:
    """Legacy PEM-bundle path: identify certs by Common Name + self-signed."""
    if len(certs) < 3:
        raise SevSnpVerificationError(
            f"expected at least 3 certs in the chain; got {len(certs)}"
        )
    vcek = ask = ark = None
    for c in certs:
        cn = c.subject.rfc4514_string().upper()
        issuer = c.issuer.rfc4514_string().upper()
        if "VCEK" in cn:
            vcek = c
        elif "ARK" in cn and cn == issuer:  # ARK is self-signed
            ark = c
        elif "ASK" in cn or "SEV-MILAN" in cn:
            ask = c
    # Fallback by chain order
    if vcek is None:
        vcek = certs[0]
    if ask is None and len(certs) >= 2:
        ask = certs[1]
    if ark is None and len(certs) >= 3:
        ark = certs[2]
    return vcek, ask, ark


def _verify_report_signature(report_bytes: bytes, vcek_cert: Any) -> None:
    """Verify the VCEK's ECDSA signature over the report body.

    AMD SEV-SNP reports are signed with ECDSA-P384 (SHA-384). The
    signature occupies the last 512 bytes of the report (the format
    reserves space for ECDSA-P521 which the spec doesn't use yet).
    Within those 512 bytes:
      - r: 72 bytes (P-384 is actually 48 bytes, padded)
      - s: 72 bytes
      - remainder: zeros
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

    body = report_bytes[:SEV_SIG_OFFSET]
    sig_blob = report_bytes[SEV_SIG_OFFSET:]

    # AMD packs r and s as LITTLE-ENDIAN P-384 (48 bytes each, padded to 72).
    # Convert to int + then to standard DSS encoding for the cryptography lib.
    r = int.from_bytes(sig_blob[0:48], "little")
    s = int.from_bytes(sig_blob[72:120], "little")
    dss_sig = encode_dss_signature(r, s)

    pub = vcek_cert.public_key()
    if not isinstance(pub, ec.EllipticCurvePublicKey):
        raise SevSnpVerificationError("VCEK is not an ECDSA key")

    try:
        pub.verify(dss_sig, body, ec.ECDSA(hashes.SHA384()))
    except InvalidSignature as exc:
        raise SevSnpVerificationError(
            "ECDSA verification of the SEV-SNP report signature against the VCEK FAILED"
        ) from exc


def _verify_cert_signed_by(child: Any, parent: Any) -> None:
    """Verify ``parent`` issued + signed ``child``."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.asymmetric.ec import ECDSA, EllipticCurvePublicKey

    parent_pub = parent.public_key()
    try:
        if isinstance(parent_pub, EllipticCurvePublicKey):
            parent_pub.verify(
                child.signature,
                child.tbs_certificate_bytes,
                ECDSA(child.signature_hash_algorithm),
            )
        elif isinstance(parent_pub, rsa.RSAPublicKey):
            # AMD ARK + ASK are RSA-PSS (4096-bit).
            parent_pub.verify(
                child.signature,
                child.tbs_certificate_bytes,
                padding.PSS(
                    mgf=padding.MGF1(child.signature_hash_algorithm),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                child.signature_hash_algorithm,
            )
        else:
            raise SevSnpVerificationError(
                f"unsupported parent key type: {type(parent_pub).__name__}"
            )
    except InvalidSignature as exc:
        child_cn = child.subject.rfc4514_string()
        parent_cn = parent.subject.rfc4514_string()
        raise SevSnpVerificationError(
            f"cert chain link broken: {parent_cn} did NOT sign {child_cn}"
        ) from exc


def _load_amd_milan_ark() -> Any:
    """Load the AMD Milan ARK from the baked-in PEM bundle."""
    from cryptography import x509

    pkg = importlib.resources.files("sablier_flow") / "_amd_roots" / "milan_ask_ark.pem"
    data = pkg.read_bytes()
    matches = re.findall(
        rb"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
        data,
        re.DOTALL,
    )
    # The bundle is [ASK, ARK]. ARK is the self-signed one — pick it that way.
    for m in matches:
        c = x509.load_pem_x509_certificate(m)
        if c.subject == c.issuer:
            return c
    raise SevSnpVerificationError("no self-signed cert (ARK) found in baked-in chain")


def _pubkey_der(cert: Any) -> bytes:
    """Serialize a cert's public key to DER bytes for equality comparison."""
    from cryptography.hazmat.primitives import serialization

    der: bytes = cert.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return der


# Suppress unused-import warning until tests import this.
_ = (Path,)
