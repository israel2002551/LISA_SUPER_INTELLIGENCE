"""ECDSA P-256 signatures over OTA manifests (verified by edge/ota_verify.h).

Signed message: "W25OTA1|<version>|<image size>|<image sha256 hex>"
Signature: base64 of the DER encoding.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

log = logging.getLogger(__name__)


def manifest_message(version: int, size: int, sha256_hex: str) -> bytes:
    return f"W25OTA1|{version}|{size}|{sha256_hex}".encode()


class ManifestSigner:
    def __init__(self, private_key: ec.EllipticCurvePrivateKey) -> None:
        if not isinstance(private_key.curve, ec.SECP256R1):
            raise ValueError("OTA signing key must be ECDSA P-256 (secp256r1)")
        self._key = private_key

    @classmethod
    def generate(cls) -> "ManifestSigner":
        return cls(ec.generate_private_key(ec.SECP256R1()))

    @classmethod
    def from_pem(cls, pem: bytes) -> "ManifestSigner":
        key = serialization.load_pem_private_key(pem, password=None)
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise ValueError("OTA signing key must be an EC private key")
        return cls(key)

    @classmethod
    def load_or_create(cls, path: Path) -> "ManifestSigner":
        """Dev mode: keep a generated key next to the local data."""
        if path.is_file():
            return cls.from_pem(path.read_bytes())
        signer = cls.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(signer.private_pem())
        log.warning("Generated a dev OTA signing key at %s. Put its public key in edge/secrets.h:\n%s",
                    path, signer.public_pem().decode())
        return signer

    def sign(self, version: int, size: int, sha256_hex: str) -> str:
        der = self._key.sign(manifest_message(version, size, sha256_hex), ec.ECDSA(hashes.SHA256()))
        return base64.b64encode(der).decode()

    def verify(self, version: int, size: int, sha256_hex: str, signature_b64: str) -> bool:
        try:
            self._key.public_key().verify(base64.b64decode(signature_b64),
                                          manifest_message(version, size, sha256_hex), ec.ECDSA(hashes.SHA256()))
            return True
        except (InvalidSignature, ValueError):
            return False

    def private_pem(self) -> bytes:
        return self._key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption())

    def public_pem(self) -> bytes:
        return self._key.public_key().public_bytes(serialization.Encoding.PEM,
                                                   serialization.PublicFormat.SubjectPublicKeyInfo)

    def public_pem_c_string(self) -> str:
        """The public key as a C string literal for EDGE_OTA_PUBLIC_KEY_PEM."""
        lines = self.public_pem().decode().strip().splitlines()
        return " \\\n    ".join(f'"{line}\\n"' for line in lines)
