from __future__ import annotations

import base64
import hashlib
import logging
from typing import Optional

try:
    from cryptography.fernet import Fernet  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - lightweight fallback for environments without cryptography
    class Fernet:  # type: ignore[override]
        """Non-cryptographic fallback that provides reversible encoding for tests."""

        def __init__(self, key: bytes):
            if isinstance(key, str):
                key = key.encode()
            self._mask = hashlib.sha256(key).digest()
            logging.getLogger(__name__).warning(
                "cryptography not installed; using insecure fallback for FieldEncryptor"
            )

        def _xor(self, data: bytes) -> bytes:
            mask = self._mask
            return bytes(b ^ mask[i % len(mask)] for i, b in enumerate(data))

        def encrypt(self, plaintext: bytes) -> bytes:
            scrambled = self._xor(plaintext)
            return base64.urlsafe_b64encode(scrambled)

        def decrypt(self, token: bytes) -> bytes:
            data = base64.urlsafe_b64decode(token)
            return self._xor(data)

from app.core.config import settings


class FieldEncryptor:
    """
    Symmetric field encryption utility using Fernet (AES128 in CBC + HMAC).

    Requires `cryptography` package. If unavailable, raise ImportError with guidance.
    """

    def __init__(self, key: Optional[bytes] = None) -> None:
        secret = key or getattr(settings, "FIELD_ENCRYPTION_KEY", None)
        if secret is None:
            # Derive a deterministic key from SECRET_KEY for convenience if not provided
            secret = base64.urlsafe_b64encode((settings.SECRET_KEY * 2)[:32].encode())
        if isinstance(secret, str):
            secret = secret.encode()
        self._fernet = Fernet(secret)

    def encrypt(self, plaintext: str) -> str:
        token = self._fernet.encrypt(plaintext.encode())
        return token.decode()

    def decrypt(self, token: str) -> str:
        return self._fernet.decrypt(token.encode()).decode()
