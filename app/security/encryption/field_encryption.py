from __future__ import annotations

import base64
from typing import Optional

from cryptography.fernet import Fernet  # type: ignore[import-not-found]

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
