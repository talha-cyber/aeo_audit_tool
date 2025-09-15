from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.core.config import settings


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


@dataclass
class JWTConfig:
    algorithm: str = "HS256"
    expires_minutes: int = 60
    issuer: Optional[str] = None


class JWTHandler:
    """
    Minimal JWT HS256 implementation without external dependencies.
    For production, ensure SECRET_KEY is strong and rotated regularly.
    """

    def __init__(self, secret: str, config: Optional[JWTConfig] = None) -> None:
        self.secret = secret.encode()
        self.config = config or JWTConfig(
            algorithm=getattr(settings, "JWT_ALGORITHM", "HS256"),
            expires_minutes=getattr(settings, "JWT_EXPIRES_MINUTES", 60),
            issuer=getattr(settings, "JWT_ISSUER", None),
        )

    def create_token(
        self, subject: str, claims: Optional[Dict[str, Any]] = None
    ) -> str:
        header = {"alg": self.config.algorithm, "typ": "JWT"}
        now = int(time.time())
        payload = {
            "sub": subject,
            "iat": now,
            "exp": now + (self.config.expires_minutes * 60),
        }
        if self.config.issuer:
            payload["iss"] = self.config.issuer
        if claims:
            payload.update(claims)

        header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
        payload_b64 = _b64url_encode(
            json.dumps(payload, separators=(",", ":")).encode()
        )
        signing_input = f"{header_b64}.{payload_b64}".encode()
        signature = hmac.new(self.secret, signing_input, hashlib.sha256).digest()
        signature_b64 = _b64url_encode(signature)
        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def verify_token(self, token: str) -> Dict[str, Any]:
        try:
            header_b64, payload_b64, signature_b64 = token.split(".")
            signing_input = f"{header_b64}.{payload_b64}".encode()
            expected = hmac.new(self.secret, signing_input, hashlib.sha256).digest()
            if not hmac.compare_digest(expected, _b64url_decode(signature_b64)):
                raise ValueError("Invalid signature")

            payload = json.loads(_b64url_decode(payload_b64))
            now = int(time.time())
            if payload.get("exp") and now > int(payload["exp"]):
                raise ValueError("Token expired")
            if self.config.issuer and payload.get("iss") != self.config.issuer:
                raise ValueError("Invalid issuer")
            return payload
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Invalid token: {e}")


def get_jwt_handler() -> JWTHandler:
    return JWTHandler(settings.SECRET_KEY)
