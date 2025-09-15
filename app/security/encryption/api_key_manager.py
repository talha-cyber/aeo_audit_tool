from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class APIKey:
    key_id: str
    secret: str  # plaintext to return once at creation
    hash: str  # stored hash
    salt: str


def _hash_key(secret: str, salt: str, pepper: str) -> str:
    msg = (salt + secret).encode()
    return hmac.new(pepper.encode(), msg, hashlib.sha256).hexdigest()


def generate_api_key() -> APIKey:
    key_id = secrets.token_hex(8)
    secret = secrets.token_urlsafe(32)
    salt = secrets.token_hex(16)
    pepper = getattr(settings, "API_KEY_PEPPER", settings.SECRET_KEY)
    hashed = _hash_key(secret, salt, pepper)
    return APIKey(key_id=key_id, secret=secret, hash=hashed, salt=salt)


def verify_api_key(provided_secret: str, salt: str, stored_hash: str) -> bool:
    pepper = getattr(settings, "API_KEY_PEPPER", settings.SECRET_KEY)
    candidate = _hash_key(provided_secret, salt, pepper)
    return hmac.compare_digest(candidate, stored_hash)
