from __future__ import annotations

from typing import Any, Dict

from app.core.config import settings


def compliance_snapshot() -> Dict[str, Any]:
    return {
        "security_headers": getattr(settings, "ENABLE_SECURITY_HEADERS", True),
        "jwt_algorithm": getattr(settings, "JWT_ALGORITHM", "HS256"),
        "encryption_enabled": bool(
            getattr(settings, "FIELD_ENCRYPTION_KEY", settings.SECRET_KEY)
        ),
        "session_ttl": getattr(settings, "SESSION_TTL_SECONDS", 3600),
    }
