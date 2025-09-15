from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.security.auth.jwt_handler import get_jwt_handler

router = APIRouter(tags=["security"], prefix="/secure")


def _authorize(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
        )
    token = authorization.split(" ", 1)[1]
    try:
        payload = get_jwt_handler().verify_token(token)
        return payload
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )


@router.get("/ping")
def secure_ping(user: dict = Depends(_authorize)) -> dict:
    return {"status": "ok", "user": user.get("sub")}
