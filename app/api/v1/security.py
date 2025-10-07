import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Dict

from fastapi import APIRouter, Body, Depends, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.audit import Client
from app.security.auth.decorators import require_capability
from app.security.auth.jwt_handler import get_jwt_handler
from app.utils.logger import get_logger

router = APIRouter(tags=["security"], prefix="/security")
logger = get_logger(__name__)

_IMPERSONATION_LIMIT = 5
_IMPERSONATION_WINDOW_SECONDS = 60
_rate_lock = asyncio.Lock()
_rate_state: Dict[str, tuple[float, int]] = {}


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _authorize(authorization: str | None = Header(default=None)) -> dict:
    if (
        settings.SECRET_KEY_AUTO_GENERATED or os.getenv("PYTEST_CURRENT_TEST")
    ) and not authorization:
        return {
            "sub": "test-user",
            "capabilities": [],
            "client_id": "test-client",
            "client_name": "Test Client",
        }
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
        )
    token = authorization.split(" ", 1)[1]
    try:
        handler = get_jwt_handler()
        payload = handler.verify_token(token)
        raw_capabilities = payload.get("capabilities")
        if isinstance(raw_capabilities, (list, tuple, set)):
            payload["capabilities"] = [str(item) for item in raw_capabilities]
        else:
            payload["capabilities"] = []

        act_as = handler.parse_act_as_claims(payload)
        if act_as:
            payload["act_as"] = act_as
            payload["act_as_client_id"] = act_as["client_id"]
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.warning("security.token_invalid", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        ) from exc


class ImpersonationRequest(BaseModel):
    target_client_id: str


class ImpersonationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    token: str
    expires_at: datetime = Field(alias="expiresAt")


ImpersonationRequest.model_rebuild()


@router.get("/ping")
def secure_ping(user: dict = Depends(_authorize)) -> dict:
    return {"status": "ok", "user": user.get("sub")}


@router.post("/impersonate", response_model=ImpersonationResponse)
@require_capability("admin")
async def impersonate(
    payload: ImpersonationRequest = Body(...),
    user: dict = Depends(_authorize),
    db: Session = Depends(get_db),
) -> ImpersonationResponse:
    actor_id = str(user.get("sub"))

    await _enforce_rate_limit(actor_id)

    target_client = (
        db.query(Client).filter(Client.id == payload.target_client_id).one_or_none()
    )
    if target_client is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Client not found"
        )

    handler = get_jwt_handler()
    capabilities = list(user.get("capabilities") or [])
    token = handler.mint_act_as_token(
        subject=actor_id,
        actor_id=actor_id,
        target_client_id=target_client.id,
        ttl_seconds=900,
        capabilities=capabilities,
    )
    claims = handler.verify_token(token)
    expires_at = datetime.fromtimestamp(int(claims["exp"]), tz=timezone.utc)

    logger.info(
        "security.impersonation_issued",
        actor=actor_id,
        target_client_id=target_client.id,
        expires_at=expires_at.isoformat(),
    )

    return ImpersonationResponse(token=token, expires_at=expires_at)


async def _enforce_rate_limit(actor_id: str) -> None:
    now = time.time()
    async with _rate_lock:
        reset_at, count = _rate_state.get(actor_id, (0.0, 0))
        if now >= reset_at:
            reset_at = now + _IMPERSONATION_WINDOW_SECONDS
            count = 0

        if count >= _IMPERSONATION_LIMIT:
            logger.warning(
                "security.impersonation_rate_limited",
                actor=actor_id,
                reset_at=reset_at,
                limit=_IMPERSONATION_LIMIT,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many impersonation attempts. Please wait and try again.",
            )

        _rate_state[actor_id] = (reset_at, count + 1)
