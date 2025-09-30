"""Persona catalog API endpoints."""

from fastapi import APIRouter, HTTPException, Query

from app.services.question_engine_v2.persona_extractor import (
    PersonaCatalogError,
    PersonaExtractor,
)
from app.services.question_engine_v2.schemas import PersonaMode

router = APIRouter(prefix="/personas", tags=["personas"])
_extractor = PersonaExtractor()


@router.get("/catalog")
async def get_persona_catalog(
    mode: str = Query(default=PersonaMode.B2C.value),
) -> dict:
    """Return persona catalog bundle for the requested mode."""
    try:
        persona_mode = PersonaMode(mode)
    except ValueError as exc:
        detail = f"Unsupported persona mode: {mode}"
        raise HTTPException(status_code=400, detail=detail) from exc

    try:
        catalog = _extractor.export_catalog(persona_mode)
    except PersonaCatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return catalog


@router.get("/modes")
async def list_persona_modes() -> dict:
    """List available persona catalog modes."""
    return {"modes": _extractor.available_modes()}
