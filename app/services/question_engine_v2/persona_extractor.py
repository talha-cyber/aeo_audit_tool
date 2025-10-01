"""Persona resolution utilities for question engine v2."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, MutableMapping, Optional, Sequence, Type, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from app.services.question_engine_v2.schemas import (
    ContextCatalogItem,
    DriverCatalogItem,
    PersonaCatalog,
    PersonaMode,
    PersonaResolution,
    PersonaSelection,
    RoleCatalogItem,
    VoiceCatalogItem,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

TCatalog = TypeVar("TCatalog", bound=BaseModel)

CATALOG_FILENAMES: Dict[PersonaMode, Dict[str, str]] = {
    PersonaMode.B2C: {
        "roles": "b2c_roles.yaml",
        "drivers": "b2c_drivers.yaml",
        "contexts": "b2c_contexts.yaml",
        "voices": "b2c_voices.yaml",
    },
    PersonaMode.B2B: {
        "roles": "b2b_roles.yaml",
        "drivers": "b2b_drivers.yaml",
        "contexts": "b2b_contexts.yaml",
        "voices": "b2b_voices.yaml",
    },
}


class PersonaCatalogError(RuntimeError):
    """Raised when persona catalogs fail to load or validate."""


class PersonaExtractor:
    """Loads persona catalogs and resolves persona presets/overrides."""

    def __init__(self, base_path: Optional[Path] = None) -> None:
        self._base_path = base_path or Path(__file__).parent / "catalogs"
        self._catalog_cache: Dict[PersonaMode, PersonaCatalog] = {}
        logger.debug("PersonaExtractor initialized", base_path=str(self._base_path))

    def available_modes(self) -> List[str]:
        """Return the persona modes supported by the extractor."""
        return [mode.value for mode in CATALOG_FILENAMES]

    def load_catalog(self, mode: PersonaMode | str) -> PersonaCatalog:
        """Load and validate persona catalog bundle for the given mode."""
        mode_enum = self._ensure_mode(mode)
        if mode_enum in self._catalog_cache:
            return self._catalog_cache[mode_enum]

        filenames = CATALOG_FILENAMES.get(mode_enum)
        if not filenames:
            raise PersonaCatalogError(f"Unsupported persona mode: {mode}")

        logger.debug("Loading persona catalogs", mode=mode_enum.value)
        raw_roles = self._load_yaml(filenames["roles"])
        raw_drivers = self._load_yaml(filenames["drivers"])
        raw_contexts = self._load_yaml(filenames["contexts"])
        raw_voices = self._load_yaml(filenames["voices"])

        try:
            catalog = PersonaCatalog(
                mode=mode_enum,
                roles=self._build_catalog(raw_roles, RoleCatalogItem),
                drivers=self._build_catalog(raw_drivers, DriverCatalogItem),
                contexts=self._build_catalog(raw_contexts, ContextCatalogItem),
                voices=self._build_catalog(raw_voices, VoiceCatalogItem),
            )
        except ValidationError as exc:
            logger.error(
                "Persona catalog validation failed",
                mode=mode_enum.value,
                error=str(exc),
            )
            raise PersonaCatalogError("Invalid persona catalog format") from exc

        self._validate_catalog(catalog)
        self._catalog_cache[mode_enum] = catalog
        return catalog

    def resolve_personas(
        self,
        mode: PersonaMode | str,
        voices: Optional[Sequence[str]] = None,
        selections: Optional[
            Sequence[PersonaSelection | MutableMapping[str, object]]
        ] = None,
    ) -> List[PersonaResolution]:
        """Resolve personas from presets and explicit selections."""
        catalog = self.load_catalog(mode)
        resolved: List[PersonaResolution] = []
        seen: set[tuple] = set()

        missing_voices: List[str] = []

        if voices:
            for voice_name in voices:
                preset = catalog.voices.get(voice_name)
                if not preset:
                    missing_voices.append(voice_name)
                    logger.warning(
                        "Persona voice not found; skipping",
                        voice=voice_name,
                        mode=catalog.mode.value,
                    )
                    continue
                persona = self._persona_from_preset(catalog, voice_name, preset)
                fingerprint = self._fingerprint(persona)
                if fingerprint not in seen:
                    resolved.append(persona)
                    seen.add(fingerprint)

        if selections:
            for selection in selections:
                if isinstance(selection, PersonaSelection):
                    spec = selection
                else:
                    spec = PersonaSelection(**selection)
                persona = self._persona_from_selection(catalog, spec)
                if persona is None:
                    continue
                fingerprint = self._fingerprint(persona)
                if fingerprint not in seen:
                    resolved.append(persona)
                    seen.add(fingerprint)

        if missing_voices:
            logger.info(
                "Skipped unresolved persona voices",
                mode=catalog.mode.value,
                voices=missing_voices,
            )

        if not resolved:
            logger.warning("No personas resolved for mode", mode=catalog.mode.value)
        return resolved

    def export_catalog(self, mode: PersonaMode | str) -> Dict[str, Dict[str, dict]]:
        """Return raw catalog data payload for API usage."""
        catalog = self.load_catalog(mode)
        return {
            "mode": catalog.mode.value,
            "roles": {key: item.model_dump() for key, item in catalog.roles.items()},
            "drivers": {
                key: item.model_dump() for key, item in catalog.drivers.items()
            },
            "contexts": {
                key: item.model_dump() for key, item in catalog.contexts.items()
            },
            "voices": {key: item.model_dump() for key, item in catalog.voices.items()},
        }

    def clear_cache(self) -> None:
        """Reset in-memory catalog cache (useful for tests)."""
        self._catalog_cache.clear()
        logger.debug("PersonaExtractor cache cleared")

    def _persona_from_preset(
        self, catalog: PersonaCatalog, voice_name: str, preset: VoiceCatalogItem
    ) -> PersonaResolution:
        contexts = self._resolve_contexts(catalog, preset.contexts, preset.role)
        emotional_anchor = (
            catalog.drivers.get(preset.driver).emotional_anchor
            if preset.driver in catalog.drivers
            else None
        )
        return PersonaResolution(
            mode=catalog.mode,
            role=preset.role,
            driver=preset.driver,
            contexts=contexts,
            voice=voice_name,
            emotional_anchor=emotional_anchor,
        )

    def _persona_from_selection(
        self, catalog: PersonaCatalog, selection: PersonaSelection
    ) -> Optional[PersonaResolution]:
        if selection.role not in catalog.roles:
            logger.warning(
                "Persona selection role not found; skipping",
                role=selection.role,
                mode=catalog.mode.value,
            )
            return None
        if selection.driver not in catalog.drivers:
            logger.warning(
                "Persona selection driver not found; skipping",
                driver=selection.driver,
                mode=catalog.mode.value,
            )
            return None
        contexts = self._resolve_contexts(catalog, selection.contexts, selection.role)
        emotional_anchor = catalog.drivers[selection.driver].emotional_anchor
        return PersonaResolution(
            mode=catalog.mode,
            role=selection.role,
            driver=selection.driver,
            contexts=contexts,
            voice=selection.voice,
            emotional_anchor=emotional_anchor,
        )

    def _resolve_contexts(
        self, catalog: PersonaCatalog, contexts: Sequence[str], role: str
    ) -> List[str]:
        resolved: List[str] = []
        for ctx in contexts:
            if ctx not in catalog.contexts:
                logger.warning(
                    "Persona context not found; skipping",
                    context=ctx,
                    mode=catalog.mode.value,
                    role=role,
                )
                continue
            resolved.append(ctx)
        if resolved:
            return resolved

        default_ctx = (
            catalog.roles.get(role).default_context_stage
            if role in catalog.roles
            else None
        )
        if default_ctx and default_ctx in catalog.contexts:
            logger.info(
                "Falling back to role default context",
                role=role,
                context=default_ctx,
                mode=catalog.mode.value,
            )
            return [default_ctx]

        fallback = next(iter(catalog.contexts.keys()), None)
        if fallback:
            logger.info(
                "Falling back to first available context",
                role=role,
                context=fallback,
                mode=catalog.mode.value,
            )
            return [fallback]

        logger.error(
            "No contexts available for persona role",
            role=role,
            mode=catalog.mode.value,
        )
        raise PersonaCatalogError(
            f"No contexts defined for role '{role}' in mode {catalog.mode.value}"
        )

    def _build_catalog(self, raw: dict, model: Type[TCatalog]) -> Dict[str, TCatalog]:
        if not isinstance(raw, dict):
            raise PersonaCatalogError(
                "Catalog file must contain a mapping at top level"
            )
        built: Dict[str, TCatalog] = {}
        for key, value in raw.items():
            try:
                built[key] = model(**value)
            except ValidationError as exc:
                raise PersonaCatalogError(f"Invalid catalog entry for '{key}'") from exc
        return built

    def _validate_catalog(self, catalog: PersonaCatalog) -> None:
        missing_roles: List[str] = []
        missing_drivers: List[str] = []
        missing_contexts: Dict[str, List[str]] = {}
        for voice_name, voice in catalog.voices.items():
            if voice.role not in catalog.roles:
                missing_roles.append(f"{voice_name}:{voice.role}")
            if voice.driver not in catalog.drivers:
                missing_drivers.append(f"{voice_name}:{voice.driver}")
            invalid = [ctx for ctx in voice.contexts if ctx not in catalog.contexts]
            if invalid:
                missing_contexts[voice_name] = invalid
        if missing_roles or missing_drivers or missing_contexts:
            logger.error(
                "Persona catalog consistency error",
                mode=catalog.mode.value,
                missing_roles=missing_roles,
                missing_drivers=missing_drivers,
                missing_contexts=missing_contexts,
            )
            raise PersonaCatalogError("Persona catalog references unknown entities")

    def _load_yaml(self, filename: str) -> dict:
        path = self._base_path / filename
        if not path.exists():
            raise PersonaCatalogError(f"Catalog file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise PersonaCatalogError(f"Unexpected structure in catalog file {path}")
        return data

    def _fingerprint(self, persona: PersonaResolution) -> tuple:
        return (
            persona.mode.value,
            persona.role,
            persona.driver,
            tuple(persona.contexts),
            persona.voice,
        )

    def _ensure_mode(self, mode: PersonaMode | str) -> PersonaMode:
        if isinstance(mode, PersonaMode):
            return mode
        try:
            return PersonaMode(mode)
        except ValueError as exc:
            raise PersonaCatalogError(f"Invalid persona mode '{mode}'") from exc


__all__ = [
    "PersonaExtractor",
    "PersonaCatalogError",
]
