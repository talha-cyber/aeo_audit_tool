"""Persona catalogue flattening and persistence for dashboard consumption."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from sqlalchemy.orm import Session

from app.api.v1.dashboard_schemas import (
    PersonaCatalogContextView,
    PersonaCatalogDriverView,
    PersonaCatalogRoleView,
    PersonaCatalogView,
    PersonaCatalogVoiceView,
    PersonaLibraryEntryView,
    PersonaLibraryResponse,
    PersonaStageView,
    PersonaView,
)
from app.api.v1.dashboard_schemas import PersonaComposeRequest, PersonaUpdateRequest
from app.services.dashboard.persona_store import PersonaLibraryStore, PersonaRecord
from app.services.dashboard.static_data import default_personas
from app.services.question_engine_v2.persona_extractor import (
    PersonaCatalogError,
    PersonaExtractor,
)
from app.services.question_engine_v2.schemas import (
    PersonaCatalog,
    PersonaMode,
    PersonaResolution,
    PersonaSelection,
)

__all__ = [
    "export_persona_catalog",
    "list_personas",
    "list_persona_library",
    "create_persona",
    "update_persona",
    "clone_persona",
    "delete_persona",
]


def export_persona_catalog(mode: PersonaMode) -> PersonaCatalogView:
    """Return catalog metadata for persona builders."""

    catalog = _load_catalog(mode)
    return PersonaCatalogView(
        mode=catalog.mode,
        roles=[
            PersonaCatalogRoleView(
                key=key,
                label=item.label,
                description=item.description,
                default_context_stage=item.default_context_stage,
            )
            for key, item in sorted(catalog.roles.items())
        ],
        drivers=[
            PersonaCatalogDriverView(
                key=key,
                label=item.label,
                emotional_anchor=item.emotional_anchor,
                weight=item.weight,
            )
            for key, item in sorted(catalog.drivers.items())
        ],
        contexts=[
            PersonaCatalogContextView(
                key=key,
                label=item.label,
                priority=item.priority,
            )
            for key, item in sorted(catalog.contexts.items())
        ],
        voices=[
            PersonaCatalogVoiceView(
                key=key,
                role=item.role,
                driver=item.driver,
                contexts=list(item.contexts),
            )
            for key, item in sorted(catalog.voices.items())
        ],
    )


def list_personas(
    db: Session,
    mode: PersonaMode,
    *,
    owner_id: Optional[str] = None,
) -> List[PersonaView]:
    """Return personas derived from the catalog plus any user-defined personas."""

    extractor = PersonaExtractor()

    try:
        catalog = extractor.load_catalog(mode)
        resolved = extractor.resolve_personas(mode)
    except PersonaCatalogError:
        catalog = None
        resolved = []

    catalog_views: List[PersonaView]
    if not resolved or catalog is None:
        catalog_views = _default_persona_views()
    else:
        catalog_views = [
            _resolution_to_view(persona, catalog, priority="primary", source="catalog")
            for persona in resolved
        ]

    if not owner_id:
        return catalog_views

    library = PersonaLibraryStore(db)
    custom_records = library.list(owner_id, mode=mode.value)
    custom_views = [
        _record_to_view(record, catalog) if catalog else _record_to_view(record, None)
        for record in custom_records
    ]
    return [*custom_views, *catalog_views]


def list_persona_library(
    db: Session, owner_id: str, mode: PersonaMode
) -> PersonaLibraryResponse:
    """Return the stored persona library for a specific owner and mode."""

    catalog = _load_catalog(mode)
    library = PersonaLibraryStore(db)
    records = library.list(owner_id, mode=mode.value)
    entries = [_record_to_library_entry(record, catalog) for record in records]
    return PersonaLibraryResponse(personas=entries)


def create_persona(
    db: Session,
    payload: PersonaComposeRequest,
) -> PersonaLibraryEntryView:
    """Compose and persist a persona for an owner from provided selections."""

    catalog = _load_catalog(payload.mode)
    persona_resolution = _resolve_from_selection(
        catalog,
        payload.mode,
        voice=payload.voice,
        role=payload.role,
        driver=payload.driver,
        contexts=payload.contexts,
    )

    journey_stage = _build_journey_stage(
        catalog,
        persona_resolution.contexts,
        payload.journey_stage,
    )

    role = catalog.roles.get(persona_resolution.role)
    driver = catalog.drivers.get(persona_resolution.driver)

    name = payload.name or (role.label if role else persona_resolution.role)
    segment = payload.segment or payload.mode.value.upper()
    priority = payload.priority or "secondary"
    key_need = (
        payload.key_need
        or (driver.label if driver and driver.label else persona_resolution.emotional_anchor or "")
    )

    meta = {
        "source": "custom",
        "role": persona_resolution.role,
        "driver": persona_resolution.driver,
        "voice": payload.voice,
        "contextKeys": list(persona_resolution.contexts),
    }

    library = PersonaLibraryStore(db)
    record = library.new_record(
        owner_id=payload.owner_id,
        mode=payload.mode.value,
        name=name,
        segment=segment,
        priority=priority,
        key_need=key_need,
        journey_stage=journey_stage,
        role=persona_resolution.role,
        driver=persona_resolution.driver,
        voice=payload.voice,
        contexts=persona_resolution.contexts,
        meta=meta,
    )
    library.save(record)
    return _record_to_library_entry(record, catalog)


def update_persona(
    db: Session, persona_id: str, payload: PersonaUpdateRequest
) -> PersonaLibraryEntryView:
    """Update an existing persona belonging to an owner."""

    mode = payload.mode if hasattr(payload, "mode") else PersonaMode.B2C
    if isinstance(mode, str):
        mode = PersonaMode(mode)

    catalog = _load_catalog(mode)
    library = PersonaLibraryStore(db)
    record = library.get(payload.owner_id, persona_id)
    if record is None:
        raise PersonaCatalogError("Persona not found for owner")

    voice = payload.voice if payload.voice is not None else record.voice
    role = payload.role if payload.role is not None else record.role
    driver = payload.driver if payload.driver is not None else record.driver
    contexts = payload.contexts if payload.contexts is not None else record.contexts

    persona_resolution = _resolve_from_selection(
        catalog,
        mode,
        voice=voice,
        role=role,
        driver=driver,
        contexts=contexts,
    )

    journey_stage = _build_journey_stage(
        catalog,
        persona_resolution.contexts,
        payload.journey_stage,
        fallback=record.journey_stage,
    )

    role_meta = catalog.roles.get(persona_resolution.role)
    driver_meta = catalog.drivers.get(persona_resolution.driver)

    record.name = payload.name or record.name or (
        role_meta.label if role_meta else persona_resolution.role
    )
    record.segment = payload.segment or record.segment or PersonaMode(record.mode).value.upper()
    record.priority = payload.priority or record.priority
    record.key_need = (
        payload.key_need
        or record.key_need
        or (driver_meta.label if driver_meta and driver_meta.label else persona_resolution.emotional_anchor or "")
    )
    record.journey_stage = [dict(stage) for stage in journey_stage]
    record.role = persona_resolution.role
    record.driver = persona_resolution.driver
    record.voice = voice
    record.contexts = list(persona_resolution.contexts)
    record.meta.update(
        {
            "source": "custom",
            "role": record.role,
            "driver": record.driver,
            "voice": record.voice,
            "contextKeys": list(record.contexts),
            "mode": mode.value,
        }
    )
    record.mode = mode.value
    library.touch(record)
    library.save(record)
    return _record_to_library_entry(record, catalog)


def clone_persona(
    db: Session, persona_id: str, owner_id: str, *, name: Optional[str] = None
) -> PersonaLibraryEntryView:
    """Duplicate a stored persona for the same owner."""

    library = PersonaLibraryStore(db)
    record = library.get(owner_id, persona_id)
    if record is None:
        raise PersonaCatalogError("Persona not found for cloning")

    catalog = _load_catalog(PersonaMode(record.mode))
    clone = library.new_record(
        owner_id=owner_id,
        mode=record.mode,
        name=name or f"{record.name} Copy",
        segment=record.segment,
        priority=record.priority,
        key_need=record.key_need,
        journey_stage=record.journey_stage,
        role=record.role,
        driver=record.driver,
        voice=record.voice,
        contexts=record.contexts,
        meta={**record.meta, "source": "custom"},
    )
    library.save(clone)
    return _record_to_library_entry(clone, catalog)


def delete_persona(db: Session, persona_id: str, owner_id: str) -> bool:
    """Delete a stored persona belonging to an owner."""

    library = PersonaLibraryStore(db)
    return library.delete(owner_id, persona_id)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_catalog(mode: PersonaMode) -> PersonaCatalog:
    extractor = PersonaExtractor()
    return extractor.load_catalog(mode)


def _resolve_from_selection(
    catalog: PersonaCatalog,
    mode: PersonaMode,
    *,
    voice: Optional[str],
    role: Optional[str],
    driver: Optional[str],
    contexts: Optional[Sequence[str]],
) -> PersonaResolution:
    voice_preset = None
    if voice:
        voice_preset = catalog.voices.get(voice)
        if voice_preset is None:
            raise PersonaCatalogError(f"Unknown voice preset: {voice}")

    role_key = role or (voice_preset.role if voice_preset else None)
    driver_key = driver or (voice_preset.driver if voice_preset else None)
    context_keys: Optional[Sequence[str]] = contexts or (
        voice_preset.contexts if voice_preset else None
    )

    if not role_key or not driver_key or not context_keys:
        raise ValueError("Role, driver, and at least one context are required")

    if role_key not in catalog.roles:
        raise PersonaCatalogError(f"Unknown persona role: {role_key}")
    if driver_key not in catalog.drivers:
        raise PersonaCatalogError(f"Unknown persona driver: {driver_key}")
    for context_key in context_keys:
        if context_key not in catalog.contexts:
            raise PersonaCatalogError(f"Unknown persona context: {context_key}")

    selection = PersonaSelection(
        role=role_key,
        driver=driver_key,
        contexts=list(context_keys),
        voice=voice,
    )

    extractor = PersonaExtractor()
    resolved = extractor.resolve_personas(mode, selections=[selection])
    if not resolved:
        raise PersonaCatalogError("Unable to resolve persona with supplied specification")
    return resolved[-1]


def _default_persona_views() -> List[PersonaView]:
    personas = []
    for item in default_personas():
        personas.append(
            PersonaView(
                id=item["id"],
                name=item["name"],
                segment=item["segment"],
                priority=item["priority"],
                key_need=item["key_need"],
                journey_stage=[
                    PersonaStageView(
                        stage=stage["stage"],
                        question=stage["question"],
                        coverage=stage["coverage"],
                    )
                    for stage in item["journey_stage"]
                ],
                meta={"source": "default"},
            )
        )
    return personas


def _resolution_to_view(
    persona: PersonaResolution,
    catalog: PersonaCatalog,
    *,
    priority: str,
    source: str,
) -> PersonaView:
    role = catalog.roles.get(persona.role)
    driver = catalog.drivers.get(persona.driver)

    contexts = _prepare_contexts(persona.contexts, catalog)
    coverage = 1.0 / len(contexts) if contexts else 1.0

    journey = [
        PersonaStageView(stage=stage, question="", coverage=coverage) for stage in contexts
    ]

    identifier = "-".join(
        [
            persona.mode.value,
            persona.role,
            persona.driver,
            "x".join(persona.contexts),
        ]
    )

    return PersonaView(
        id=identifier,
        name=role.label if role else persona.role,
        segment=persona.mode.value.upper(),
        priority=priority,
        key_need=(driver.label if driver and driver.label else persona.emotional_anchor or ""),
        journey_stage=journey,
        meta={
            "source": source,
            "role": persona.role,
            "driver": persona.driver,
            "voice": persona.voice,
            "contextKeys": list(persona.contexts),
        },
    )


def _prepare_contexts(context_keys: Iterable[str], catalog: PersonaCatalog) -> List[str]:
    contexts: List[str] = []
    for key in context_keys:
        context = catalog.contexts.get(key)
        contexts.append(context.label if context else key)
    return contexts


def _build_journey_stage(
    catalog: PersonaCatalog,
    context_keys: Sequence[str],
    override: Optional[Sequence[PersonaStageView]] = None,
    fallback: Optional[Sequence[dict]] = None,
) -> List[dict]:
    if override:
        return [
            {
                "stage": stage.stage,
                "question": stage.question,
                "coverage": stage.coverage,
            }
            for stage in override
        ]

    if fallback:
        fallback_list = [dict(entry) for entry in fallback]
        if len(fallback_list) == len(context_keys):
            return fallback_list

    labels = _prepare_contexts(context_keys, catalog)
    coverage = 1.0 / len(labels) if labels else 1.0
    return [
        {"stage": label, "question": "", "coverage": coverage} for label in labels
    ]


def _record_to_view(record: PersonaRecord, catalog: Optional[PersonaCatalog]) -> PersonaView:
    contexts = record.contexts
    if catalog:
        labels = _prepare_contexts(contexts, catalog)
    else:
        labels = contexts

    journey_stage = [
        PersonaStageView(
            stage=entry.get("stage", labels[index] if index < len(labels) else ""),
            question=str(entry.get("question", "")),
            coverage=float(entry.get("coverage", 0.0)),
        )
        for index, entry in enumerate(record.journey_stage)
    ]

    return PersonaView(
        id=str(record.id),
        name=record.name,
        segment=record.segment,
        priority=record.priority,  # type: ignore[arg-type]
        key_need=record.key_need,
        journey_stage=journey_stage,
        meta={
            "source": "custom",
            "ownerId": record.owner_id,
            "role": record.role,
            "driver": record.driver,
            "voice": record.voice,
            "contextKeys": list(contexts),
            "createdAt": record.created_at.isoformat() if hasattr(record.created_at, 'isoformat') else str(record.created_at),
            "updatedAt": record.updated_at.isoformat() if hasattr(record.updated_at, 'isoformat') else str(record.updated_at),
            "mode": record.mode,
        },
    )


def _record_to_library_entry(
    record: PersonaRecord, catalog: PersonaCatalog
) -> PersonaLibraryEntryView:
    base_view = _record_to_view(record, catalog)
    return PersonaLibraryEntryView(
        id=base_view.id,
        name=base_view.name,
        segment=base_view.segment,
        priority=base_view.priority,
        key_need=base_view.key_need,
        journey_stage=base_view.journey_stage,
        meta=base_view.meta,
        owner_id=record.owner_id,
        mode=PersonaMode(record.mode),
        role=record.role,
        driver=record.driver,
        voice=record.voice,
        context_keys=list(record.contexts),
        created_at=record.created_at.isoformat() if hasattr(record.created_at, 'isoformat') else str(record.created_at),
        updated_at=record.updated_at.isoformat() if hasattr(record.updated_at, 'isoformat') else str(record.updated_at),
    )
