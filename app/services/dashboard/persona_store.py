"""Persistent storage helpers for dashboard personas."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.persona import Persona as PersonaModel


@dataclass(slots=True)
class PersonaRecord:
    """Serializable record representing a stored persona."""

    id: UUID
    owner_id: str
    mode: str
    name: str
    segment: str
    priority: str
    key_need: str
    journey_stage: List[Dict[str, object]]
    role: str
    driver: str
    voice: Optional[str]
    contexts: List[str]
    meta: Dict[str, object]
    created_at: datetime
    updated_at: datetime


class PersonaLibraryStore:
    """Database-backed store for user-specific dashboard personas."""

    def __init__(self, db_session: Session) -> None:
        self._db = db_session

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def list(
        self, owner_id: str, *, mode: Optional[str] = None
    ) -> List[PersonaRecord]:
        """Return persona records for an owner, filtered by mode when provided."""
        query = self._db.query(PersonaModel).filter(PersonaModel.owner_id == owner_id)
        if mode:
            query = query.filter(PersonaModel.mode == mode)
        return [self._from_model(model) for model in query.all()]

    def get(self, owner_id: str, persona_id: UUID) -> Optional[PersonaRecord]:
        """Fetch a persona record for the given owner by identifier."""
        model = self._db.get(PersonaModel, persona_id)
        if model and model.owner_id == owner_id:
            return self._from_model(model)
        return None

    def save(self, record: PersonaRecord) -> PersonaRecord:
        """Persist (create or replace) a persona record."""
        model = self._db.get(PersonaModel, record.id)
        if model:
            # Update existing model
            for key, value in asdict(record).items():
                if hasattr(model, key):
                    setattr(model, key, value)
            model.updated_at = self._now()
        else:
            # Create new model
            model = PersonaModel(**asdict(record))

        self._db.add(model)
        self._db.commit()
        self._db.refresh(model)
        return self._from_model(model)

    def delete(self, owner_id: str, persona_id: UUID) -> bool:
        """Remove a persona from the store."""
        model = self._db.get(PersonaModel, persona_id)
        if model and model.owner_id == owner_id:
            self._db.delete(model)
            self._db.commit()
            return True
        return False

    def new_record(
        self,
        *,
        owner_id: str,
        mode: str,
        name: str,
        segment: str,
        priority: str,
        key_need: str,
        journey_stage: Iterable[Dict[str, object]],
        role: str,
        driver: str,
        voice: Optional[str],
        contexts: Iterable[str],
        meta: Optional[Dict[str, object]] = None,
    ) -> PersonaRecord:
        """Create a persona record with generated identifiers and timestamps."""
        now = self._now()
        return PersonaRecord(
            id=uuid.uuid4(),
            owner_id=owner_id,
            mode=mode,
            name=name,
            segment=segment,
            priority=priority,
            key_need=key_need,
            journey_stage=list(journey_stage),
            role=role,
            driver=driver,
            voice=voice,
            contexts=list(contexts),
            meta=meta or {},
            created_at=now,
            updated_at=now,
        )

    def touch(self, record: PersonaRecord) -> PersonaRecord:
        """Return a copy of the record with updated timestamp."""
        record.updated_at = self._now()
        return record

    def _from_model(self, model: PersonaModel) -> PersonaRecord:
        """Convert a SQLAlchemy model to a PersonaRecord."""
        return PersonaRecord(
            id=model.id,
            owner_id=model.owner_id,
            mode=model.mode,
            name=model.name,
            segment=model.segment,
            priority=model.priority,
            key_need=model.key_need,
            journey_stage=model.journey_stage,
            role=model.role,
            driver=model.driver,
            voice=model.voice,
            contexts=model.contexts,
            meta=model.meta,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


__all__ = ["PersonaLibraryStore", "PersonaRecord"]
