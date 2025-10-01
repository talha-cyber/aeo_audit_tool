"""Persona model for storing dashboard-specific compositions."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Column,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Persona(Base):
    __tablename__ = "personas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(String(255), nullable=False, index=True)
    mode = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    segment = Column(String(255))
    priority = Column(String(50))
    key_need = Column(Text)
    journey_stage = Column(JSON)
    role = Column(String(255))
    driver = Column(String(255))
    voice = Column(String(255))
    contexts = Column(JSON)
    meta = Column(JSON)
    created_at = Column(
        TIMESTAMP(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
