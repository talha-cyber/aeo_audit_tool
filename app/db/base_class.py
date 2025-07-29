from typing import Any

from sqlalchemy import Column, DateTime
from sqlalchemy.orm import DeclarativeBase, declared_attr
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    @declared_attr.directive  # type: ignore[misc]
    def __tablename__(cls) -> Any:  # type: ignore[misc]  # noqa: N805
        return cls.__name__.lower()
