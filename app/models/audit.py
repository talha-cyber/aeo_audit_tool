import uuid

from sqlalchemy import JSON, Column, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Client(Base):
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    audits = relationship("AuditRun", back_populates="client")


class AuditRun(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(Integer, ForeignKey("client.id"))
    client = relationship("Client", back_populates="audits")
    config = Column(JSON)
    responses = relationship("Response", back_populates="audit_run")
    report = relationship("Report", back_populates="audit_run", uselist=False)
