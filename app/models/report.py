import uuid

from sqlalchemy import JSON, Column, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Question(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    text = Column(Text, nullable=False)
    responses = relationship("Response", back_populates="question")


class Response(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audit_run_id = Column(UUID(as_uuid=True), ForeignKey("auditrun.id"))
    audit_run = relationship("AuditRun", back_populates="responses")
    question_id = Column(UUID(as_uuid=True), ForeignKey("question.id"))
    question = relationship("Question", back_populates="responses")
    raw_response = Column(JSON)


class Report(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audit_run_id = Column(UUID(as_uuid=True), ForeignKey("auditrun.id"))
    audit_run = relationship("AuditRun", back_populates="report")
    file_path = Column(String)
