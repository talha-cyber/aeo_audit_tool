from sqlalchemy import JSON, Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Response(Base):
    __tablename__ = "responses"  # type: ignore
    id = Column(String, primary_key=True)
    audit_run_id = Column(String, ForeignKey("auditrun.id"), nullable=False)
    question_id = Column(
        String, ForeignKey("questions.id"), nullable=True
    )  # Optional question link
    platform = Column(String, nullable=False)  # openai, anthropic, etc.
    response_text = Column(
        Text, nullable=False
    )  # Normalized response text (renamed from response)
    raw_response = Column(JSON, nullable=False)  # Complete platform response
    brand_mentions = Column(JSON, nullable=True)  # Brand detection results
    response_metadata = Column(JSON, nullable=True)  # Timing, tokens, cost, etc.
    processing_time_ms = Column(Integer, nullable=True)  # Query execution time

    # Relationships
    audit_run = relationship("AuditRun", back_populates="responses")
    question = relationship("Question", back_populates="responses")
