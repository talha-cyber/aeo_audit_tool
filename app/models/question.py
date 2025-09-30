from sqlalchemy import JSON, Column, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Question(Base):
    __tablename__ = "questions"  # type: ignore
    __table_args__ = (
        Index("ix_questions_audit_run_persona", "audit_run_id", "persona"),
        Index("ix_questions_driver_context_stage", "driver", "context_stage"),
    )

    id = Column(String, primary_key=True)
    audit_run_id = Column(
        String, ForeignKey("auditrun.id"), nullable=False
    )  # Link to specific audit run
    question_text = Column(Text, nullable=False)
    category = Column(String, nullable=False)  # comparison, recommendation, etc.
    question_type = Column(String)  # industry_general, brand_specific, etc.
    priority_score = Column(Float, default=0.0)  # Calculated priority
    target_brand = Column(String, nullable=True)  # Optional target brand
    provider = Column(String)  # Which provider generated it
    provider_version = Column(String, nullable=True)
    persona = Column(String, nullable=True)
    role = Column(String, nullable=True)
    driver = Column(String, nullable=True)
    emotional_anchor = Column(String, nullable=True)
    context_stage = Column(String, nullable=True)
    seed_type = Column(String, nullable=True)
    cost = Column(Float)  # Keep existing cost field
    tokens = Column(JSON)  # Keep existing tokens field
    question_metadata = Column(JSON)  # Additional question context

    # Relationships
    audit_run = relationship("AuditRun", back_populates="questions")
    responses = relationship("Response", back_populates="question")
