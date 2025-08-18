from sqlalchemy import JSON, Column, Float, ForeignKey, String, Text
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Question(Base):
    __tablename__ = "questions"  # type: ignore
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
    cost = Column(Float)  # Keep existing cost field
    tokens = Column(JSON)  # Keep existing tokens field
    question_metadata = Column(JSON)  # Additional question context

    # Relationships
    audit_run = relationship("AuditRun", back_populates="questions")
    responses = relationship("Response", back_populates="question")
