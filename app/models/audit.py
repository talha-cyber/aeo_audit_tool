from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Client(Base):
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    industry = Column(String)
    product_type = Column(String)
    competitors = Column(JSON)
    audits = relationship("AuditRun", back_populates="client")


class AuditRun(Base):
    id = Column(String, primary_key=True)
    client_id = Column(String, ForeignKey("client.id"))
    client = relationship("Client", back_populates="audits")
    config = Column(JSON, nullable=False)  # Audit configuration snapshot
    status = Column(
        String, nullable=False, default="pending"
    )  # pending, running, completed, failed, cancelled
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    total_questions = Column(Integer, default=0)  # Total planned questions
    processed_questions = Column(Integer, default=0)  # Questions actually processed
    error_log = Column(Text, nullable=True)  # Detailed error information
    progress_data = Column(JSON, nullable=True)  # Real-time progress details
    platform_stats = Column(JSON, nullable=True)  # Per-platform statistics

    # Relationships
    responses = relationship("Response", back_populates="audit_run")
    questions = relationship("Question", back_populates="audit_run")
    report = relationship("Report", back_populates="audit_run", uselist=False)
