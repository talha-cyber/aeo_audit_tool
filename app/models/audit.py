from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import expression

from app.db.base_class import Base


class Client(Base):
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    industry = Column(String)
    product_type = Column(String)
    competitors = Column(JSON)
    is_internal = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default=expression.false(),
    )
    admin_settings = Column(JSON, nullable=False, default=dict, server_default="{}")
    audits = relationship("AuditRun", back_populates="client")

    def __init__(self, **kwargs):
        if kwargs.get("is_internal") is None:
            kwargs["is_internal"] = False
        if "admin_settings" not in kwargs or kwargs.get("admin_settings") is None:
            kwargs["admin_settings"] = {}
        super().__init__(**kwargs)


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
