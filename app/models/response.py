from sqlalchemy import JSON, Column, ForeignKey, String, Text
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Response(Base):
    __tablename__ = "responses"  # type: ignore
    id = Column(String, primary_key=True)
    audit_run_id = Column(String, ForeignKey("auditrun.id"))
    audit_run = relationship("AuditRun", back_populates="responses")
    question = Column(String)
    response = Column(Text)
    raw_response = Column(JSON)
    platform = Column(String)
