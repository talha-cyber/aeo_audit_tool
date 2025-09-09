from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Report(Base):
    id = Column(String, primary_key=True)
    audit_run_id = Column(String, ForeignKey("auditrun.id"))
    audit_run = relationship("AuditRun", back_populates="report")
    file_path = Column(String)
    report_type = Column(String)
    generated_at = Column(DateTime(timezone=True))
    template_version = Column(String, nullable=True)  # v2 template version
    theme_key = Column(String, nullable=True)  # v2 theme identifier
