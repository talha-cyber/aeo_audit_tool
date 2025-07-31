from sqlalchemy import Column, ForeignKey, String
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Report(Base):
    id = Column(String, primary_key=True)
    audit_run_id = Column(String, ForeignKey("auditrun.id"))
    audit_run = relationship("AuditRun", back_populates="report")
    file_path = Column(String)
