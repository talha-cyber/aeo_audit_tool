from sqlalchemy import JSON, Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class Client(Base):
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    audits = relationship("AuditRun", back_populates="client")


class AuditRun(Base):
    id = Column(String, primary_key=True)
    client_id = Column(Integer, ForeignKey("client.id"))
    client = relationship("Client", back_populates="audits")
    config = Column(JSON)
    status = Column(String, nullable=False, default="pending")
    error_log = Column(Text, nullable=True)
    responses = relationship("Response", back_populates="audit_run")
    report = relationship("Report", back_populates="audit_run", uselist=False)
