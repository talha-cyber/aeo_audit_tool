from sqlalchemy import JSON, Column, ForeignKey, String, Text
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
    config = Column(JSON)
    status = Column(String, nullable=False, default="pending")
    error_log = Column(Text, nullable=True)
    responses = relationship("Response", back_populates="audit_run")
    report = relationship("Report", back_populates="audit_run", uselist=False)
