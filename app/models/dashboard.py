"""Persistence models that back dashboard data contracts."""

from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class DashboardSettings(Base):
    """Tenant-level dashboard configuration (branding, billing, etc.)."""

    __tablename__ = "dashboard_settings"

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(String, nullable=False, unique=True)
    branding_primary_color = Column(String, nullable=False, default="#111214")
    branding_logo_url = Column(String, nullable=True)
    branding_tone = Column(String, nullable=False, default="Measured and confident")
    billing_plan = Column(String, nullable=False, default="Agency Pro")
    billing_renews_on = Column(DateTime(timezone=True), nullable=True)

    members = relationship("DashboardMember", cascade="all, delete-orphan", back_populates="settings")
    integrations = relationship(
        "DashboardIntegration", cascade="all, delete-orphan", back_populates="settings"
    )


class DashboardMember(Base):
    """Members listed in the dashboard settings panel."""

    __tablename__ = "dashboard_members"

    id = Column(Integer, primary_key=True, index=True)
    settings_id = Column(Integer, ForeignKey("dashboard_settings.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)
    email = Column(String, nullable=False)

    settings = relationship("DashboardSettings", back_populates="members")


class DashboardIntegration(Base):
    """Third-party integrations surfaced to the dashboard."""

    __tablename__ = "dashboard_integrations"

    id = Column(Integer, primary_key=True, index=True)
    settings_id = Column(Integer, ForeignKey("dashboard_settings.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    connected = Column(Boolean, nullable=False, default=False)

    settings = relationship("DashboardSettings", back_populates="integrations")


class DashboardWidget(Base):
    """Embeddable dashboard widgets."""

    __tablename__ = "dashboard_widgets"

    id = Column(Integer, primary_key=True, index=True)
    slug = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=False)
    preview = Column(String, nullable=True)
    status = Column(String, nullable=False, default="draft")
    category = Column(String, nullable=True)
    description = Column(String, nullable=True)

