#!/usr/bin/env python3
"""Seed the Internal Preview tenant for admin-as-tenant workflows."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Iterable

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.audit import AuditRun, Client
from app.models.dashboard import DashboardMember, DashboardSettings
from app.models.persona import Persona
from app.services.dashboard.persona_store import PersonaLibraryStore
from app.utils.logger import get_logger

logger = get_logger(__name__)

INTERNAL_CLIENT_NAME = "Internal Preview – AEO"
INTERNAL_OWNER_ID = "admin@aeo.internal"
INTERNAL_ADMIN_EMAIL = "admin@aeo.internal"
INTERNAL_PERSONAS: Iterable[dict] = [
    {
        "name": "Operations Excellence Lead",
        "segment": "Enterprise",
        "priority": "primary",
        "key_need": "Unified visibility across search assistants",
        "journey_stage": [
            {
                "stage": "Awareness",
                "question": "Where are gaps today?",
                "coverage": 0.33,
            },
            {
                "stage": "Consideration",
                "question": "Which platform is lagging?",
                "coverage": 0.33,
            },
            {
                "stage": "Decision",
                "question": "What should we fix first?",
                "coverage": 0.34,
            },
        ],
        "role": "operations_lead",
        "driver": "efficiency",
        "voice": "operational_strategist",
        "contexts": ["awareness", "consideration", "decision"],
    },
    {
        "name": "Growth Marketing Strategist",
        "segment": "Agency",
        "priority": "secondary",
        "key_need": "Campaign-ready insights for clients",
        "journey_stage": [
            {
                "stage": "Awareness",
                "question": "What narratives are trending?",
                "coverage": 0.4,
            },
            {
                "stage": "Consideration",
                "question": "Which competitors dominate?",
                "coverage": 0.3,
            },
            {
                "stage": "Retention",
                "question": "How do we prove lift?",
                "coverage": 0.3,
            },
        ],
        "role": "marketing_lead",
        "driver": "growth",
        "voice": "growth_strategist",
        "contexts": ["awareness", "consideration", "retention"],
    },
]


def _ensure_internal_client(db: Session) -> Client:
    client = db.query(Client).filter(Client.name == INTERNAL_CLIENT_NAME).one_or_none()
    if client:
        if not client.is_internal:
            client.is_internal = True
            db.add(client)
            db.commit()
        admin_settings = client.admin_settings or {}
        admin_settings.setdefault("qe_version", "v2")
        admin_settings.setdefault("feature_flags", {"overlay_experiment": True})
        admin_settings.setdefault(
            "platform_overrides", {"openai": True, "claude": True}
        )
        client.admin_settings = admin_settings
        db.add(client)
        db.commit()
        return client

    client = Client(
        id=str(uuid.uuid4()),
        name=INTERNAL_CLIENT_NAME,
        industry="AI Agents",
        product_type="Analytics Platform",
        competitors=[
            "AgentIQ",
            "CompassLabs",
            "PromptOps",
            "SignalFoundry",
        ],
        is_internal=True,
        admin_settings={
            "qe_version": "v2",
            "feature_flags": {"overlay_experiment": True},
            "platform_overrides": {"openai": True, "claude": True},
        },
    )
    db.add(client)
    db.commit()
    db.refresh(client)
    logger.info("Created internal preview client", client_id=client.id)
    return client


def _ensure_dashboard_settings(db: Session, client: Client) -> DashboardSettings:
    settings = (
        db.query(DashboardSettings)
        .filter(DashboardSettings.organization_id == client.id)
        .one_or_none()
    )
    if settings:
        return settings

    settings = DashboardSettings(
        organization_id=client.id,
        branding_primary_color="#13284B",
        branding_tone="Operational confidence",
        billing_plan="Internal Preview",
    )
    db.add(settings)
    db.commit()
    db.refresh(settings)
    logger.info(
        "Provisioned dashboard settings for internal preview", settings_id=settings.id
    )
    return settings


def _ensure_admin_member(db: Session, settings: DashboardSettings) -> None:
    existing = (
        db.query(DashboardMember)
        .filter(
            DashboardMember.settings_id == settings.id,
            DashboardMember.email == INTERNAL_ADMIN_EMAIL,
        )
        .one_or_none()
    )
    if existing:
        return

    member = DashboardMember(
        settings_id=settings.id,
        name="AEO Internal Admin",
        role="Administrator",
        email=INTERNAL_ADMIN_EMAIL,
    )
    db.add(member)
    db.commit()
    logger.info("Added admin member to internal preview settings", member_id=member.id)


def _ensure_personas(db: Session, client: Client) -> None:
    store = PersonaLibraryStore(db)
    existing_names = {
        persona.name
        for persona in db.query(Persona)
        .filter(Persona.client_id == client.id, Persona.owner_id == INTERNAL_OWNER_ID)
        .all()
    }

    for persona in INTERNAL_PERSONAS:
        if persona["name"] in existing_names:
            continue
        record = store.new_record(
            owner_id=INTERNAL_OWNER_ID,
            client_id=client.id,
            mode="b2b",
            name=persona["name"],
            segment=persona["segment"],
            priority=persona["priority"],
            key_need=persona["key_need"],
            journey_stage=persona["journey_stage"],
            role=persona["role"],
            driver=persona["driver"],
            voice=persona.get("voice"),
            contexts=persona["contexts"],
            meta={
                "source": "seed",
                "clientId": client.id,
                "contextKeys": persona["contexts"],
            },
        )
        store.save(record)
        logger.info(
            "Seeded internal persona",
            persona_id=str(record.id),
            client_id=client.id,
            owner_id=INTERNAL_OWNER_ID,
        )


def seed_internal_preview(db: Session) -> Client:
    client = _ensure_internal_client(db)
    admin_settings = client.admin_settings or {}
    admin_settings.setdefault("qe_version", "v2")
    admin_settings.setdefault("feature_flags", {"overlay_experiment": True})
    admin_settings.setdefault("platform_overrides", {"openai": True, "claude": True})
    client.admin_settings = admin_settings
    db.add(client)
    db.commit()
    settings = _ensure_dashboard_settings(db, client)
    _ensure_admin_member(db, settings)
    _ensure_personas(db, client)
    _ensure_sample_runs(db, client)
    return client


def _ensure_sample_runs(db: Session, client: Client) -> None:
    existing_ids = {
        row[0]
        for row in db.query(AuditRun.id).filter(AuditRun.client_id == client.id).all()
    }
    now = datetime.utcnow()
    templates = [
        (
            "completed",
            "completed",
            now - timedelta(days=2),
            now - timedelta(days=2, hours=-2),
        ),
        ("running", "running", now - timedelta(hours=3), None),
        ("pending", "pending", None, None),
    ]
    created = 0
    for suffix, status, started, completed in templates:
        run_id = f"{client.id}-seed-{suffix}"
        if run_id in existing_ids:
            continue
        run = AuditRun(
            id=run_id,
            client_id=client.id,
            status=status,
            started_at=started,
            completed_at=completed,
            config={
                "client": {"id": client.id, "name": client.name},
                "platforms": ["openai", "claude"],
            },
            processed_questions=28 if status != "pending" else 0,
            total_questions=40,
        )
        db.add(run)
        created += 1
    if created:
        db.commit()
        logger.info("Seeded internal preview runs", count=created)


async def main() -> None:
    db = SessionLocal()
    try:
        client = seed_internal_preview(db)
        logger.info("Internal preview seeding complete", client_id=client.id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to seed internal preview", error=str(exc), exc_info=True)
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
