#!/usr/bin/env python3
"""
Seeds the database with an initial test client for running audits.
"""
import asyncio
import uuid

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.audit import Client
from app.utils.logger import get_logger

logger = get_logger(__name__)


def seed_test_client(db: Session) -> Client:
    """
    Creates a standard test client if one doesn't already exist.

    Args:
        db: The database session.

    Returns:
        The existing or newly created test client.
    """
    client_name = "TestCRM"
    existing_client = db.query(Client).filter(Client.name == client_name).first()

    if existing_client:
        logger.info(
            f"Test client '{client_name}' already exists.", client_id=existing_client.id
        )
        return existing_client

    logger.info(f"Creating new test client: {client_name}")
    new_client = Client(
        id=str(uuid.uuid4()),
        name=client_name,
        industry="CRM",
        product_type="Software",
        competitors=["Salesforce", "HubSpot", "Microsoft Dynamics", "SAP"],
    )
    db.add(new_client)
    db.commit()
    db.refresh(new_client)
    logger.info("Test client created successfully.", client_id=new_client.id)
    return new_client


async def main():
    """Main function to seed the database."""
    logger.info("--- Starting Database Seeding ---")
    db = SessionLocal()
    try:
        seed_test_client(db)
        logger.info("--- Database Seeding Completed ---")
    except Exception as e:
        logger.error(
            "An error occurred during database seeding.", error=str(e), exc_info=True
        )
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
