#!/usr/bin/env python3
"""
Quick test script to verify the FastAPI app and Celery worker setup
"""

import os
import sys
from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from starlette.testclient import TestClient

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Fixture for the FastAPI test client
@pytest.fixture(scope="module")
def client() -> Generator:
    from app.main import app

    with TestClient(app) as c:
        yield c


# Fixture for the Celery app for testing
@pytest.fixture(scope="module")
def celery_app_fixture() -> tuple:
    from celery_worker import celery_app, test_task

    celery_app.conf.update(task_always_eager=True)
    return celery_app, test_task


# Fixture for an in-memory SQLite database for testing
@pytest.fixture(scope="session")
def db_engine() -> Generator:
    """Yield a SQLAlchemy engine for an in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:")
    # Base.metadata.create_all(bind=engine)  # Create tables
    yield engine
    # Base.metadata.drop_all(bind=engine)  # Drop tables after tests


@pytest.fixture(scope="function")
def db_session(db_engine: Engine) -> Generator:
    """Yield a database session for a single test function."""
    connection = db_engine.connect()
    transaction = connection.begin()
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = session_local()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
