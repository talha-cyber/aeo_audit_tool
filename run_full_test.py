#!/usr/bin/env python3
"""
Runs a complete, end-to-end audit and generates a PDF report.

This script orchestrates the entire process:
1.  Seeds the database with a test client.
2.  Triggers a new audit run for that client.
3.  Monitors the audit status until completion.
4.  Triggers the report generation for the completed audit.
5.  Prints the path to the final PDF report.
"""
import asyncio
import os
import time

import requests
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE importing application modules.
# This is crucial for ensuring the correct database host is used for local scripts.
load_dotenv()

from app.utils.logger import get_logger

# --- Configuration ---
BASE_URL = "http://localhost:8000/api/v1"
CLIENT_NAME = "TestCRM"

logger = get_logger(__name__)


def seed_database() -> str:
    """
    Runs the seeding script and returns the client ID.
    """
    logger.info("--- 1. Seeding Database ---")
    from app.db.session import SessionLocal
    from scripts.create_test_client import seed_test_client

    db = SessionLocal()
    try:
        client = seed_test_client(db)
        logger.info(f"Database seeded. Client ID: {client.id}")
        return client.id
    finally:
        db.close()


def trigger_audit(client_id: str) -> str:
    """
    Triggers an audit run and returns the run ID.
    """
    logger.info("--- 2. Triggering Audit Run ---")
    url = f"{BASE_URL}/audits/configs/{client_id}/run"
    response = requests.post(url)
    response.raise_for_status()
    data = response.json()
    run_id = data["id"]
    logger.info(f"Audit triggered successfully. Run ID: {run_id}")
    return run_id


def monitor_audit_status(run_id: str):
    """
    Monitors the audit status until it completes or fails.
    """
    logger.info("--- 3. Monitoring Audit Status ---")
    url = f"{BASE_URL}/audits/runs/{run_id}/status"
    while True:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            status = data["status"]
            logger.info(f"Current audit status: {status}")

            if status in ["completed", "failed"]:
                logger.info(f"Audit finished with status: {status}")
                if status == "failed":
                    logger.error("Audit failed.", error_log=data.get("error_log"))
                    raise Exception("Audit run failed.")
                break
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking status: {e}")

        time.sleep(10)


def trigger_report_generation(run_id: str):
    """
    Triggers report generation for the completed audit.
    """
    logger.info("--- 4. Triggering Report Generation ---")
    url = f"{BASE_URL}/audits/runs/{run_id}/generate-report"
    response = requests.post(url, params={"report_type": "comprehensive"})
    response.raise_for_status()
    data = response.json()
    logger.info("Report generation queued.", response=data)


def get_report_path(run_id: str) -> str:
    """
    Polls for the report path until it's available.
    """
    logger.info("--- 5. Retrieving Report Path ---")
    url = f"{BASE_URL}/audits/runs/{run_id}/report"
    for _ in range(12):  # Poll for 2 minutes max
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                report_path = data.get("file_path")
                if report_path:
                    logger.info(f"Report path retrieved: {report_path}")
                    return report_path
            elif response.status_code in [404, 500]:
                logger.info(
                    f"API server error (status {response.status_code}), trying direct database lookup..."
                )
                # Fallback: check database directly
                from app.db.session import SessionLocal
                from app.models.report import Report

                db = SessionLocal()
                try:
                    report = (
                        db.query(Report).filter(Report.audit_run_id == run_id).first()
                    )
                    if report and report.file_path:
                        logger.info(f"Report found via database: {report.file_path}")
                        return report.file_path
                finally:
                    db.close()
            logger.info("Report not ready yet, waiting...")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching report path: {e}")
        time.sleep(10)
    raise TimeoutError("Timed out waiting for the report to be generated.")


async def main():
    """Main orchestration function."""

    # Verify API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set.")
        return

    logger.info("--- Starting End-to-End Audit Test ---")
    try:
        client_id = seed_database()
        run_id = trigger_audit(client_id)
        monitor_audit_status(run_id)
        trigger_report_generation(run_id)
        report_path = get_report_path(run_id)

        logger.info("--- ✅ End-to-End Audit Test Successful! ---")
        logger.info(f"📄 Report generated at: {report_path}")

    except Exception as e:
        logger.error(
            "--- ❌ End-to-End Audit Test Failed! ---", error=str(e), exc_info=True
        )


if __name__ == "__main__":
    asyncio.run(main())
