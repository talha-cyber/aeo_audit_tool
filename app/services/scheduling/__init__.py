"""
Comprehensive Audit Scheduling System

Production-ready job scheduling system with persistent storage, distributed
execution, retry policies, and comprehensive monitoring.

Key Features:
- Persistent job scheduling with database storage
- Multiple trigger types (cron, interval, one-time, dependencies)
- Distributed execution with leader election
- Comprehensive retry policies and error handling
- Job dependency management
- Performance monitoring and health checks
- Integration with existing Celery infrastructure
"""

from .engine import ExecutionResult, JobDefinition, SchedulerEngine, SchedulerStatus
from .execution_manager import ExecutionManager
from .repository import SchedulingRepository
from .triggers.factory import TriggerFactory

# Main scheduler instance
scheduler_engine: SchedulerEngine = None


def get_scheduler_engine() -> SchedulerEngine:
    """Get or create the global scheduler engine instance"""
    global scheduler_engine
    if scheduler_engine is None:
        scheduler_engine = SchedulerEngine()
    return scheduler_engine


# Convenience functions
async def schedule_job(job_definition: JobDefinition) -> str:
    """Schedule a new job"""
    engine = get_scheduler_engine()
    return await engine.schedule_job(job_definition)


async def cancel_job(job_id: str) -> bool:
    """Cancel a scheduled job"""
    engine = get_scheduler_engine()
    return await engine.cancel_job(job_id)


async def get_job_status(job_id: str) -> dict:
    """Get status of a scheduled job"""
    engine = get_scheduler_engine()
    return await engine.get_job_status(job_id)


__all__ = [
    "SchedulerEngine",
    "JobDefinition",
    "ExecutionResult",
    "SchedulerStatus",
    "TriggerFactory",
    "SchedulingRepository",
    "ExecutionManager",
    "get_scheduler_engine",
    "schedule_job",
    "cancel_job",
    "get_job_status",
]
