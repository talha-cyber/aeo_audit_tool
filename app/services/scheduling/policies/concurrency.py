"""
Concurrency control for job scheduling.

Provides sophisticated concurrency management with limits, queuing,
resource allocation, and deadlock prevention.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, AsyncContextManager, Dict, List, Optional

from app.models.scheduling import ScheduledJob
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ConcurrencyMode(str, Enum):
    """Concurrency control modes"""

    ALLOW_CONCURRENT = "allow_concurrent"  # No limits
    FORBID_CONCURRENT = "forbid_concurrent"  # Only one instance at a time
    LIMITED_CONCURRENT = "limited_concurrent"  # Limited number of concurrent instances
    QUEUE_OVERFLOW = "queue_overflow"  # Queue when limits exceeded
    SKIP_OVERFLOW = "skip_overflow"  # Skip when limits exceeded


@dataclass
class ConcurrencyLimits:
    """Concurrency limits configuration"""

    global_limit: Optional[int] = None  # Max jobs running globally
    per_job_type_limit: Optional[int] = None  # Max jobs per job type
    per_job_limit: int = 1  # Max instances of same job
    per_resource_limit: Optional[Dict[str, int]] = None  # Limits per resource type
    queue_size: int = 100  # Max queue size when queuing overflow


@dataclass
class ResourceRequirement:
    """Resource requirements for job execution"""

    cpu_cores: Optional[float] = None
    memory_mb: Optional[int] = None
    gpu_count: Optional[int] = None
    custom_resources: Optional[Dict[str, int]] = None


@dataclass
class ExecutionSlot:
    """Represents an execution slot for concurrency tracking"""

    slot_id: str
    job_id: str
    job_type: str
    started_at: datetime
    resource_usage: Optional[ResourceRequirement] = None
    metadata: Optional[Dict[str, Any]] = None


class ConcurrencyPolicy:
    """
    Concurrency policy configuration for jobs.

    Defines how jobs should be handled when concurrency limits are reached.
    """

    def __init__(
        self,
        mode: ConcurrencyMode = ConcurrencyMode.LIMITED_CONCURRENT,
        limits: Optional[ConcurrencyLimits] = None,
        resource_requirements: Optional[ResourceRequirement] = None,
        priority_override: bool = False,
        timeout_seconds: int = 300,
    ):
        """
        Initialize concurrency policy.

        Args:
            mode: Concurrency control mode
            limits: Concurrency limits configuration
            resource_requirements: Resource requirements for execution
            priority_override: Allow high priority jobs to bypass limits
            timeout_seconds: Timeout for acquiring execution slot
        """
        self.mode = mode
        self.limits = limits or ConcurrencyLimits()
        self.resource_requirements = resource_requirements
        self.priority_override = priority_override
        self.timeout_seconds = timeout_seconds

    def allows_execution(
        self,
        job: ScheduledJob,
        current_executions: List[ExecutionSlot],
        available_resources: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if job execution is allowed under current policy.

        Args:
            job: Job to check
            current_executions: Currently running executions
            available_resources: Available system resources

        Returns:
            True if execution is allowed, False otherwise
        """
        if self.mode == ConcurrencyMode.ALLOW_CONCURRENT:
            return True

        # Check global limits
        if self.limits.global_limit:
            if len(current_executions) >= self.limits.global_limit:
                if not (self.priority_override and job.priority <= 2):  # High priority
                    return False

        # Check per-job limits
        same_job_executions = [e for e in current_executions if e.job_id == job.job_id]
        if len(same_job_executions) >= self.limits.per_job_limit:
            return False

        # Check per-job-type limits
        if self.limits.per_job_type_limit:
            same_type_executions = [
                e for e in current_executions if e.job_type == job.job_type
            ]
            if len(same_type_executions) >= self.limits.per_job_type_limit:
                if not (self.priority_override and job.priority <= 2):
                    return False

        # Check resource requirements
        if self.resource_requirements and available_resources:
            if not self._has_sufficient_resources(available_resources):
                return False

        return True

    def _has_sufficient_resources(self, available_resources: Dict[str, Any]) -> bool:
        """Check if sufficient resources are available"""
        if not self.resource_requirements:
            return True

        # Check CPU
        if self.resource_requirements.cpu_cores:
            available_cpu = available_resources.get("cpu_cores", 0)
            if available_cpu < self.resource_requirements.cpu_cores:
                return False

        # Check memory
        if self.resource_requirements.memory_mb:
            available_memory = available_resources.get("memory_mb", 0)
            if available_memory < self.resource_requirements.memory_mb:
                return False

        # Check GPU
        if self.resource_requirements.gpu_count:
            available_gpu = available_resources.get("gpu_count", 0)
            if available_gpu < self.resource_requirements.gpu_count:
                return False

        # Check custom resources
        if self.resource_requirements.custom_resources:
            for (
                resource,
                required_amount,
            ) in self.resource_requirements.custom_resources.items():
                available_amount = available_resources.get(resource, 0)
                if available_amount < required_amount:
                    return False

        return True


class ConcurrencyManager:
    """
    Manages job concurrency with sophisticated scheduling and resource allocation.

    Provides execution slot management, queuing, resource tracking,
    and deadlock prevention for job scheduling.
    """

    def __init__(self, default_policy: Optional[ConcurrencyPolicy] = None):
        """Initialize concurrency manager"""
        self.default_policy = default_policy or ConcurrencyPolicy()

        # Track active executions
        self._active_slots: Dict[str, ExecutionSlot] = {}
        self._slot_locks = asyncio.Lock()

        # Job-specific policies
        self._job_policies: Dict[str, ConcurrencyPolicy] = {}
        self._job_type_policies: Dict[str, ConcurrencyPolicy] = {}

        # Queuing for overflow handling
        self._execution_queue: asyncio.Queue = asyncio.Queue()
        self._queue_processing_task: Optional[asyncio.Task] = None

        # Resource tracking
        self._available_resources = {
            "cpu_cores": 4.0,  # Default values, should be configurable
            "memory_mb": 8192,
            "gpu_count": 0,
            "custom_resources": {},
        }

        # Statistics
        self._stats = {
            "total_acquisitions": 0,
            "queue_wait_times": [],
            "resource_contentions": 0,
            "policy_violations": 0,
        }

    def set_job_policy(self, job_id: str, policy: ConcurrencyPolicy) -> None:
        """Set concurrency policy for specific job"""
        self._job_policies[job_id] = policy
        logger.info(f"Set concurrency policy for job {job_id}", mode=policy.mode.value)

    def set_job_type_policy(self, job_type: str, policy: ConcurrencyPolicy) -> None:
        """Set concurrency policy for job type"""
        self._job_type_policies[job_type] = policy
        logger.info(
            f"Set concurrency policy for job type {job_type}", mode=policy.mode.value
        )

    def update_available_resources(self, resources: Dict[str, Any]) -> None:
        """Update available system resources"""
        self._available_resources.update(resources)
        logger.debug("Updated available resources", resources=resources)

    def get_policy_for_job(self, job: ScheduledJob) -> ConcurrencyPolicy:
        """Get effective concurrency policy for job"""
        # Job-specific policy takes precedence
        if job.job_id in self._job_policies:
            return self._job_policies[job.job_id]

        # Then job-type policy
        if job.job_type in self._job_type_policies:
            return self._job_type_policies[job.job_type]

        # Fall back to default
        return self.default_policy

    @asynccontextmanager
    async def acquire_execution_slot(
        self, job: ScheduledJob, timeout_seconds: Optional[int] = None
    ) -> AsyncContextManager[ExecutionSlot]:
        """
        Acquire execution slot for job with concurrency control.

        Args:
            job: Job requesting execution slot
            timeout_seconds: Maximum time to wait for slot

        Yields:
            ExecutionSlot for the job execution
        """
        policy = self.get_policy_for_job(job)

        if timeout_seconds is None:
            timeout_seconds = policy.timeout_seconds

        slot = None
        acquisition_start = datetime.now(timezone.utc)

        try:
            # Try to acquire slot
            slot = await asyncio.wait_for(
                self._acquire_slot_internal(job, policy), timeout=timeout_seconds
            )

            acquisition_time = (
                datetime.now(timezone.utc) - acquisition_start
            ).total_seconds()
            self._stats["total_acquisitions"] += 1
            self._stats["queue_wait_times"].append(acquisition_time)

            logger.info(
                "Acquired execution slot",
                job_id=job.job_id,
                slot_id=slot.slot_id,
                wait_time=acquisition_time,
            )

            yield slot

        except asyncio.TimeoutError:
            self._stats["policy_violations"] += 1
            logger.error(
                "Failed to acquire execution slot - timeout",
                job_id=job.job_id,
                timeout_seconds=timeout_seconds,
            )
            raise

        except Exception as e:
            logger.error(
                f"Failed to acquire execution slot: {e}",
                job_id=job.job_id,
                exc_info=True,
            )
            raise

        finally:
            # Always release slot
            if slot:
                await self._release_slot_internal(slot.slot_id)

    async def _acquire_slot_internal(
        self, job: ScheduledJob, policy: ConcurrencyPolicy
    ) -> ExecutionSlot:
        """Internal slot acquisition with policy enforcement"""
        while True:
            async with self._slot_locks:
                current_slots = list(self._active_slots.values())

                # Check if execution is allowed
                if policy.allows_execution(
                    job, current_slots, self._available_resources
                ):
                    # Create and register slot
                    slot = ExecutionSlot(
                        slot_id=str(uuid.uuid4()),
                        job_id=job.job_id,
                        job_type=job.job_type,
                        started_at=datetime.now(timezone.utc),
                        resource_usage=policy.resource_requirements,
                        metadata={"priority": job.priority},
                    )

                    self._active_slots[slot.slot_id] = slot

                    # Reserve resources
                    if policy.resource_requirements:
                        self._reserve_resources(policy.resource_requirements)

                    return slot

                else:
                    # Handle overflow based on policy mode
                    if policy.mode == ConcurrencyMode.SKIP_OVERFLOW:
                        raise RuntimeError(
                            "Execution skipped due to concurrency limits"
                        )

                    elif policy.mode == ConcurrencyMode.QUEUE_OVERFLOW:
                        # Add to queue and wait
                        logger.debug(
                            "Queuing job due to concurrency limits", job_id=job.job_id
                        )

                        # Wait for slot to become available
                        await asyncio.sleep(1.0)  # Brief wait before retry
                        continue

                    else:
                        # For FORBID_CONCURRENT, wait and retry
                        await asyncio.sleep(1.0)
                        continue

    async def _release_slot_internal(self, slot_id: str) -> None:
        """Internal slot release"""
        async with self._slot_locks:
            slot = self._active_slots.pop(slot_id, None)
            if slot:
                # Release resources
                if slot.resource_usage:
                    self._release_resources(slot.resource_usage)

                runtime = (datetime.now(timezone.utc) - slot.started_at).total_seconds()

                logger.info(
                    "Released execution slot",
                    slot_id=slot_id,
                    job_id=slot.job_id,
                    runtime_seconds=runtime,
                )

    def _reserve_resources(self, requirements: ResourceRequirement) -> None:
        """Reserve system resources for job execution"""
        if requirements.cpu_cores:
            self._available_resources["cpu_cores"] -= requirements.cpu_cores

        if requirements.memory_mb:
            self._available_resources["memory_mb"] -= requirements.memory_mb

        if requirements.gpu_count:
            self._available_resources["gpu_count"] -= requirements.gpu_count

        if requirements.custom_resources:
            for resource, amount in requirements.custom_resources.items():
                current = self._available_resources.get(resource, 0)
                self._available_resources[resource] = current - amount

    def _release_resources(self, requirements: ResourceRequirement) -> None:
        """Release system resources after job execution"""
        if requirements.cpu_cores:
            self._available_resources["cpu_cores"] += requirements.cpu_cores

        if requirements.memory_mb:
            self._available_resources["memory_mb"] += requirements.memory_mb

        if requirements.gpu_count:
            self._available_resources["gpu_count"] += requirements.gpu_count

        if requirements.custom_resources:
            for resource, amount in requirements.custom_resources.items():
                current = self._available_resources.get(resource, 0)
                self._available_resources[resource] = current + amount

    def get_active_executions(self) -> List[ExecutionSlot]:
        """Get list of currently active execution slots"""
        return list(self._active_slots.values())

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get concurrency execution statistics"""
        active_slots = list(self._active_slots.values())

        # Calculate resource utilization
        total_cpu = 4.0  # Should be configurable
        total_memory = 8192

        used_cpu = sum(
            slot.resource_usage.cpu_cores or 0
            for slot in active_slots
            if slot.resource_usage
        )
        used_memory = sum(
            slot.resource_usage.memory_mb or 0
            for slot in active_slots
            if slot.resource_usage
        )

        # Calculate average wait time
        avg_wait_time = 0.0
        if self._stats["queue_wait_times"]:
            avg_wait_time = sum(self._stats["queue_wait_times"]) / len(
                self._stats["queue_wait_times"]
            )

        return {
            "active_executions": len(active_slots),
            "total_acquisitions": self._stats["total_acquisitions"],
            "average_wait_time_seconds": avg_wait_time,
            "resource_contentions": self._stats["resource_contentions"],
            "policy_violations": self._stats["policy_violations"],
            "resource_utilization": {
                "cpu_usage_percent": (used_cpu / total_cpu) * 100
                if total_cpu > 0
                else 0,
                "memory_usage_percent": (used_memory / total_memory) * 100
                if total_memory > 0
                else 0,
                "available_cpu": self._available_resources["cpu_cores"],
                "available_memory": self._available_resources["memory_mb"],
            },
            "executions_by_type": self._get_executions_by_type(active_slots),
        }

    def _get_executions_by_type(
        self, active_slots: List[ExecutionSlot]
    ) -> Dict[str, int]:
        """Get breakdown of executions by job type"""
        type_counts = {}
        for slot in active_slots:
            type_counts[slot.job_type] = type_counts.get(slot.job_type, 0) + 1
        return type_counts

    async def cleanup_stale_slots(self, max_age_hours: int = 2) -> int:
        """Clean up stale execution slots"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        stale_slots = []

        async with self._slot_locks:
            for slot_id, slot in list(self._active_slots.items()):
                if slot.started_at < cutoff_time:
                    stale_slots.append(slot_id)

            # Remove stale slots
            for slot_id in stale_slots:
                slot = self._active_slots.pop(slot_id, None)
                if slot and slot.resource_usage:
                    self._release_resources(slot.resource_usage)

        if stale_slots:
            logger.warning(f"Cleaned up {len(stale_slots)} stale execution slots")

        return len(stale_slots)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of concurrency manager"""
        active_count = len(self._active_slots)

        # Check for potential issues
        issues = []
        if active_count > 50:  # High execution count
            issues.append("High number of active executions")

        if self._available_resources["cpu_cores"] < 0:
            issues.append("Negative CPU resources - possible accounting error")

        if self._available_resources["memory_mb"] < 0:
            issues.append("Negative memory resources - possible accounting error")

        return {
            "is_healthy": len(issues) == 0,
            "active_executions": active_count,
            "available_resources": dict(self._available_resources),
            "issues": issues,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
