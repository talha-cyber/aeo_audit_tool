"""
Priority-based job scheduling and queue management.

Provides sophisticated priority handling with multiple priority levels,
aging mechanisms, and fair scheduling algorithms.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from heapq import heapify, heappop, heappush
from typing import Any, Dict, List, Optional, Tuple

from app.models.scheduling import ScheduledJob
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PriorityLevel(IntEnum):
    """Standard priority levels (lower number = higher priority)"""

    CRITICAL = 1  # System critical jobs
    HIGH = 2  # High priority business jobs
    NORMAL = 5  # Default priority
    LOW = 8  # Background/cleanup jobs
    BULK = 10  # Bulk processing jobs


@dataclass
class PriorityJobWrapper:
    """Wrapper for jobs in priority queue"""

    priority: int
    scheduled_time: datetime
    job_id: str
    job: ScheduledJob
    queue_entry_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    aging_factor: float = 0.0

    def __lt__(self, other) -> bool:
        """Comparison for heap queue (lower priority value = higher priority)"""
        # Primary sort by effective priority (including aging)
        effective_priority_self = self.priority - self.aging_factor
        effective_priority_other = other.priority - other.aging_factor

        if effective_priority_self != effective_priority_other:
            return effective_priority_self < effective_priority_other

        # Secondary sort by scheduled time (earlier = higher priority)
        if self.scheduled_time != other.scheduled_time:
            return self.scheduled_time < other.scheduled_time

        # Tertiary sort by queue entry time for fairness
        return self.queue_entry_time < other.queue_entry_time


class PriorityPolicy:
    """
    Priority scheduling policy configuration.

    Defines how job priorities are calculated, aged, and managed
    for fair and efficient scheduling.
    """

    def __init__(
        self,
        enable_aging: bool = True,
        aging_rate: float = 0.1,  # Priority reduction per hour of waiting
        max_aging: float = 3.0,  # Maximum priority reduction from aging
        priority_boost_conditions: Optional[Dict[str, float]] = None,
        deadline_priority_boost: bool = True,
        fair_share_enabled: bool = False,
    ):
        """
        Initialize priority policy.

        Args:
            enable_aging: Enable priority aging to prevent starvation
            aging_rate: Rate of priority improvement per hour waiting
            max_aging: Maximum priority improvement from aging
            priority_boost_conditions: Conditions that boost job priority
            deadline_priority_boost: Boost priority as deadline approaches
            fair_share_enabled: Enable fair share scheduling
        """
        self.enable_aging = enable_aging
        self.aging_rate = aging_rate
        self.max_aging = max_aging
        self.priority_boost_conditions = priority_boost_conditions or {}
        self.deadline_priority_boost = deadline_priority_boost
        self.fair_share_enabled = fair_share_enabled

        # Fair share tracking
        self._job_type_execution_counts: Dict[str, int] = {}
        self._job_type_fair_shares: Dict[str, float] = {}

    def calculate_effective_priority(
        self,
        job: ScheduledJob,
        queue_time: datetime,
        current_time: Optional[datetime] = None,
    ) -> float:
        """
        Calculate effective priority including aging and boosts.

        Args:
            job: Job to calculate priority for
            queue_time: When job entered queue
            current_time: Current time for calculations

        Returns:
            Effective priority (lower = higher priority)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        base_priority = float(job.priority)
        effective_priority = base_priority

        # Apply aging if enabled
        if self.enable_aging:
            wait_time_hours = (current_time - queue_time).total_seconds() / 3600
            aging_bonus = min(wait_time_hours * self.aging_rate, self.max_aging)
            effective_priority -= aging_bonus

        # Apply deadline boost if enabled and job has deadline
        if self.deadline_priority_boost and hasattr(job, "deadline"):
            deadline = getattr(job, "deadline")
            if deadline:
                time_to_deadline = (deadline - current_time).total_seconds() / 3600
                if time_to_deadline < 24:  # Less than 24 hours
                    deadline_boost = (
                        max(0, (24 - time_to_deadline) / 24) * 2.0
                    )  # Up to 2 priority points
                    effective_priority -= deadline_boost

        # Apply conditional boosts
        for condition, boost_amount in self.priority_boost_conditions.items():
            if self._check_boost_condition(job, condition):
                effective_priority -= boost_amount

        # Apply fair share adjustment
        if self.fair_share_enabled:
            fair_share_adjustment = self._calculate_fair_share_adjustment(job)
            effective_priority += fair_share_adjustment

        # Ensure priority stays within reasonable bounds
        effective_priority = max(0.1, effective_priority)  # Minimum priority

        return effective_priority

    def _check_boost_condition(self, job: ScheduledJob, condition: str) -> bool:
        """Check if job meets boost condition"""
        if condition == "retry_job":
            # Boost retry attempts
            return hasattr(job, "retry_count") and getattr(job, "retry_count", 0) > 0

        elif condition == "user_facing":
            # Boost jobs affecting users
            return job.job_type in [
                "user_report",
                "user_notification",
                "dashboard_update",
            ]

        elif condition == "dependency_blocker":
            # Boost jobs that other jobs depend on
            return (
                hasattr(job, "dependent_job_count")
                and getattr(job, "dependent_job_count", 0) > 0
            )

        elif condition.startswith("tag:"):
            # Boost jobs with specific tags
            tag = condition[4:]  # Remove "tag:" prefix
            return tag in (job.tags or [])

        return False

    def _calculate_fair_share_adjustment(self, job: ScheduledJob) -> float:
        """Calculate fair share priority adjustment"""
        job_type = job.job_type

        # Get execution counts
        executed_count = self._job_type_execution_counts.get(job_type, 0)
        total_executed = sum(self._job_type_execution_counts.values())

        if total_executed == 0:
            return 0.0

        # Calculate actual share
        actual_share = executed_count / total_executed

        # Get configured fair share (default to equal share)
        total_types = len(self._job_type_execution_counts)
        expected_share = self._job_type_fair_shares.get(
            job_type, 1.0 / total_types if total_types > 0 else 1.0
        )

        # If job type is under-represented, reduce priority (boost)
        # If over-represented, increase priority (penalty)
        share_difference = actual_share - expected_share

        # Scale the adjustment (max ±2 priority points)
        adjustment = share_difference * 10.0  # Scale factor
        adjustment = max(-2.0, min(2.0, adjustment))

        return adjustment

    def record_job_execution(self, job: ScheduledJob) -> None:
        """Record job execution for fair share calculations"""
        if self.fair_share_enabled:
            job_type = job.job_type
            self._job_type_execution_counts[job_type] = (
                self._job_type_execution_counts.get(job_type, 0) + 1
            )

    def set_fair_share(self, job_type: str, share: float) -> None:
        """Set fair share percentage for job type"""
        if 0.0 <= share <= 1.0:
            self._job_type_fair_shares[job_type] = share
        else:
            raise ValueError(f"Fair share must be between 0.0 and 1.0, got {share}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get priority policy statistics"""
        return {
            "enable_aging": self.enable_aging,
            "aging_rate": self.aging_rate,
            "max_aging": self.max_aging,
            "fair_share_enabled": self.fair_share_enabled,
            "job_type_executions": dict(self._job_type_execution_counts),
            "fair_share_configuration": dict(self._job_type_fair_shares),
            "priority_boost_conditions": dict(self.priority_boost_conditions),
        }


class PriorityQueue:
    """
    Thread-safe priority queue for job scheduling.

    Implements a heap-based priority queue with aging, fair share,
    and advanced scheduling features.
    """

    def __init__(self, policy: Optional[PriorityPolicy] = None):
        """Initialize priority queue"""
        self.policy = policy or PriorityPolicy()
        self._heap: List[PriorityJobWrapper] = []
        self._job_entries: Dict[str, PriorityJobWrapper] = {}
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "total_enqueued": 0,
            "total_dequeued": 0,
            "current_size": 0,
            "priority_distribution": {level.name: 0 for level in PriorityLevel},
            "average_wait_time": 0.0,
            "max_wait_time": 0.0,
        }

    def enqueue(
        self, job: ScheduledJob, scheduled_time: Optional[datetime] = None
    ) -> None:
        """
        Add job to priority queue.

        Args:
            job: Job to add to queue
            scheduled_time: When job was scheduled to run
        """
        if scheduled_time is None:
            scheduled_time = datetime.now(timezone.utc)

        with self._lock:
            # Remove existing entry if job already queued
            if job.job_id in self._job_entries:
                self._remove_entry_unsafe(job.job_id)

            # Create wrapper
            wrapper = PriorityJobWrapper(
                priority=job.priority,
                scheduled_time=scheduled_time,
                job_id=job.job_id,
                job=job,
                queue_entry_time=datetime.now(timezone.utc),
            )

            # Add to heap and tracking
            heappush(self._heap, wrapper)
            self._job_entries[job.job_id] = wrapper

            # Update statistics
            self._stats["total_enqueued"] += 1
            self._stats["current_size"] = len(self._heap)

            priority_name = self._get_priority_name(job.priority)
            self._stats["priority_distribution"][priority_name] += 1

            logger.debug(
                "Job enqueued",
                job_id=job.job_id,
                priority=job.priority,
                queue_size=len(self._heap),
            )

    def dequeue(self) -> Optional[ScheduledJob]:
        """
        Remove and return highest priority job from queue.

        Returns:
            Highest priority job or None if queue is empty
        """
        with self._lock:
            # Update aging for all jobs in queue
            if self.policy.enable_aging:
                self._update_aging_factors()

            # Re-heapify if aging changed priorities significantly
            if self.policy.enable_aging and len(self._heap) > 10:
                heapify(self._heap)

            # Get highest priority job
            while self._heap:
                wrapper = heappop(self._heap)

                # Verify job still exists in tracking (handle concurrent removal)
                if wrapper.job_id in self._job_entries:
                    # Remove from tracking
                    del self._job_entries[wrapper.job_id]

                    # Update statistics
                    self._stats["total_dequeued"] += 1
                    self._stats["current_size"] = len(self._heap)

                    wait_time = (
                        datetime.now(timezone.utc) - wrapper.queue_entry_time
                    ).total_seconds()
                    self._update_wait_time_stats(wait_time)

                    priority_name = self._get_priority_name(wrapper.priority)
                    self._stats["priority_distribution"][priority_name] -= 1

                    # Record execution for fair share
                    self.policy.record_job_execution(wrapper.job)

                    logger.debug(
                        "Job dequeued",
                        job_id=wrapper.job_id,
                        priority=wrapper.priority,
                        wait_time_seconds=wait_time,
                        queue_size=len(self._heap),
                    )

                    return wrapper.job

            return None

    def peek(self) -> Optional[ScheduledJob]:
        """
        Get highest priority job without removing it.

        Returns:
            Highest priority job or None if queue is empty
        """
        with self._lock:
            if self._heap:
                # Update aging before peeking
                if self.policy.enable_aging:
                    self._update_aging_factors()
                    heapify(self._heap)

                return self._heap[0].job
            return None

    def remove(self, job_id: str) -> bool:
        """
        Remove specific job from queue.

        Args:
            job_id: ID of job to remove

        Returns:
            True if job was removed, False if not found
        """
        with self._lock:
            return self._remove_entry_unsafe(job_id)

    def _remove_entry_unsafe(self, job_id: str) -> bool:
        """Remove entry without locking (internal use)"""
        if job_id not in self._job_entries:
            return False

        wrapper = self._job_entries[job_id]
        del self._job_entries[job_id]

        # Mark as removed in heap (will be cleaned up during dequeue)
        wrapper.job_id = "__REMOVED__"

        # Update statistics
        self._stats["current_size"] = len(self._job_entries)
        priority_name = self._get_priority_name(wrapper.priority)
        self._stats["priority_distribution"][priority_name] -= 1

        return True

    def _update_aging_factors(self) -> None:
        """Update aging factors for all jobs in queue"""
        current_time = datetime.now(timezone.utc)

        for wrapper in self._heap:
            if wrapper.job_id == "__REMOVED__":
                continue

            queue_time = wrapper.queue_entry_time
            wait_hours = (current_time - queue_time).total_seconds() / 3600

            aging_bonus = min(
                wait_hours * self.policy.aging_rate, self.policy.max_aging
            )
            wrapper.aging_factor = aging_bonus

    def _update_wait_time_stats(self, wait_time_seconds: float) -> None:
        """Update wait time statistics"""
        # Simple running average (could be improved with more sophisticated tracking)
        current_avg = self._stats["average_wait_time"]
        total_dequeued = self._stats["total_dequeued"]

        # Update average
        if total_dequeued == 1:
            self._stats["average_wait_time"] = wait_time_seconds
        else:
            self._stats["average_wait_time"] = (
                current_avg * (total_dequeued - 1) + wait_time_seconds
            ) / total_dequeued

        # Update max
        if wait_time_seconds > self._stats["max_wait_time"]:
            self._stats["max_wait_time"] = wait_time_seconds

    def _get_priority_name(self, priority: int) -> str:
        """Get human-readable priority name"""
        for level in PriorityLevel:
            if level.value == priority:
                return level.name
        return "CUSTOM"

    def size(self) -> int:
        """Get current queue size"""
        with self._lock:
            return len(self._job_entries)

    def is_empty(self) -> bool:
        """Check if queue is empty"""
        with self._lock:
            return len(self._job_entries) == 0

    def clear(self) -> int:
        """Clear all jobs from queue"""
        with self._lock:
            cleared_count = len(self._job_entries)
            self._heap.clear()
            self._job_entries.clear()
            self._stats["current_size"] = 0

            # Reset priority distribution
            for priority_name in self._stats["priority_distribution"]:
                self._stats["priority_distribution"][priority_name] = 0

            return cleared_count

    def get_queue_contents(self) -> List[Dict[str, Any]]:
        """Get snapshot of current queue contents"""
        with self._lock:
            current_time = datetime.now(timezone.utc)
            contents = []

            for wrapper in sorted(self._heap):
                if wrapper.job_id == "__REMOVED__":
                    continue

                wait_time = (current_time - wrapper.queue_entry_time).total_seconds()
                effective_priority = self.policy.calculate_effective_priority(
                    wrapper.job, wrapper.queue_entry_time, current_time
                )

                contents.append(
                    {
                        "job_id": wrapper.job_id,
                        "job_name": wrapper.job.name,
                        "job_type": wrapper.job.job_type,
                        "base_priority": wrapper.priority,
                        "effective_priority": effective_priority,
                        "aging_factor": wrapper.aging_factor,
                        "scheduled_time": wrapper.scheduled_time.isoformat(),
                        "queue_entry_time": wrapper.queue_entry_time.isoformat(),
                        "wait_time_seconds": wait_time,
                    }
                )

            return contents

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics"""
        with self._lock:
            stats = dict(self._stats)

            # Add current queue analysis
            if self._job_entries:
                current_time = datetime.now(timezone.utc)
                wait_times = [
                    (current_time - wrapper.queue_entry_time).total_seconds()
                    for wrapper in self._job_entries.values()
                ]

                stats.update(
                    {
                        "current_min_wait": min(wait_times),
                        "current_max_wait": max(wait_times),
                        "current_avg_wait": sum(wait_times) / len(wait_times),
                    }
                )

            # Add policy statistics
            stats["policy"] = self.policy.get_statistics()

            return stats


class PriorityManager:
    """
    High-level priority management for job scheduling system.

    Coordinates priority queues, policies, and provides unified
    interface for priority-based scheduling.
    """

    def __init__(self, default_policy: Optional[PriorityPolicy] = None):
        """Initialize priority manager"""
        self.default_policy = default_policy or PriorityPolicy()
        self.main_queue = PriorityQueue(self.default_policy)

        # Multiple queues for different categories
        self._category_queues: Dict[str, PriorityQueue] = {}
        self._category_policies: Dict[str, PriorityPolicy] = {}

        # Statistics
        self._global_stats = {
            "total_jobs_processed": 0,
            "priority_escalations": 0,
            "policy_changes": 0,
        }

    def create_category_queue(
        self, category: str, policy: Optional[PriorityPolicy] = None
    ) -> None:
        """Create priority queue for specific job category"""
        if policy is None:
            policy = PriorityPolicy()

        self._category_queues[category] = PriorityQueue(policy)
        self._category_policies[category] = policy

        logger.info(f"Created priority queue for category: {category}")

    def enqueue_job(
        self,
        job: ScheduledJob,
        category: Optional[str] = None,
        scheduled_time: Optional[datetime] = None,
    ) -> None:
        """Enqueue job in appropriate priority queue"""
        if category and category in self._category_queues:
            queue = self._category_queues[category]
        else:
            queue = self.main_queue

        queue.enqueue(job, scheduled_time)
        logger.debug(f"Job enqueued in {category or 'main'} queue", job_id=job.job_id)

    def dequeue_highest_priority_job(self) -> Optional[Tuple[ScheduledJob, str]]:
        """
        Dequeue highest priority job from all queues.

        Returns:
            Tuple of (job, queue_category) or None if no jobs available
        """
        highest_priority_job = None
        highest_priority = float("inf")
        selected_queue = None
        selected_category = "main"

        # Check main queue
        main_job = self.main_queue.peek()
        if main_job:
            priority = self.default_policy.calculate_effective_priority(
                main_job, datetime.now(timezone.utc)
            )
            if priority < highest_priority:
                highest_priority = priority
                highest_priority_job = main_job
                selected_queue = self.main_queue
                selected_category = "main"

        # Check category queues
        for category, queue in self._category_queues.items():
            job = queue.peek()
            if job:
                policy = self._category_policies[category]
                priority = policy.calculate_effective_priority(
                    job, datetime.now(timezone.utc)
                )
                if priority < highest_priority:
                    highest_priority = priority
                    highest_priority_job = job
                    selected_queue = queue
                    selected_category = category

        # Dequeue the selected job
        if selected_queue and highest_priority_job:
            dequeued_job = selected_queue.dequeue()
            if dequeued_job:
                self._global_stats["total_jobs_processed"] += 1
                return dequeued_job, selected_category

        return None

    def escalate_job_priority(
        self, job_id: str, new_priority: int, reason: str = "Manual escalation"
    ) -> bool:
        """Escalate job priority across all queues"""
        escalated = False

        # Check main queue
        if job_id in self.main_queue._job_entries:
            wrapper = self.main_queue._job_entries[job_id]
            old_priority = wrapper.priority
            wrapper.priority = new_priority
            wrapper.job.priority = new_priority
            escalated = True

            logger.info(
                "Escalated job priority in main queue",
                job_id=job_id,
                old_priority=old_priority,
                new_priority=new_priority,
                reason=reason,
            )

        # Check category queues
        for category, queue in self._category_queues.items():
            if job_id in queue._job_entries:
                wrapper = queue._job_entries[job_id]
                old_priority = wrapper.priority
                wrapper.priority = new_priority
                wrapper.job.priority = new_priority
                escalated = True

                logger.info(
                    f"Escalated job priority in {category} queue",
                    job_id=job_id,
                    old_priority=old_priority,
                    new_priority=new_priority,
                    reason=reason,
                )

        if escalated:
            self._global_stats["priority_escalations"] += 1

        return escalated

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get statistics across all priority queues"""
        stats = {
            "global": dict(self._global_stats),
            "main_queue": self.main_queue.get_statistics(),
            "category_queues": {},
        }

        for category, queue in self._category_queues.items():
            stats["category_queues"][category] = queue.get_statistics()

        # Calculate totals across all queues
        total_queued = stats["main_queue"]["current_size"]
        for category_stats in stats["category_queues"].values():
            total_queued += category_stats["current_size"]

        stats["global"]["total_jobs_queued"] = total_queued

        return stats

    def cleanup_empty_categories(self) -> List[str]:
        """Remove empty category queues to free resources"""
        empty_categories = []

        for category, queue in list(self._category_queues.items()):
            if queue.is_empty():
                empty_categories.append(category)
                del self._category_queues[category]
                del self._category_policies[category]

        if empty_categories:
            logger.info(f"Cleaned up empty category queues: {empty_categories}")

        return empty_categories
