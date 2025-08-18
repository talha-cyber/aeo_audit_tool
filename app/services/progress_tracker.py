"""
Comprehensive progress tracking system for audit runs.

Provides detailed progress snapshots with real-time updates, stage tracking,
timing information, performance metrics, and estimated completion times.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from app.core.audit_config import get_audit_settings
from app.models.audit import AuditRun
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ProgressStage(str, Enum):
    """Enumeration of audit processing stages"""

    INITIALIZING = "initializing"
    GENERATING_QUESTIONS = "generating_questions"
    PROCESSING_QUESTIONS = "processing_questions"
    DETECTING_BRANDS = "detecting_brands"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressSnapshot:
    """Detailed progress snapshot for an audit run"""

    audit_run_id: str
    stage: ProgressStage
    overall_progress: float  # 0.0 to 1.0
    stage_progress: float  # 0.0 to 1.0

    # Question processing details
    total_questions: int
    processed_questions: int
    failed_questions: int

    # Platform details
    platforms: List[str]
    platform_progress: Dict[str, float]
    platform_errors: Dict[str, int]
    platform_response_times: Dict[str, float]

    # Timing information
    started_at: datetime
    current_stage_started_at: datetime
    estimated_completion: Optional[datetime]

    # Performance metrics
    avg_question_time_ms: Optional[float]
    avg_platform_time_ms: Dict[str, float]

    # Current operation details
    current_operation: Optional[str]
    current_batch: Optional[int]
    total_batches: Optional[int]

    # Error information
    last_error: Optional[str]
    error_count: int

    # Resource usage
    memory_usage_mb: Optional[float]
    cpu_usage_percent: Optional[float]

    # Cost tracking
    estimated_cost_usd: float
    tokens_used: int


@dataclass
class StageMetrics:
    """Metrics for a specific processing stage"""

    stage: ProgressStage
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[int]
    success: bool
    error_message: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "stage": self.stage.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_message": self.error_message,
        }


class ProgressTracker:
    """Advanced progress tracking for audit runs with real-time updates"""

    def __init__(self, db_session, audit_run_id: str):
        self.db = db_session
        self.audit_run_id = audit_run_id
        self.settings = get_audit_settings()

        # Current state
        self.current_snapshot: Optional[ProgressSnapshot] = None
        self.stage_history: List[StageMetrics] = []

        # Timing tracking
        self.start_time = datetime.now(timezone.utc)
        self.stage_start_time = self.start_time
        self.question_times: List[float] = []
        self.platform_times: Dict[str, List[float]] = {}

        # Performance tracking
        self.error_history: List[Dict[str, Any]] = []
        self.performance_samples: List[Dict[str, Any]] = []

        logger.info("Progress tracker initialized", audit_run_id=audit_run_id)

    async def initialize(self):
        """Initialize progress tracking from database state"""
        await self._initialize_snapshot()
        await self.update_stage(ProgressStage.INITIALIZING, "Setting up audit run")

    async def update_stage(self, stage: ProgressStage, operation: Optional[str] = None):
        """Update the current processing stage"""
        previous_stage = self.current_snapshot.stage if self.current_snapshot else None

        # Complete previous stage if moving to a new one
        if previous_stage and previous_stage != stage:
            await self._complete_stage(previous_stage, True)

        # Start new stage
        self.stage_start_time = datetime.now(timezone.utc)

        if self.current_snapshot:
            self.current_snapshot.stage = stage
            self.current_snapshot.current_stage_started_at = self.stage_start_time
            self.current_snapshot.current_operation = operation

        # Record stage metrics
        stage_metric = StageMetrics(
            stage=stage,
            start_time=self.stage_start_time,
            end_time=None,
            duration_ms=None,
            success=False,  # Will be updated when stage completes
            error_message=None,
        )
        self.stage_history.append(stage_metric)

        await self._persist_progress()

        logger.info(
            "Stage updated",
            audit_run_id=self.audit_run_id,
            stage=stage.value,
            operation=operation,
        )

    async def update_question_progress(
        self,
        processed: int,
        total: int,
        current_batch: Optional[int] = None,
        total_batches: Optional[int] = None,
    ):
        """Update question processing progress"""
        if not self.current_snapshot:
            await self._initialize_snapshot()

        self.current_snapshot.processed_questions = processed
        self.current_snapshot.total_questions = total
        self.current_snapshot.current_batch = current_batch
        self.current_snapshot.total_batches = total_batches

        # Calculate overall progress based on stage weights
        stage_weights = {
            ProgressStage.INITIALIZING: 0.05,
            ProgressStage.GENERATING_QUESTIONS: 0.15,
            ProgressStage.PROCESSING_QUESTIONS: 0.70,
            ProgressStage.DETECTING_BRANDS: 0.08,
            ProgressStage.FINALIZING: 0.02,
        }

        current_stage_weight = stage_weights.get(self.current_snapshot.stage, 0.0)
        previous_stages_weight = sum(
            weight
            for s, weight in stage_weights.items()
            if list(stage_weights.keys()).index(s)
            < list(stage_weights.keys()).index(self.current_snapshot.stage)
        )

        if total > 0:
            stage_progress = processed / total
            self.current_snapshot.stage_progress = stage_progress
            self.current_snapshot.overall_progress = previous_stages_weight + (
                current_stage_weight * stage_progress
            )

        # Update estimated completion
        await self._calculate_estimated_completion()

        await self._persist_progress()

        logger.debug(
            "Question progress updated",
            audit_run_id=self.audit_run_id,
            processed=processed,
            total=total,
            overall_progress=f"{self.current_snapshot.overall_progress:.2%}",
        )

    async def update_platform_progress(
        self, platform: str, progress: float, response_time_ms: Optional[float] = None
    ):
        """Update progress for a specific platform"""
        if not self.current_snapshot:
            await self._initialize_snapshot()

        self.current_snapshot.platform_progress[platform] = progress

        if response_time_ms:
            self.current_snapshot.platform_response_times[platform] = response_time_ms

        await self._persist_progress()

    async def record_question_time(self, duration_ms: float):
        """Record timing for question processing"""
        self.question_times.append(duration_ms)

        # Keep only recent timings for better estimation
        if len(self.question_times) > 50:
            self.question_times = self.question_times[-50:]

        if self.current_snapshot:
            self.current_snapshot.avg_question_time_ms = sum(self.question_times) / len(
                self.question_times
            )

    async def record_platform_time(self, platform: str, duration_ms: float):
        """Record timing for platform queries"""
        if platform not in self.platform_times:
            self.platform_times[platform] = []

        self.platform_times[platform].append(duration_ms)

        # Keep only recent timings
        if len(self.platform_times[platform]) > 20:
            self.platform_times[platform] = self.platform_times[platform][-20:]

        if self.current_snapshot:
            self.current_snapshot.avg_platform_time_ms[platform] = sum(
                self.platform_times[platform]
            ) / len(self.platform_times[platform])

    async def record_error(
        self, platform: Optional[str] = None, error_message: Optional[str] = None
    ):
        """Record an error occurrence"""
        if not self.current_snapshot:
            await self._initialize_snapshot()

        self.current_snapshot.error_count += 1

        if platform:
            if platform not in self.current_snapshot.platform_errors:
                self.current_snapshot.platform_errors[platform] = 0
            self.current_snapshot.platform_errors[platform] += 1

        if error_message:
            self.current_snapshot.last_error = error_message

            # Add to error history
            error_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "platform": platform,
                "message": error_message,
                "stage": self.current_snapshot.stage.value,
            }
            self.error_history.append(error_record)

            # Keep only recent errors
            if len(self.error_history) > 100:
                self.error_history = self.error_history[-100:]

        await self._persist_progress()

    async def update_resource_usage(
        self, memory_mb: Optional[float] = None, cpu_percent: Optional[float] = None
    ):
        """Update resource usage metrics"""
        if not self.current_snapshot:
            await self._initialize_snapshot()

        if memory_mb is not None:
            self.current_snapshot.memory_usage_mb = memory_mb

        if cpu_percent is not None:
            self.current_snapshot.cpu_usage_percent = cpu_percent

        # Record performance sample
        sample = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "stage": self.current_snapshot.stage.value,
        }
        self.performance_samples.append(sample)

        # Keep only recent samples
        if len(self.performance_samples) > 100:
            self.performance_samples = self.performance_samples[-100:]

    async def update_cost_tracking(self, cost_usd: float, tokens: int):
        """Update cost and token usage tracking"""
        if not self.current_snapshot:
            await self._initialize_snapshot()

        self.current_snapshot.estimated_cost_usd += cost_usd
        self.current_snapshot.tokens_used += tokens

        await self._persist_progress()

    async def get_current_progress(self) -> Optional[ProgressSnapshot]:
        """Get current progress snapshot"""
        if not self.current_snapshot:
            await self._initialize_snapshot()

        return self.current_snapshot

    async def get_detailed_status(self) -> Dict[str, Any]:
        """Get comprehensive status information"""
        if not self.current_snapshot:
            await self._initialize_snapshot()

        return {
            "progress": asdict(self.current_snapshot),
            "stage_history": [stage.to_dict() for stage in self.stage_history],
            "error_history": self.error_history[-10:],  # Last 10 errors
            "performance_samples": self.performance_samples[-20:],  # Last 20 samples
            "timing_statistics": {
                "avg_question_time_ms": sum(self.question_times)
                / len(self.question_times)
                if self.question_times
                else None,
                "platform_times": {
                    platform: sum(times) / len(times)
                    for platform, times in self.platform_times.items()
                },
                "total_runtime_seconds": (
                    datetime.now(timezone.utc) - self.start_time
                ).total_seconds(),
            },
        }

    async def complete_stage(
        self, success: bool = True, error_message: Optional[str] = None
    ):
        """Mark current stage as completed"""
        if self.current_snapshot:
            await self._complete_stage(
                self.current_snapshot.stage, success, error_message
            )

    async def finalize_tracking(self, final_stage: ProgressStage, success: bool = True):
        """Finalize progress tracking"""
        if self.current_snapshot:
            self.current_snapshot.stage = final_stage
            self.current_snapshot.overall_progress = (
                1.0 if success else self.current_snapshot.overall_progress
            )

            await self._complete_stage(final_stage, success)
            await self._persist_progress()

        logger.info(
            "Progress tracking finalized",
            audit_run_id=self.audit_run_id,
            final_stage=final_stage.value,
            success=success,
        )

    # === Private methods ===

    async def _initialize_snapshot(self):
        """Initialize progress snapshot from database"""
        audit_run = (
            self.db.query(AuditRun).filter(AuditRun.id == self.audit_run_id).first()
        )

        if not audit_run:
            raise ValueError(f"Audit run {self.audit_run_id} not found")

        # Load existing progress or create new
        existing_progress = audit_run.progress_data or {}

        self.current_snapshot = ProgressSnapshot(
            audit_run_id=self.audit_run_id,
            stage=ProgressStage(
                existing_progress.get("stage", ProgressStage.INITIALIZING.value)
            ),
            overall_progress=existing_progress.get("overall_progress", 0.0),
            stage_progress=existing_progress.get("stage_progress", 0.0),
            total_questions=existing_progress.get("total_questions", 0),
            processed_questions=existing_progress.get("processed_questions", 0),
            failed_questions=existing_progress.get("failed_questions", 0),
            platforms=existing_progress.get("platforms", []),
            platform_progress=existing_progress.get("platform_progress", {}),
            platform_errors=existing_progress.get("platform_errors", {}),
            platform_response_times=existing_progress.get(
                "platform_response_times", {}
            ),
            started_at=audit_run.started_at or self.start_time,
            current_stage_started_at=self.stage_start_time,
            estimated_completion=None,
            avg_question_time_ms=existing_progress.get("avg_question_time_ms"),
            avg_platform_time_ms=existing_progress.get("avg_platform_time_ms", {}),
            current_operation=existing_progress.get("current_operation"),
            current_batch=existing_progress.get("current_batch"),
            total_batches=existing_progress.get("total_batches"),
            last_error=existing_progress.get("last_error"),
            error_count=existing_progress.get("error_count", 0),
            memory_usage_mb=existing_progress.get("memory_usage_mb"),
            cpu_usage_percent=existing_progress.get("cpu_usage_percent"),
            estimated_cost_usd=existing_progress.get("estimated_cost_usd", 0.0),
            tokens_used=existing_progress.get("tokens_used", 0),
        )

    async def _calculate_estimated_completion(self):
        """Calculate estimated completion time"""
        if not self.current_snapshot or self.current_snapshot.overall_progress <= 0:
            return

        elapsed_time = datetime.now(timezone.utc) - self.current_snapshot.started_at
        total_estimated_time = elapsed_time / self.current_snapshot.overall_progress

        self.current_snapshot.estimated_completion = (
            self.current_snapshot.started_at + total_estimated_time
        )

    async def _complete_stage(
        self, stage: ProgressStage, success: bool, error_message: Optional[str] = None
    ):
        """Complete a processing stage"""
        # Find the stage in history and update it
        for stage_metric in reversed(self.stage_history):
            if stage_metric.stage == stage and stage_metric.end_time is None:
                stage_metric.end_time = datetime.now(timezone.utc)
                stage_metric.duration_ms = int(
                    (stage_metric.end_time - stage_metric.start_time).total_seconds()
                    * 1000
                )
                stage_metric.success = success
                stage_metric.error_message = error_message
                break

    async def _persist_progress(self):
        """Persist current progress to database"""
        if not self.current_snapshot or not self.settings.AUDIT_PROGRESS_PERSISTENCE:
            return

        try:
            audit_run = (
                self.db.query(AuditRun).filter(AuditRun.id == self.audit_run_id).first()
            )
            if audit_run:
                # Convert snapshot to dict for JSON storage
                progress_dict = asdict(self.current_snapshot)

                # Convert datetime objects to ISO strings
                for key, value in progress_dict.items():
                    if isinstance(value, datetime):
                        progress_dict[key] = value.isoformat()

                # Add additional tracking data
                progress_dict["stage_history"] = [
                    stage.to_dict() for stage in self.stage_history[-10:]
                ]
                progress_dict["error_history"] = self.error_history[-5:]
                progress_dict["performance_samples"] = self.performance_samples[-10:]

                audit_run.progress_data = progress_dict
                audit_run.processed_questions = (
                    self.current_snapshot.processed_questions
                )
                audit_run.updated_at = datetime.now(timezone.utc)

                self.db.commit()

        except Exception as e:
            logger.error(
                "Failed to persist progress",
                audit_run_id=self.audit_run_id,
                error=str(e),
            )
            self.db.rollback()


class ProgressAggregator:
    """Aggregates progress data across multiple audit runs"""

    def __init__(self, db_session):
        self.db = db_session

    async def get_overall_system_progress(self) -> Dict[str, Any]:
        """Get system-wide progress statistics"""
        try:
            # Get running audits
            running_audits = (
                self.db.query(AuditRun).filter(AuditRun.status == "running").all()
            )

            # Calculate aggregate statistics
            total_running = len(running_audits)
            total_questions = sum(
                audit.total_questions or 0 for audit in running_audits
            )
            total_processed = sum(
                audit.processed_questions or 0 for audit in running_audits
            )

            overall_progress = (
                (total_processed / total_questions * 100) if total_questions > 0 else 0
            )

            # Platform distribution
            platform_stats = {}
            for audit in running_audits:
                if audit.platform_stats:
                    for platform, stats in audit.platform_stats.items():
                        if platform not in platform_stats:
                            platform_stats[platform] = {"audits": 0, "total_queries": 0}
                        platform_stats[platform]["audits"] += 1
                        platform_stats[platform]["total_queries"] += stats.get(
                            "total_queries", 0
                        )

            return {
                "total_running_audits": total_running,
                "overall_progress_percent": round(overall_progress, 2),
                "total_questions": total_questions,
                "total_processed": total_processed,
                "platform_distribution": platform_stats,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error("Failed to get system progress", error=str(e))
            return {
                "error": str(e),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }


def create_progress_tracker(db_session, audit_run_id: str) -> ProgressTracker:
    """Factory function to create a progress tracker"""
    return ProgressTracker(db_session, audit_run_id)
