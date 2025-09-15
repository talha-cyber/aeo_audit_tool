"""
Main audit processor orchestrator for AEO competitive intelligence audits.

This module coordinates the entire audit workflow from question generation to brand
detection, managing multi-platform AI queries with proper concurrency, rate limiting,
and progress tracking.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

# Helper to make data JSON serializable (UUID/datetime)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


from app.core.audit_config import get_audit_settings
from app.models import audit as audit_models
from app.models import question as question_models
from app.models import response as response_models
from app.services.audit_context import (
    add_audit_context,
    add_stage_context,
    contextual_logger,
)
from app.services.audit_metrics import get_audit_metrics
from app.services.brand_detection.core.detector import (
    BrandDetectionEngine,
    DetectionConfig,
)
from app.services.platform_manager import PlatformManager
from app.services.question_engine import QuestionContext, QuestionEngine
from app.utils.error_handler import AuditConfigurationError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AuditStatus(Enum):
    """Audit run status enumeration"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingResult:
    """Result from processing a single question"""

    question_id: str
    platform: str
    success: bool
    processing_time_ms: int
    error: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    response_length: Optional[int] = None


@dataclass
class BatchProgress:
    """Progress tracking for a batch of questions"""

    batch_index: int
    total_batches: int
    questions_in_batch: int
    successful_responses: int
    failed_responses: int
    batch_duration_ms: int
    estimated_completion: Optional[str] = None


class AuditProcessor:
    """
    Main orchestrator for AEO competitive intelligence audits.

    Coordinates question generation, multi-platform querying, brand detection,
    and progress tracking with comprehensive error handling and observability.
    """

    def __init__(self, db: Session, platform_manager: Optional[PlatformManager] = None):
        self.db = db
        self.platform_manager = platform_manager or PlatformManager()
        self.question_engine = QuestionEngine()
        self.brand_detector = None  # Will be initialized when needed
        self.metrics = get_audit_metrics()
        self.settings = get_audit_settings()

        # Configuration from settings
        self.batch_size = self.settings.AUDIT_BATCH_SIZE
        self.max_questions = self.settings.AUDIT_MAX_QUESTIONS
        self.inter_batch_delay = self.settings.AUDIT_INTER_BATCH_DELAY
        self.platform_timeout = self.settings.AUDIT_PLATFORM_TIMEOUT_SECONDS

    async def run_audit(self, audit_run_id: str) -> str:
        """
        Execute complete audit process with full error handling and progress tracking.

        Args:
            audit_run_id: ID of the AuditRun to execute

        Returns:
            audit_run_id: Same ID for confirmation

        Raises:
            ValueError: If audit run not found or invalid configuration
            Exception: For unexpected errors (logged and status updated)
        """
        # Add audit context to all logs in this execution
        with add_audit_context(audit_run_id=audit_run_id):
            contextual_logger.info("Starting audit run execution")

            # Load and validate audit run
            audit_run = await self._load_and_validate_audit_run(audit_run_id)

            try:
                # Update status to running
                await self._update_audit_status(audit_run, AuditStatus.RUNNING)
                self.metrics.increment_audit_started()

                # Prepare execution context
                context = await self._prepare_execution_context(audit_run)

                # Generate and prioritize questions
                questions = await self._generate_questions(audit_run, context)

                # Set total work and persist questions
                total_work = len(questions) * len(context["platforms"])
                await self._update_total_questions(audit_run, total_work)
                await self._persist_questions(audit_run.id, questions)

                # Process questions in batches across platforms
                results = await self._process_questions_batched(
                    audit_run, questions, context
                )

                # Finalize audit run
                await self._finalize_audit_run(
                    audit_run, results, AuditStatus.COMPLETED
                )

                contextual_logger.info(
                    "Audit run completed successfully",
                    processed_responses=len([r for r in results if r.success]),
                    failed_responses=len([r for r in results if not r.success]),
                    total_questions=len(questions),
                    platforms=len(context["platforms"]),
                )

                self.metrics.increment_audit_completed()
                return audit_run_id

            except Exception as e:
                contextual_logger.error(
                    "Audit run failed with exception",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )

                await self._finalize_audit_run(
                    audit_run, [], AuditStatus.FAILED, str(e)
                )
                self.metrics.increment_audit_failed()
                raise

    async def _load_and_validate_audit_run(
        self, audit_run_id: str
    ) -> audit_models.AuditRun:
        """Load audit run and validate it's ready for execution"""
        audit_run = (
            self.db.query(audit_models.AuditRun)
            .filter(audit_models.AuditRun.id == audit_run_id)
            .first()
        )

        if not audit_run:
            raise ValueError(f"Audit run {audit_run_id} not found")

        if audit_run.status not in [
            AuditStatus.PENDING.value,
            AuditStatus.RUNNING.value,
        ]:
            raise ValueError(
                f"Audit run {audit_run_id} is not in executable state: "
                f"{audit_run.status}"
            )

        # Validate configuration
        config = audit_run.config or {}
        if not config.get("client"):
            raise ValueError("Audit run missing client configuration")

        contextual_logger.info(
            "Audit run loaded and validated", status=audit_run.status
        )
        return audit_run

    async def _prepare_execution_context(
        self, audit_run: audit_models.AuditRun
    ) -> Dict[str, Any]:
        """Prepare all context needed for audit execution"""
        with add_stage_context("context_preparation"):
            config = audit_run.config
            client = config["client"]

            # Determine target brands (client + competitors)
            target_brands = [client["name"]]
            if client.get("competitors"):
                target_brands.extend(client["competitors"])

            # Get available platforms
            selected_platforms = config.get("platforms", [])
            available_platforms = []

            for platform_name in selected_platforms:
                if self.platform_manager.is_platform_available(platform_name):
                    available_platforms.append(platform_name)
                else:
                    contextual_logger.warning(
                        "Platform not available, skipping", platform=platform_name
                    )

            if not available_platforms:
                raise ValueError("No platforms available for audit execution")

            contextual_logger.info(
                "Execution context prepared",
                client_name=client["name"],
                target_brands=target_brands,
                available_platforms=available_platforms,
                selected_categories=config.get("question_categories", []),
            )

            return {
                "client": client,
                "target_brands": target_brands,
                "platforms": available_platforms,
                "categories": config.get("question_categories", []),
                "language": config.get("language", "en"),
            }

    async def _generate_questions(
        self, audit_run: audit_models.AuditRun, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate and prioritize questions for the audit"""

        with add_stage_context("question_generation"):
            start_time = time.time()

            # Create question context for the engine
            question_context = QuestionContext(
                client_brand=context["client"]["name"],
                competitors=context["client"].get("competitors", []),
                industry=context["client"]["industry"],
                product_type=context["client"].get("product_type"),
                audit_run_id=uuid.UUID(audit_run.id),
            )

            # Generate questions via QuestionEngine
            raw_questions = await self.question_engine.generate_questions(
                client_brand=question_context.client_brand,
                competitors=question_context.competitors,
                industry=question_context.industry,
                product_type=question_context.product_type,
                audit_run_id=question_context.audit_run_id,
                language=context.get("language", "en"),
                max_questions=context.get("max_questions", 100),
            )

            # Convert to dict format for further processing
            questions_list = []
            for question in raw_questions:
                question_dict = {
                    "question": question.question_text,
                    "category": question.category,
                    "question_type": getattr(question, "question_type", "general"),
                    "priority_score": getattr(question, "priority_score", 5.0),
                    "target_brand": getattr(question, "target_brand", None),
                    "provider": question.provider,
                    "metadata": question.model_dump()
                    if hasattr(question, "model_dump")
                    else {},
                }
                questions_list.append(question_dict)

            # Apply prioritization and limit
            prioritized_questions = self._prioritize_questions(
                questions_list, self.max_questions
            )

            generation_time = (time.time() - start_time) * 1000

            contextual_logger.info(
                "Questions generated and prioritized",
                raw_question_count=len(questions_list),
                prioritized_count=len(prioritized_questions),
                generation_time_ms=int(generation_time),
            )

            self.metrics.record_question_generation_time(generation_time)

            return prioritized_questions

    def _prioritize_questions(
        self, questions: List[Dict[str, Any]], max_questions: int
    ) -> List[Dict[str, Any]]:
        """Prioritize questions based on strategic value"""

        # Priority scoring weights
        category_weights = {
            "comparison": 10,  # High value - direct competitive intelligence
            "recommendation": 9,  # High value - recommendation scenarios
            "alternatives": 8,  # High value - competitor displacement
            "features": 6,  # Medium value - feature positioning
            "pricing": 5,  # Medium value - pricing intelligence
        }

        # Score each question
        for question in questions:
            base_score = category_weights.get(question.get("category", ""), 5)

            # Boost score for certain question types
            question_type = question.get("question_type", "")
            if question_type == "industry_general":
                question["priority_score"] = base_score + 2
            elif question_type == "alternative_seeking":
                question["priority_score"] = base_score + 1
            else:
                question["priority_score"] = base_score

        # Sort by priority and return top questions
        sorted_questions = sorted(
            questions, key=lambda x: x.get("priority_score", 0), reverse=True
        )
        return sorted_questions[:max_questions]

    async def _persist_questions(
        self, audit_run_id: str, questions: List[Dict[str, Any]]
    ) -> None:
        """Persist generated questions to database"""
        with add_stage_context("question_persistence"):
            question_records = []

            for question_data in questions:
                serialized_metadata = _to_jsonable(question_data.get("metadata", {}))
                question_record = question_models.Question(
                    id=str(uuid.uuid4()),
                    audit_run_id=audit_run_id,
                    question_text=question_data["question"],
                    category=question_data.get("category", "unknown"),
                    question_type=question_data.get("question_type", "unknown"),
                    priority_score=question_data.get("priority_score", 0.0),
                    target_brand=question_data.get("target_brand"),
                    provider=question_data.get("provider", "question_engine"),
                    question_metadata=serialized_metadata,
                )
                question_records.append(question_record)

            self.db.bulk_save_objects(question_records)
            self.db.commit()

            contextual_logger.info(
                f"Persisted {len(question_records)} questions to database"
            )
            self.metrics.increment_database_operation("bulk_insert", "questions")

    async def _process_questions_batched(
        self,
        audit_run: audit_models.AuditRun,
        questions: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[ProcessingResult]:
        """Process questions in batches across platforms with progress tracking"""

        with add_stage_context("batch_processing"):
            all_results = []
            total_batches = (len(questions) + self.batch_size - 1) // self.batch_size

            contextual_logger.info(
                "Starting batched question processing",
                total_questions=len(questions),
                batch_size=self.batch_size,
                total_batches=total_batches,
                platforms=context["platforms"],
            )

            # Initialize brand detector if needed
            if not self.brand_detector and not self.settings.AUDIT_SKIP_BRAND_DETECTION:
                self.brand_detector = BrandDetectionEngine(
                    openai_api_key=getattr(
                        self.settings, "OPENAI_API_KEY", "dummy_key"
                    ),
                    config=DetectionConfig(
                        confidence_threshold=self.settings.AUDIT_BRAND_CONFIDENCE_THRESHOLD,
                        max_context_window=200,
                        enable_caching=True,
                    ),
                )

            for batch_index in range(0, len(questions), self.batch_size):
                batch_start_time = time.time()
                batch_questions = questions[batch_index : batch_index + self.batch_size]
                batch_num = (batch_index // self.batch_size) + 1

                contextual_logger.info(
                    "Processing batch",
                    batch_number=batch_num,
                    total_batches=total_batches,
                    questions_in_batch=len(batch_questions),
                )

                # Create tasks for all platform-question combinations
                tasks = []
                for question_data in batch_questions:
                    for platform_name in context["platforms"]:
                        task = self._process_single_question(
                            audit_run_id=audit_run.id,
                            question_data=question_data,
                            platform_name=platform_name,
                            target_brands=context["target_brands"],
                        )
                        tasks.append(task)

                # Execute batch with timeout
                try:
                    batch_results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=self.platform_timeout * len(tasks),
                    )
                except asyncio.TimeoutError:
                    contextual_logger.error(
                        "Batch processing timeout",
                        batch_number=batch_num,
                        timeout_seconds=self.platform_timeout * len(tasks),
                    )
                    batch_results = [None] * len(tasks)

                # Filter and collect results
                valid_results = [
                    result
                    for result in batch_results
                    if isinstance(result, ProcessingResult)
                ]
                all_results.extend(valid_results)

                # Track batch progress
                batch_duration = int((time.time() - batch_start_time) * 1000)
                progress = BatchProgress(
                    batch_index=batch_num,
                    total_batches=total_batches,
                    questions_in_batch=len(batch_questions),
                    successful_responses=len([r for r in valid_results if r.success]),
                    failed_responses=len([r for r in valid_results if not r.success]),
                    batch_duration_ms=batch_duration,
                )

                # Update progress in database
                await self._update_progress(audit_run, len(all_results), progress)

                # Record metrics
                self.metrics.record_batch_processing_time(batch_duration)
                self.metrics.record_batch_size(len(batch_questions))

                # Inter-batch delay for rate limiting
                if batch_num < total_batches:
                    await asyncio.sleep(self.inter_batch_delay)

            contextual_logger.info(
                "Batched processing completed",
                total_results=len(all_results),
                successful=len([r for r in all_results if r.success]),
                failed=len([r for r in all_results if not r.success]),
            )

            return all_results

    async def _process_single_question(
        self,
        audit_run_id: str,
        question_data: Dict[str, Any],
        platform_name: str,
        target_brands: List[str],
    ) -> Optional[ProcessingResult]:
        """Process a single question on a specific platform"""

        question_text = question_data["question"]
        start_time = time.time()

        try:
            # Check if we're in mock mode
            if self.settings.AUDIT_MOCK_AI_RESPONSES:
                return await self._process_mock_response(
                    audit_run_id,
                    question_data,
                    platform_name,
                    target_brands,
                    start_time,
                )

            # Get platform client
            platform = self.platform_manager.get_platform(platform_name)

            # Execute query with platform (using context manager to initialize session)
            async with platform:
                response_envelope = await platform.safe_query(question_text)

            processing_time = int((time.time() - start_time) * 1000)

            if not response_envelope.get("success"):
                contextual_logger.warning(
                    "Platform query failed",
                    platform=platform_name,
                    question_preview=question_text[:100],
                    error=response_envelope.get("error"),
                )

                self.metrics.increment_platform_error(platform_name, "query_failed")

                return ProcessingResult(
                    question_id=str(uuid.uuid4()),
                    platform=platform_name,
                    success=False,
                    processing_time_ms=processing_time,
                    error=response_envelope.get("error"),
                )

            # Extract text response
            response_text = platform.extract_text_response(
                response_envelope["response"]
            )

            # Perform brand detection
            brand_mentions = {}
            if self.brand_detector and not self.settings.AUDIT_SKIP_BRAND_DETECTION:
                brand_detection_start = time.time()
                try:
                    detection_result = await self.brand_detector.detect_brands(
                        text=response_text, target_brands=target_brands
                    )

                    # Convert to serializable format
                    brand_mentions = detection_result.to_summary_dict()

                    brand_detection_time = (time.time() - brand_detection_start) * 1000
                    self.metrics.record_brand_detection_time(brand_detection_time)

                except Exception as e:
                    contextual_logger.warning(
                        "Brand detection failed", error=str(e), platform=platform_name
                    )
                    self.metrics.increment_brand_detection_error()

            # Persist response to database
            response_record = response_models.Response(
                id=str(uuid.uuid4()),
                audit_run_id=audit_run_id,
                question_id=None,  # We'll need to map questions properly in a future iteration
                platform=platform_name,
                response_text=response_text,
                raw_response=response_envelope["response"],
                brand_mentions=brand_mentions,
                response_metadata={
                    "processing_time_ms": processing_time,
                    "question_preview": question_text[:200],
                    "platform_metadata": response_envelope.get("metadata", {}),
                },
                processing_time_ms=processing_time,
            )

            if not self.settings.AUDIT_DRY_RUN:
                self.db.add(response_record)
                self.db.commit()
                self.metrics.increment_database_operation("insert", "responses")

            # Record metrics
            self.metrics.record_platform_query_time(platform_name, processing_time)
            self.metrics.increment_successful_queries(platform_name)

            return ProcessingResult(
                question_id=response_record.id,
                platform=platform_name,
                success=True,
                processing_time_ms=processing_time,
                tokens_used=response_envelope.get("metadata", {}).get("tokens"),
                cost_estimate=response_envelope.get("metadata", {}).get("cost"),
                response_length=len(response_text),
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)

            contextual_logger.error(
                "Error processing single question",
                platform=platform_name,
                question_preview=question_text[:100],
                error=str(e),
                processing_time_ms=processing_time,
                exc_info=True,
            )

            self.metrics.increment_platform_error(platform_name, "processing_error")

            return ProcessingResult(
                question_id=str(uuid.uuid4()),
                platform=platform_name,
                success=False,
                processing_time_ms=processing_time,
                error=str(e),
            )

    async def _process_mock_response(
        self,
        audit_run_id: str,
        question_data: Dict[str, Any],
        platform_name: str,
        target_brands: List[str],
        start_time: float,
    ) -> ProcessingResult:
        """Process a mock response for testing purposes"""

        question_text = question_data["question"]

        # Simulate processing time
        await asyncio.sleep(0.1)

        processing_time = int((time.time() - start_time) * 1000)

        # Generate mock response that mentions some target brands
        mock_response = f"Mock response for: {question_text}. "
        if target_brands:
            mock_response += (
                f"This mentions {target_brands[0]} and provides competitive insights."
            )

        # Mock brand mentions
        brand_mentions = {}
        if target_brands and not self.settings.AUDIT_SKIP_BRAND_DETECTION:
            brand_mentions = {
                target_brands[0]: {
                    "mentions": 1,
                    "sentiment_score": 0.5,
                    "confidence": 0.8,
                    "contexts": [mock_response[:100]],
                }
            }

        # Create mock response record
        response_record = response_models.Response(
            id=str(uuid.uuid4()),
            audit_run_id=audit_run_id,
            question_id=None,
            platform=platform_name,
            response_text=mock_response,
            raw_response={"mock": True, "content": mock_response},
            brand_mentions=brand_mentions,
            response_metadata={
                "processing_time_ms": processing_time,
                "question_preview": question_text[:200],
                "mock_response": True,
            },
            processing_time_ms=processing_time,
        )

        if not self.settings.AUDIT_DRY_RUN:
            self.db.add(response_record)
            self.db.commit()

        self.metrics.record_platform_query_time(platform_name, processing_time)
        self.metrics.increment_successful_queries(platform_name)

        return ProcessingResult(
            question_id=response_record.id,
            platform=platform_name,
            success=True,
            processing_time_ms=processing_time,
            tokens_used=100,  # Mock token count
            cost_estimate=0.01,  # Mock cost
            response_length=len(mock_response),
        )

    async def _update_audit_status(
        self, audit_run: audit_models.AuditRun, status: AuditStatus
    ) -> None:
        """Update audit run status with timestamp"""
        audit_run.status = status.value
        audit_run.updated_at = datetime.now(timezone.utc)

        if status == AuditStatus.RUNNING:
            audit_run.started_at = datetime.now(timezone.utc)
        elif status in [AuditStatus.COMPLETED, AuditStatus.FAILED]:
            audit_run.completed_at = datetime.now(timezone.utc)

        self.db.commit()

        contextual_logger.info(f"Audit status updated to {status.value}")

    async def _update_total_questions(
        self, audit_run: audit_models.AuditRun, total: int
    ) -> None:
        """Update total questions count"""
        audit_run.total_questions = total
        self.db.commit()

        contextual_logger.info(f"Total questions set to {total}")

    async def _update_progress(
        self,
        audit_run: audit_models.AuditRun,
        processed_count: int,
        batch_progress: BatchProgress,
    ) -> None:
        """Update audit progress with detailed tracking"""
        audit_run.processed_questions = processed_count
        audit_run.progress_data = asdict(batch_progress)
        audit_run.updated_at = datetime.now(timezone.utc)

        self.db.commit()

        progress_percentage = (
            (processed_count / audit_run.total_questions * 100)
            if audit_run.total_questions
            else 0
        )

        # Update metrics
        self.metrics.update_progress(audit_run.id, progress_percentage / 100)

        contextual_logger.info(
            "Progress updated",
            processed=processed_count,
            total=audit_run.total_questions,
            progress_percentage=f"{progress_percentage:.1f}%",
            batch_info=f"{batch_progress.batch_index}/{batch_progress.total_batches}",
        )

    async def _finalize_audit_run(
        self,
        audit_run: audit_models.AuditRun,
        results: List[ProcessingResult],
        status: AuditStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """Finalize audit run with summary statistics"""

        with add_stage_context("finalization"):
            # Calculate platform statistics
            platform_stats = {}
            for result in results:
                if result.platform not in platform_stats:
                    platform_stats[result.platform] = {
                        "total_queries": 0,
                        "successful_queries": 0,
                        "failed_queries": 0,
                        "avg_processing_time_ms": 0,
                        "total_tokens": 0,
                        "total_cost": 0.0,
                        "avg_response_length": 0,
                    }

                stats = platform_stats[result.platform]
                stats["total_queries"] += 1

                if result.success:
                    stats["successful_queries"] += 1
                    if result.tokens_used:
                        stats["total_tokens"] += result.tokens_used
                    if result.cost_estimate:
                        stats["total_cost"] += result.cost_estimate
                    if result.response_length:
                        stats["avg_response_length"] += result.response_length
                else:
                    stats["failed_queries"] += 1

            # Calculate averages
            for platform, stats in platform_stats.items():
                platform_results = [
                    r for r in results if r.platform == platform and r.success
                ]
                if platform_results:
                    avg_time = sum(
                        r.processing_time_ms for r in platform_results
                    ) / len(platform_results)
                    stats["avg_processing_time_ms"] = int(avg_time)

                    if stats["successful_queries"] > 0:
                        stats["avg_response_length"] = int(
                            stats["avg_response_length"] / stats["successful_queries"]
                        )

            # Update audit run
            audit_run.status = status.value
            audit_run.completed_at = datetime.now(timezone.utc)
            audit_run.platform_stats = platform_stats
            audit_run.processed_questions = len([r for r in results if r.success])

            if error_message:
                audit_run.error_log = error_message

            self.db.commit()

            # Clear progress metrics
            self.metrics.clear_progress(audit_run.id)

            contextual_logger.info(
                "Audit run finalized",
                status=status.value,
                platform_stats=platform_stats,
                error=error_message,
            )

    # === Additional helper methods ===

    async def get_audit_progress(self, audit_run_id: str) -> Dict[str, Any]:
        """Get current progress of an audit run"""
        audit_run = (
            self.db.query(audit_models.AuditRun)
            .filter(audit_models.AuditRun.id == audit_run_id)
            .first()
        )

        if not audit_run:
            raise ValueError(f"Audit run {audit_run_id} not found")

        progress_percentage = 0
        if audit_run.total_questions and audit_run.total_questions > 0:
            progress_percentage = (
                audit_run.processed_questions / audit_run.total_questions
            ) * 100

        return {
            "audit_run_id": audit_run_id,
            "status": audit_run.status,
            "progress_percentage": round(progress_percentage, 1),
            "processed_questions": audit_run.processed_questions,
            "total_questions": audit_run.total_questions,
            "started_at": audit_run.started_at,
            "estimated_completion": self._estimate_completion_time(audit_run),
            "platform_stats": audit_run.platform_stats,
            "progress_data": audit_run.progress_data,
        }

    def _estimate_completion_time(
        self, audit_run: audit_models.AuditRun
    ) -> Optional[datetime]:
        """Estimate completion time based on current progress"""
        if (
            not audit_run.started_at
            or not audit_run.total_questions
            or audit_run.processed_questions == 0
        ):
            return None

        elapsed = datetime.now(timezone.utc) - audit_run.started_at
        progress_ratio = audit_run.processed_questions / audit_run.total_questions

        if progress_ratio > 0:
            estimated_total_time = elapsed / progress_ratio
            return audit_run.started_at + estimated_total_time

        return None

    async def cancel_audit(self, audit_run_id: str) -> bool:
        """Cancel a running audit run"""
        audit_run = (
            self.db.query(audit_models.AuditRun)
            .filter(audit_models.AuditRun.id == audit_run_id)
            .first()
        )

        if not audit_run:
            return False

        if audit_run.status not in [
            AuditStatus.PENDING.value,
            AuditStatus.RUNNING.value,
        ]:
            return False

        await self._update_audit_status(audit_run, AuditStatus.CANCELLED)
        self.metrics.increment_audit_cancelled()

        contextual_logger.info(f"Audit run {audit_run_id} cancelled")
        return True

    def _validate_audit_config(self, config: Dict[str, Any]) -> bool:
        """Validate audit configuration"""
        if not config.get("platforms"):
            raise AuditConfigurationError(
                "No platforms specified in audit configuration"
            )

        # Additional validation can be added here
        return True
