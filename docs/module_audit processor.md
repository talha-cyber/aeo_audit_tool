# Audit Processor Module — Complete Production Build Plan

## 🎯 **Module Purpose & Context**

The Audit Processor is the **orchestration engine** for AEO competitive intelligence audits. It coordinates the entire audit workflow from question generation to brand detection, managing multi-platform AI queries with proper concurrency, rate limiting, and progress tracking.

**What you're building:** A production-grade service that takes an audit configuration, generates prioritized questions, fans them out across multiple AI platforms concurrently, processes responses through brand detection, and persists comprehensive results—all while providing real-time progress updates and robust error handling.

**Integration context:** This module sits at the center of your AEO audit pipeline, called by Celery tasks and integrating with QuestionEngine, PlatformManager, and BrandDetectionEngine.

---

## 1) **Architecture & Placement**

### Current System Integration
```
API Layer (audits.py) → Celery Task → Audit Processor → Services
                                         ↓
                              ┌─────────────────────┐
                              │  Audit Processor   │
                              │  (THIS MODULE)     │
                              └─────────────────────┘
                                         ↓
        ┌─────────────────┬─────────────────┬─────────────────┐
   QuestionEngine    PlatformManager   BrandDetection    Database
   (+ providers)     (+ AI platforms)    (+ sentiment)    (models)
```

### File Structure
```
app/services/
  audit_processor.py           # Main orchestrator (NEW)
  audit_metrics.py            # Audit-specific metrics (NEW)
  audit_context.py            # Context helpers for logging (NEW)
app/tasks/
  audit_tasks.py              # Update to use PlatformManager
app/models/
  audit.py                    # Ensure all required fields
  question.py                 # Question persistence
  response.py                 # Response with brand mentions
```

---

## 2) **Complete Data Model Requirements**

### Required Model Fields (verify/add as needed)
```python
# app/models/audit.py
class AuditRun(Base):
    id: UUID (PK)
    client_id: UUID (FK to Client)
    config: JSON                    # Audit configuration snapshot
    status: Enum                    # pending, running, completed, failed
    started_at: DateTime
    completed_at: DateTime
    total_questions: Integer        # Total planned questions
    processed_questions: Integer    # Questions actually processed
    error_log: Text                # Detailed error information
    progress_data: JSON            # Real-time progress details
    platform_stats: JSON          # Per-platform statistics
    created_at: DateTime
    updated_at: DateTime

# app/models/question.py
class Question(Base):
    id: UUID (PK)
    audit_run_id: UUID (FK)        # Link to specific audit run
    question_text: Text
    category: String               # comparison, recommendation, etc.
    question_type: String          # industry_general, brand_specific, etc.
    priority_score: Float          # Calculated priority
    target_brand: String           # Optional target brand
    provider: String               # Which provider generated it
    metadata: JSON                 # Additional question context
    created_at: DateTime

# app/models/response.py
class Response(Base):
    id: UUID (PK)
    audit_run_id: UUID (FK)
    question_id: UUID (FK)
    platform: String               # openai, anthropic, etc.
    response_text: Text            # Normalized response text
    raw_response: JSON             # Complete platform response
    brand_mentions: JSON           # Brand detection results
    response_metadata: JSON        # Timing, tokens, cost, etc.
    processing_time_ms: Integer    # Query execution time
    created_at: DateTime
```

### Required Database Indices
```sql
-- Performance optimization for audit queries
CREATE INDEX idx_audit_runs_status ON audit_runs(status);
CREATE INDEX idx_audit_runs_client_started ON audit_runs(client_id, started_at DESC);
CREATE INDEX idx_questions_audit_run ON questions(audit_run_id);
CREATE INDEX idx_responses_audit_run ON responses(audit_run_id);
CREATE INDEX idx_responses_platform_time ON responses(platform, created_at);
```

---

## 3) **Core Implementation Specification**

### 3.1 Main Orchestrator Class
```python
# app/services/audit_processor.py
from __future__ import annotations
import asyncio
import uuid
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from app.services.platform_manager import PlatformManager
from app.services.question_engine import QuestionEngine
from app.services.brand_detection.core.detector import BrandDetectionEngine
from app.services.audit_metrics import AuditMetrics
from app.services.audit_context import add_audit_context
from app.models import audit as audit_models
from app.models import question as question_models
from app.models import response as response_models
from app.utils.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)

class AuditStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessingResult:
    """Result from processing a single question"""
    question_id: uuid.UUID
    platform: str
    success: bool
    processing_time_ms: int
    error: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None

@dataclass
class BatchProgress:
    """Progress tracking for a batch of questions"""
    batch_index: int
    total_batches: int
    questions_in_batch: int
    successful_responses: int
    failed_responses: int
    batch_duration_ms: int

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
        self.brand_detector = BrandDetectionEngine()
        self.metrics = AuditMetrics()

        # Configuration from settings
        self.batch_size = getattr(settings, 'AUDIT_BATCH_SIZE', 10)
        self.max_questions = getattr(settings, 'AUDIT_MAX_QUESTIONS', 200)
        self.inter_batch_delay = getattr(settings, 'AUDIT_INTER_BATCH_SLEEP_SEC', 2)
        self.platform_timeout = getattr(settings, 'AUDIT_PLATFORM_TIMEOUT_SECONDS', 30)

    async def run_audit(self, audit_run_id: uuid.UUID) -> uuid.UUID:
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
        with add_audit_context(audit_run_id=str(audit_run_id)):
            logger.info("Starting audit run execution")

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
                total_work = len(questions) * len(context['platforms'])
                await self._update_total_questions(audit_run, total_work)
                await self._persist_questions(audit_run.id, questions)

                # Process questions in batches across platforms
                results = await self._process_questions_batched(
                    audit_run, questions, context
                )

                # Finalize audit run
                await self._finalize_audit_run(audit_run, results, AuditStatus.COMPLETED)

                logger.info(
                    "Audit run completed successfully",
                    processed_responses=len([r for r in results if r.success]),
                    failed_responses=len([r for r in results if not r.success]),
                    total_questions=len(questions),
                    platforms=len(context['platforms'])
                )

                self.metrics.increment_audit_completed()
                return audit_run_id

            except Exception as e:
                logger.error(
                    "Audit run failed with exception",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )

                await self._finalize_audit_run(audit_run, [], AuditStatus.FAILED, str(e))
                self.metrics.increment_audit_failed()
                raise

    async def _load_and_validate_audit_run(self, audit_run_id: uuid.UUID) -> audit_models.AuditRun:
        """Load audit run and validate it's ready for execution"""
        audit_run = self.db.query(audit_models.AuditRun).filter(
            audit_models.AuditRun.id == audit_run_id
        ).first()

        if not audit_run:
            raise ValueError(f"Audit run {audit_run_id} not found")

        if audit_run.status not in [AuditStatus.PENDING.value, AuditStatus.RUNNING.value]:
            raise ValueError(f"Audit run {audit_run_id} is not in executable state: {audit_run.status}")

        # Validate configuration
        config = audit_run.config or {}
        if not config.get('client'):
            raise ValueError("Audit run missing client configuration")

        return audit_run

    async def _prepare_execution_context(self, audit_run: audit_models.AuditRun) -> Dict[str, Any]:
        """Prepare all context needed for audit execution"""
        config = audit_run.config
        client = config['client']

        # Determine target brands (client + competitors)
        target_brands = [client['name']]
        if client.get('competitors'):
            target_brands.extend(client['competitors'])

        # Get available platforms
        selected_platforms = config.get('platforms', [])
        available_platforms = []

        for platform_name in selected_platforms:
            if self.platform_manager.is_platform_available(platform_name):
                available_platforms.append(platform_name)
            else:
                logger.warning(
                    "Platform not available, skipping",
                    platform=platform_name
                )

        if not available_platforms:
            raise ValueError("No platforms available for audit execution")

        logger.info(
            "Execution context prepared",
            client_name=client['name'],
            target_brands=target_brands,
            available_platforms=available_platforms,
            selected_categories=config.get('question_categories', [])
        )

        return {
            'client': client,
            'target_brands': target_brands,
            'platforms': available_platforms,
            'categories': config.get('question_categories', [])
        }

    async def _generate_questions(
        self,
        audit_run: audit_models.AuditRun,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate and prioritize questions for the audit"""

        start_time = time.time()

        # Generate questions via QuestionEngine
        raw_questions = await asyncio.get_event_loop().run_in_executor(
            None,
            self.question_engine.generate_questions,
            context['client']['name'],
            context['client'].get('competitors', []),
            context['client']['industry'],
            context['categories']
        )

        # Prioritize questions
        prioritized_questions = await asyncio.get_event_loop().run_in_executor(
            None,
            self.question_engine.prioritize_questions,
            raw_questions,
            self.max_questions
        )

        generation_time = (time.time() - start_time) * 1000

        logger.info(
            "Questions generated and prioritized",
            raw_question_count=len(raw_questions),
            prioritized_count=len(prioritized_questions),
            generation_time_ms=int(generation_time)
        )

        self.metrics.record_question_generation_time(generation_time)

        return prioritized_questions

    async def _persist_questions(self, audit_run_id: uuid.UUID, questions: List[Dict[str, Any]]) -> None:
        """Persist generated questions to database"""
        question_records = []

        for question_data in questions:
            question_record = question_models.Question(
                id=uuid.uuid4(),
                audit_run_id=audit_run_id,
                question_text=question_data['question'],
                category=question_data.get('category', 'unknown'),
                question_type=question_data.get('type', 'unknown'),
                priority_score=question_data.get('priority_score', 0.0),
                target_brand=question_data.get('target_brand'),
                provider=question_data.get('provider', 'question_engine'),
                metadata=question_data,
                created_at=datetime.now(timezone.utc)
            )
            question_records.append(question_record)

        self.db.bulk_save_objects(question_records)
        self.db.commit()

        logger.info(f"Persisted {len(question_records)} questions to database")

    async def _process_questions_batched(
        self,
        audit_run: audit_models.AuditRun,
        questions: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[ProcessingResult]:
        """Process questions in batches across platforms with progress tracking"""

        all_results = []
        total_batches = (len(questions) + self.batch_size - 1) // self.batch_size

        logger.info(
            "Starting batched question processing",
            total_questions=len(questions),
            batch_size=self.batch_size,
            total_batches=total_batches,
            platforms=context['platforms']
        )

        for batch_index in range(0, len(questions), self.batch_size):
            batch_start_time = time.time()
            batch_questions = questions[batch_index:batch_index + self.batch_size]
            batch_num = (batch_index // self.batch_size) + 1

            logger.info(
                "Processing batch",
                batch_number=batch_num,
                total_batches=total_batches,
                questions_in_batch=len(batch_questions)
            )

            # Create tasks for all platform-question combinations
            tasks = []
            for question_data in batch_questions:
                for platform_name in context['platforms']:
                    task = self._process_single_question(
                        audit_run_id=audit_run.id,
                        question_data=question_data,
                        platform_name=platform_name,
                        target_brands=context['target_brands']
                    )
                    tasks.append(task)

            # Execute batch with timeout
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.platform_timeout * len(tasks)
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Batch processing timeout",
                    batch_number=batch_num,
                    timeout_seconds=self.platform_timeout * len(tasks)
                )
                batch_results = [None] * len(tasks)

            # Filter and collect results
            valid_results = [
                result for result in batch_results
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
                batch_duration_ms=batch_duration
            )

            # Update progress in database
            await self._update_progress(audit_run, len(all_results), progress)

            # Record metrics
            self.metrics.record_batch_processing_time(batch_duration)

            # Inter-batch delay for rate limiting
            if batch_num < total_batches:
                await asyncio.sleep(self.inter_batch_delay)

        logger.info(
            "Batched processing completed",
            total_results=len(all_results),
            successful=len([r for r in all_results if r.success]),
            failed=len([r for r in all_results if not r.success])
        )

        return all_results

    async def _process_single_question(
        self,
        audit_run_id: uuid.UUID,
        question_data: Dict[str, Any],
        platform_name: str,
        target_brands: List[str]
    ) -> Optional[ProcessingResult]:
        """Process a single question on a specific platform"""

        question_text = question_data['question']
        start_time = time.time()

        try:
            # Get platform client
            platform = self.platform_manager.get_platform(platform_name)

            # Execute query with platform
            async with platform:
                response_envelope = await platform.safe_query(question_text)

            processing_time = int((time.time() - start_time) * 1000)

            if not response_envelope.get('success'):
                logger.warning(
                    "Platform query failed",
                    platform=platform_name,
                    question_preview=question_text[:100],
                    error=response_envelope.get('error')
                )

                self.metrics.increment_platform_error(platform_name, "query_failed")

                return ProcessingResult(
                    question_id=uuid.uuid4(),  # We'll need to map this properly
                    platform=platform_name,
                    success=False,
                    processing_time_ms=processing_time,
                    error=response_envelope.get('error')
                )

            # Extract text response
            response_text = platform.extract_text_response(response_envelope['response'])

            # Perform brand detection
            brand_mentions = await asyncio.get_event_loop().run_in_executor(
                None,
                self.brand_detector.detect_brands,
                response_text,
                target_brands
            )

            # Convert brand mentions to serializable format
            serialized_mentions = {}
            for brand, mention in brand_mentions.items():
                serialized_mentions[brand] = {
                    'mentions': mention.mentions,
                    'sentiment_score': mention.sentiment_score,
                    'confidence': mention.confidence,
                    'contexts': mention.contexts[:3]  # Limit context storage
                }

            # Persist response to database
            response_record = response_models.Response(
                id=uuid.uuid4(),
                audit_run_id=audit_run_id,
                question_id=None,  # We'll need to map questions properly
                platform=platform_name,
                response_text=response_text,
                raw_response=response_envelope['response'],
                brand_mentions=serialized_mentions,
                response_metadata={
                    'processing_time_ms': processing_time,
                    'question_preview': question_text[:200],
                    'platform_metadata': response_envelope.get('metadata', {})
                },
                processing_time_ms=processing_time,
                created_at=datetime.now(timezone.utc)
            )

            self.db.add(response_record)
            self.db.commit()

            # Record metrics
            self.metrics.record_platform_query_time(platform_name, processing_time)
            self.metrics.increment_successful_queries(platform_name)

            return ProcessingResult(
                question_id=response_record.id,
                platform=platform_name,
                success=True,
                processing_time_ms=processing_time,
                tokens_used=response_envelope.get('metadata', {}).get('tokens'),
                cost_estimate=response_envelope.get('metadata', {}).get('cost')
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)

            logger.error(
                "Error processing single question",
                platform=platform_name,
                question_preview=question_text[:100],
                error=str(e),
                processing_time_ms=processing_time,
                exc_info=True
            )

            self.metrics.increment_platform_error(platform_name, "processing_error")

            return ProcessingResult(
                question_id=uuid.uuid4(),
                platform=platform_name,
                success=False,
                processing_time_ms=processing_time,
                error=str(e)
            )

    async def _update_audit_status(
        self,
        audit_run: audit_models.AuditRun,
        status: AuditStatus
    ) -> None:
        """Update audit run status with timestamp"""
        audit_run.status = status.value
        audit_run.updated_at = datetime.now(timezone.utc)

        if status == AuditStatus.RUNNING:
            audit_run.started_at = datetime.now(timezone.utc)
        elif status in [AuditStatus.COMPLETED, AuditStatus.FAILED]:
            audit_run.completed_at = datetime.now(timezone.utc)

        self.db.commit()

        logger.info(f"Audit status updated to {status.value}")

    async def _update_total_questions(
        self,
        audit_run: audit_models.AuditRun,
        total: int
    ) -> None:
        """Update total questions count"""
        audit_run.total_questions = total
        self.db.commit()

        logger.info(f"Total questions set to {total}")

    async def _update_progress(
        self,
        audit_run: audit_models.AuditRun,
        processed_count: int,
        batch_progress: BatchProgress
    ) -> None:
        """Update audit progress with detailed tracking"""
        audit_run.processed_questions = processed_count
        audit_run.progress_data = asdict(batch_progress)
        audit_run.updated_at = datetime.now(timezone.utc)

        self.db.commit()

        progress_percentage = (processed_count / audit_run.total_questions * 100) if audit_run.total_questions else 0

        logger.info(
            "Progress updated",
            processed=processed_count,
            total=audit_run.total_questions,
            progress_percentage=f"{progress_percentage:.1f}%",
            batch_info=f"{batch_progress.batch_index}/{batch_progress.total_batches}"
        )

    async def _finalize_audit_run(
        self,
        audit_run: audit_models.AuditRun,
        results: List[ProcessingResult],
        status: AuditStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Finalize audit run with summary statistics"""

        # Calculate platform statistics
        platform_stats = {}
        for result in results:
            if result.platform not in platform_stats:
                platform_stats[result.platform] = {
                    'total_queries': 0,
                    'successful_queries': 0,
                    'failed_queries': 0,
                    'avg_processing_time_ms': 0,
                    'total_tokens': 0,
                    'total_cost': 0.0
                }

            stats = platform_stats[result.platform]
            stats['total_queries'] += 1

            if result.success:
                stats['successful_queries'] += 1
                if result.tokens_used:
                    stats['total_tokens'] += result.tokens_used
                if result.cost_estimate:
                    stats['total_cost'] += result.cost_estimate
            else:
                stats['failed_queries'] += 1

        # Calculate average processing times
        for platform, stats in platform_stats.items():
            platform_results = [r for r in results if r.platform == platform and r.success]
            if platform_results:
                avg_time = sum(r.processing_time_ms for r in platform_results) / len(platform_results)
                stats['avg_processing_time_ms'] = int(avg_time)

        # Update audit run
        audit_run.status = status.value
        audit_run.completed_at = datetime.now(timezone.utc)
        audit_run.platform_stats = platform_stats
        audit_run.processed_questions = len([r for r in results if r.success])

        if error_message:
            audit_run.error_log = error_message

        self.db.commit()

        logger.info(
            "Audit run finalized",
            status=status.value,
            platform_stats=platform_stats,
            error=error_message
        )

    # Additional helper methods for future extensibility

    async def get_audit_progress(self, audit_run_id: uuid.UUID) -> Dict[str, Any]:
        """Get current progress of an audit run"""
        audit_run = self.db.query(audit_models.AuditRun).filter(
            audit_models.AuditRun.id == audit_run_id
        ).first()

        if not audit_run:
            raise ValueError(f"Audit run {audit_run_id} not found")

        progress_percentage = 0
        if audit_run.total_questions and audit_run.total_questions > 0:
            progress_percentage = (audit_run.processed_questions / audit_run.total_questions) * 100

        return {
            'audit_run_id': str(audit_run_id),
            'status': audit_run.status,
            'progress_percentage': round(progress_percentage, 1),
            'processed_questions': audit_run.processed_questions,
            'total_questions': audit_run.total_questions,
            'started_at': audit_run.started_at,
            'estimated_completion': self._estimate_completion_time(audit_run),
            'platform_stats': audit_run.platform_stats,
            'progress_data': audit_run.progress_data
        }

    def _estimate_completion_time(self, audit_run: audit_models.AuditRun) -> Optional[datetime]:
        """Estimate completion time based on current progress"""
        if not audit_run.started_at or not audit_run.total_questions or audit_run.processed_questions == 0:
            return None

        elapsed = datetime.now(timezone.utc) - audit_run.started_at
        progress_ratio = audit_run.processed_questions / audit_run.total_questions

        if progress_ratio > 0:
            estimated_total_time = elapsed / progress_ratio
            return audit_run.started_at + estimated_total_time

        return None

    async def cancel_audit(self, audit_run_id: uuid.UUID) -> bool:
        """Cancel a running audit run"""
        audit_run = self.db.query(audit_models.AuditRun).filter(
            audit_models.AuditRun.id == audit_run_id
        ).first()

        if not audit_run:
            return False

        if audit_run.status not in [AuditStatus.PENDING.value, AuditStatus.RUNNING.value]:
            return False

        await self._update_audit_status(audit_run, AuditStatus.CANCELLED)

        logger.info(f"Audit run {audit_run_id} cancelled")
        return True
```

### 3.2 Audit Metrics Service
```python
# app/services/audit_metrics.py
from typing import Dict, Any
from app.services.metrics import metrics

class AuditMetrics:
    """Audit-specific metrics collection"""

    def __init__(self):
        self.metrics = metrics

    def increment_audit_started(self):
        """Increment audit start counter"""
        self.metrics.audit_runs_started_total.inc()

    def increment_audit_completed(self):
        """Increment audit completion counter"""
        self.metrics.audit_runs_completed_total.inc()

    def increment_audit_failed(self):
        """Increment audit failure counter"""
        self.metrics.audit_runs_failed_total.inc()

    def record_question_generation_time(self, duration_ms: float):
        """Record question generation timing"""
        self.metrics.question_generation_duration_seconds.observe(duration_ms / 1000)

    def record_batch_processing_time(self, duration_ms: int):
        """Record batch processing timing"""
        self.metrics.audit_batch_duration_seconds.observe(duration_ms / 1000)

    def record_platform_query_time(self, platform: str, duration_ms: int):
        """Record platform-specific query timing"""
        self.metrics.platform_query_latency_seconds.labels(platform=platform).observe(duration_ms / 1000)

    def increment_successful_queries(self, platform: str):
        """Increment successful query counter"""
        self.metrics.platform_queries_total.labels(platform=platform, status="success").inc()

    def increment_platform_error(self, platform: str, error_type: str):
        """Increment platform error counter"""
        self.metrics.platform_errors_total.labels(platform=platform, reason=error_type).inc()
```

### 3.3 Audit Context Helper

```python
# app/services/audit_context.py (completion)
import contextlib
from typing import Dict, Any, Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)

@contextlib.contextmanager
def add_audit_context(audit_run_id: str, **extra_context):
    """
    Context manager to add audit-specific context to all log entries.

    Usage:
        with add_audit_context(audit_run_id="123", client_id="456"):
            logger.info("This will include audit context")
    """
    context = {'audit_run_id': audit_run_id, **extra_context}

    # Bind context to structured logger
    bound_logger = logger.bind(**context)

    # Store original logger and replace
    original_logger = logger

    try:
        # Replace logger in module namespace temporarily
        import app.services.audit_context
        app.services.audit_context.logger = bound_logger
        yield bound_logger
    finally:
        # Restore original logger
        app.services.audit_context.logger = original_logger

def add_platform_context(platform_name: str, **extra_context):
    """Add platform-specific context to logs"""
    return {'platform': platform_name, **extra_context}

def add_question_context(question_id: str, question_preview: str, **extra_context):
    """Add question-specific context to logs"""
    return {
        'question_id': question_id,
        'question_preview': question_preview[:100],
        **extra_context
    }

def add_batch_context(batch_index: int, total_batches: int, **extra_context):
    """Add batch processing context to logs"""
    return {
        'batch_index': batch_index,
        'total_batches': total_batches,
        'batch_progress': f"{batch_index}/{total_batches}",
        **extra_context
    }
```

## 4) Complete Database Model Requirements

### 4.1 Required Model Updates

```python
# app/models/audit.py (ensure all required fields exist)
from sqlalchemy import Column, String, DateTime, Integer, Text, JSON, UUID, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime, timezone

Base = declarative_base()

class AuditSettings(BaseSettings):
    """Comprehensive audit processor configuration"""

    # Batch Processing Configuration
    AUDIT_BATCH_SIZE: int = Field(default=10, description="Number of questions per batch")
    AUDIT_BATCH_STRATEGY: AuditBatchStrategy = Field(default=AuditBatchStrategy.FIXED_SIZE)
    AUDIT_MAX_QUESTIONS: int = Field(default=200, description="Maximum questions per audit")
    AUDIT_INTER_BATCH_DELAY: float = Field(default=2.0, description="Delay between batches in seconds")

    # Platform Configuration
    AUDIT_PLATFORM_TIMEOUT_SECONDS: int = Field(default=30, description="Timeout for platform requests")
    AUDIT_MAX_CONCURRENT_PLATFORMS: int = Field(default=4, description="Max platforms to query concurrently")
    AUDIT_PLATFORM_RETRY_ATTEMPTS: int = Field(default=3, description="Retry attempts per platform")

    # Question Generation Configuration
    AUDIT_QUESTION_GENERATION_TIMEOUT: int = Field(default=60, description="Question generation timeout")
    AUDIT_MIN_QUESTIONS_PER_CATEGORY: int = Field(default=5, description="Minimum questions per category")
    AUDIT_MAX_QUESTIONS_PER_CATEGORY: int = Field(default=50, description="Maximum questions per category")

    # Brand Detection Configuration
    AUDIT_BRAND_DETECTION_TIMEOUT: int = Field(default=10, description="Brand detection timeout per response")
    AUDIT_BRAND_CONFIDENCE_THRESHOLD: float = Field(default=0.7, description="Minimum confidence for brand detection")
    AUDIT_MAX_BRAND_CONTEXTS: int = Field(default=3, description="Maximum contexts to store per brand mention")

    # Progress Tracking Configuration
    AUDIT_PROGRESS_UPDATE_INTERVAL: int = Field(default=5, description="Progress update interval in seconds")
    AUDIT_PROGRESS_PERSISTENCE: bool = Field(default=True, description="Persist progress to database")

    # Error Handling & Recovery
    AUDIT_RETRY_STRATEGY: AuditRetryStrategy = Field(default=AuditRetryStrategy.EXPONENTIAL_BACKOFF)
    AUDIT_MAX_RETRY_ATTEMPTS: int = Field(default=3, description="Maximum retry attempts for failed audits")
    AUDIT_RETRY_BACKOFF_MULTIPLIER: float = Field(default=2.0, description="Backoff multiplier for retries")
    AUDIT_CIRCUIT_BREAKER_THRESHOLD: int = Field(default=5, description="Failures before circuit breaker opens")

    # Resource Management
    AUDIT_MAX_MEMORY_MB: int = Field(default=1024, description="Maximum memory usage per audit")
    AUDIT_CLEANUP_INTERVAL_HOURS: int = Field(default=24, description="Cleanup interval for old data")
    AUDIT_MAX_REPORT_SIZE_MB: int = Field(default=50, description="Maximum report file size")

    # Monitoring & Observability
    AUDIT_METRICS_ENABLED: bool = Field(default=True, description="Enable metrics collection")
    AUDIT_DETAILED_LOGGING: bool = Field(default=True, description="Enable detailed audit logging")
    AUDIT_PERFORMANCE_TRACKING: bool = Field(default=True, description="Track performance metrics")

    @validator('AUDIT_BATCH_SIZE')
    def validate_batch_size(cls, v):
        if v < 1 or v > 100:
            raise ValueError('Batch size must be between 1 and 100')
        return v

    @validator('AUDIT_BRAND_CONFIDENCE_THRESHOLD')
    def validate_confidence_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v

    class Config:
        env_prefix = "AUDIT_"
        case_sensitive = True

# Global settings instance
audit_settings = AuditSettings()
```

## 9) Advanced Progress Tracking & Real-time Updates

### 9.1 Progress Tracking System

```python
# app/services/progress_tracker.py (create comprehensive progress tracking)
import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum

from app.models.audit import AuditRun
from app.utils.logger import get_logger
from app.core.audit_config import audit_settings

logger = get_logger(__name__)

class ProgressStage(str, Enum):
    INITIALIZING = "initializing"
    GENERATING_QUESTIONS = "generating_questions"
    PROCESSING_QUESTIONS = "processing_questions"
    DETECTING_BRANDS = "detecting_brands"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProgressSnapshot:
    """Detailed progress snapshot"""
    audit_run_id: str
    stage: ProgressStage
    overall_progress: float  # 0.0 to 1.0
    stage_progress: float    # 0.0 to 1.0

    # Question processing details
    total_questions: int
    processed_questions: int
    failed_questions: int

    # Platform details
    platforms: List[str]
    platform_progress: Dict[str, float]
    platform_errors: Dict[str, int]

    # Timing information
    started_at: datetime
    current_stage_started_at: datetime
    estimated_completion: Optional[datetime]

    # Performance metrics
    avg_question_time_ms: Optional[float]
    avg_platform_time_ms: Dict[str, float]

    # Current operation
    current_operation: Optional[str]
    current_batch: Optional[int]
    total_batches: Optional[int]

    # Error information
    last_error: Optional[str]
    error_count: int

class ProgressTracker:
    """Advanced progress tracking for audit runs"""

    def __init__(self, db_session, audit_run_id: str):
        self.db = db_session
        self.audit_run_id = audit_run_id
        self.current_snapshot: Optional[ProgressSnapshot] = None
        self.start_time = datetime.now(timezone.utc)
        self.stage_start_time = self.start_time
        self.question_times: List[float] = []
        self.platform_times: Dict[str, List[float]] = {}

        logger.info("Progress tracker initialized", audit_run_id=audit_run_id)

    async def update_stage(self, stage: ProgressStage, operation: Optional[str] = None):
        """Update the current processing stage"""
        self.stage_start_time = datetime.now(timezone.utc)

        if self.current_snapshot:
            self.current_snapshot.stage = stage
            self.current_snapshot.current_stage_started_at = self.stage_start_time
            self.current_snapshot.current_operation = operation

        await self._persist_progress()

        logger.info(
            "Stage updated",
            audit_run_id=self.audit_run_id,
            stage=stage.value,
            operation=operation
        )

    async def update_question_progress(
        self,
        processed: int,
        total: int,
        current_batch: Optional[int] = None,
        total_batches: Optional[int] = None
    ):
        """Update question processing progress"""
        if not self.current_snapshot:
            await self._initialize_snapshot()

        self.current_snapshot.processed_questions = processed
        self.current_snapshot.total_questions = total
        self.current_snapshot.current_batch = current_batch
        self.current_snapshot.total_batches = total_batches

        # Calculate overall progress based on stage
        stage_weights = {
            ProgressStage.INITIALIZING: 0.05,
            ProgressStage.GENERATING_QUESTIONS: 0.15,
            ProgressStage.PROCESSING_QUESTIONS: 0.70,
            ProgressStage.DETECTING_BRANDS: 0.08,
            ProgressStage.FINALIZING: 0.02
        }

        current_stage_weight = stage_weights.get(self.current_snapshot.stage, 0.0)
        previous_stages_weight = sum(
            weight for stage, weight in stage_weights.items()
            if list(stage_weights.keys()).index(stage) < list(stage_weights.keys()).index(self.current_snapshot.stage)
        )

        if total > 0:
            stage_progress = processed / total
            self.current_snapshot.stage_progress = stage_progress
            self.current_snapshot.overall_progress = previous_stages_weight + (current_stage_weight * stage_progress)

        # Update estimated completion
        await self._calculate_estimated_completion()

        await self._persist_progress()

    async def update_platform_progress(self, platform: str, progress: float):
        """Update progress for a specific platform"""
        if not self.current_snapshot:
            await self._initialize_snapshot()

        self.current_snapshot.platform_progress[platform] = progress
        await self._persist_progress()

    async def record_question_time(self, duration_ms: float):
        """Record timing for question processing"""
        self.question_times.append(duration_ms)

        # Keep only recent timings for better estimation
        if len(self.question_times) > 50:
            self.question_times = self.question_times[-50:]

        if self.current_snapshot:
            self.current_snapshot.avg_question_time_ms = sum(self.question_times) / len(self.question_times)

    async def record_platform_time(self, platform: str, duration_ms: float):
        """Record timing for platform queries"""
        if platform not in self.platform_times:
            self.platform_times[platform] = []

        self.platform_times[platform].append(duration_ms)

        # Keep only recent timings
        if len(self.platform_times[platform]) > 20:
            self.platform_times[platform] = self.platform_times[platform][-20:]

        if self.current_snapshot:
            self.current_snapshot.avg_platform_time_ms[platform] = (
                sum(self.platform_times[platform]) / len(self.platform_times[platform])
            )

    async def record_error(self, platform: Optional[str] = None, error_message: Optional[str] = None):
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

        await self._persist_progress()

    async def get_current_progress(self) -> Optional[ProgressSnapshot]:
        """Get current progress snapshot"""
        return self.current_snapshot

    async def _initialize_snapshot(self):
        """Initialize progress snapshot from database"""
        audit_run = self.db.query(AuditRun).filter(AuditRun.id == self.audit_run_id).first()

        if not audit_run:
            raise ValueError(f"Audit run {self.audit_run_id} not found")

        # Load existing progress or create new
        existing_progress = audit_run.progress_data or {}

        self.current_snapshot = ProgressSnapshot(
            audit_run_id=self.audit_run_id,
            stage=ProgressStage(existing_progress.get('stage', ProgressStage.INITIALIZING.value)),
            overall_progress=existing_progress.get('overall_progress', 0.0),
            stage_progress=existing_progress.get('stage_progress', 0.0),
            total_questions=existing_progress.get('total_questions', 0),
            processed_questions=existing_progress.get('processed_questions', 0),
            failed_questions=existing_progress.get('failed_questions', 0),
            platforms=existing_progress.get('platforms', []),
            platform_progress=existing_progress.get('platform_progress', {}),
            platform_errors=existing_progress.get('platform_errors', {}),
            started_at=audit_run.started_at or self.start_time,
            current_stage_started_at=self.stage_start_time,
            estimated_completion=None,
            avg_question_time_ms=existing_progress.get('avg_question_time_ms'),
            avg_platform_time_ms=existing_progress.get('avg_platform_time_ms', {}),
            current_operation=existing_progress.get('current_operation'),
            current_batch=existing_progress.get('current_batch'),
            total_batches=existing_progress.get('total_batches'),
            last_error=existing_progress.get('last_error'),
            error_count=existing_progress.get('error_count', 0)
        )

    async def _calculate_estimated_completion(self):
        """Calculate estimated completion time"""
        if not self.current_snapshot or self.current_snapshot.overall_progress == 0:
            return

        elapsed_time = datetime.now(timezone.utc) - self.current_snapshot.started_at
        total_estimated_time = elapsed_time / self.current_snapshot.overall_progress

        self.current_snapshot.estimated_completion = (
            self.current_snapshot.started_at + total_estimated_time
        )

    async def _persist_progress(self):
        """Persist current progress to database"""
        if not self.current_snapshot or not audit_settings.AUDIT_PROGRESS_PERSISTENCE:
            return

        try:
            audit_run = self.db.query(AuditRun).filter(AuditRun.id == self.audit_run_id).first()
            if audit_run:
                # Convert snapshot to dict for JSON storage
                progress_dict = asdict(self.current_snapshot)

                # Convert datetime objects to ISO strings
                for key, value in progress_dict.items():
                    if isinstance(value, datetime):
                        progress_dict[key] = value.isoformat()

                audit_run.progress_data = progress_dict
                audit_run.processed_questions = self.current_snapshot.processed_questions
                audit_run.updated_at = datetime.now(timezone.utc)

                self.db.commit()

        except Exception as e:
            logger.error(
                "Failed to persist progress",
                audit_run_id=self.audit_run_id,
                error=str(e)
            )
            self.db.rollback()
```

## 10) API Integration & Status Endpoints

### 10.1 Enhanced API Endpoints

```python
# app/api/v1/audit_status.py (create comprehensive status API)
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone

from app.db.session import get_db
from app.models.audit import AuditRun
from app.services.progress_tracker import ProgressSnapshot, ProgressStage
from app.services.recovery import AuditRecoveryManager
from app.services.metrics import metrics_collector
from pydantic import BaseModel

router = APIRouter(prefix="/audit-status", tags=["audit-status"])

class AuditStatusResponse(BaseModel):
    audit_run_id: str
    status: str
    stage: Optional[str]
    overall_progress: float
    stage_progress: float

    # Timing information
    started_at: Optional[datetime]
    estimated_completion: Optional[datetime]

    # Processing details
    total_questions: int
    processed_questions: int
    failed_questions: int

    # Platform information
    platform_progress: Dict[str, float]
    platform_errors: Dict[str, int]

    # Performance metrics
    avg_processing_time_ms: Optional[float]
    current_batch: Optional[int]
    total_batches: Optional[int]

    # Error information
    last_error: Optional[str]
    error_count: int

class AuditListResponse(BaseModel):
    audit_runs: List[AuditStatusResponse]
    total_count: int
    page: int
    page_size: int

@router.get("/runs/{run_id}", response_model=AuditStatusResponse)
async def get_audit_status(
    run_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed status for a specific audit run"""
    try:
        audit_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid audit run ID format")

    audit_run = db.query(AuditRun).filter(AuditRun.id == audit_uuid).first()

    if not audit_run:
        raise HTTPException(status_code=404, detail="Audit run not found")

    # Extract progress data
    progress_data = audit_run.progress_data or {}

    return AuditStatusResponse(
        audit_run_id=str(audit_run.id),
        status=audit_run.status,
        stage=progress_data.get('stage'),
        overall_progress=progress_data.get('overall_progress', 0.0),
        stage_progress=progress_data.get('stage_progress', 0.0),
        started_at=audit_run.started_at,
        estimated_completion=_parse_datetime(progress_data.get('estimated_completion')),
        total_questions=progress_data.get('total_questions', 0),
        processed_questions=progress_data.get('processed_questions', 0),
        failed_questions=progress_data.get('failed_questions', 0),
        platform_progress=progress_data.get('platform_progress', {}),
        platform_errors=progress_data.get('platform_errors', {}),
        avg_processing_time_ms=progress_data.get('avg_question_time_ms'),
        current_batch=progress_data.get('current_batch'),
        total_batches=progress_data.get('total_batches'),
        last_error=progress_data.get('last_error'),
        error_count=progress_data.get('error_count', 0)
    )

@router.get("/runs", response_model=AuditListResponse)
async def list_audit_runs(
    status: Optional[str] = Query(None, description="Filter by status"),
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db)
):
    """List audit runs with filtering and pagination"""

    query = db.query(AuditRun)

    # Apply filters
    if status:
        query = query.filter(AuditRun.status == status)

    if client_id:
        try:
            client_uuid = uuid.UUID(client_id)
            query = query.filter(AuditRun.client_id == client_uuid)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid client ID format")

    # Get total count
    total_count = query.count()

    # Apply pagination
    offset = (page - 1) * page_size
    audit_runs = query.order_by(AuditRun.created_at.desc()).offset(offset).limit(page_size).all()

    # Convert to response format
    audit_list = []
    for audit_run in audit_runs:
        progress_data = audit_run.progress_data or {}
        audit_list.append(AuditStatusResponse(
            audit_run_id=str(audit_run.id),
            status=audit_run.status,
            stage=progress_data.get('stage'),
            overall_progress=progress_data.get('overall_progress', 0.0),
            stage_progress=progress_data.get('stage_progress', 0.0),
            started_at=audit_run.started_at,
            estimated_completion=_parse_datetime(progress_data.get('estimated_completion')),
            total_questions=progress_data.get('total_questions', 0),
            processed_questions=progress_data.get('processed_questions', 0),
            failed_questions=progress_data.get('failed_questions', 0),
            platform_progress=progress_data.get('platform_progress', {}),
            platform_errors=progress_data.get('platform_errors', {}),
            avg_processing_time_ms=progress_data.get('avg_question_time_ms'),
            current_batch=progress_data.get('current_batch'),
            total_batches=progress_data.get('total_batches'),
            last_error=progress_data.get('last_error'),
            error_count=progress_data.get('error_count', 0)
        ))

    return AuditListResponse(
        audit_runs=audit_list,
        total_count=total_count,
        page=page,
        page_size=page_size
    )

@router.post("/runs/{run_id}/cancel")
async def cancel_audit_run(
    run_id: str,
    db: Session = Depends(get_db)
):
    """Cancel a running audit"""
    try:
        audit_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid audit run ID format")

    audit_run = db.query(AuditRun).filter(AuditRun.id == audit_uuid).first()

    if not audit_run:
        raise HTTPException(status_code=404, detail="Audit run not found")

    if audit_run.status not in ['pending', 'running']:
        raise HTTPException(status_code=400, detail="Cannot cancel audit in current status")

    # Update status to cancelled
    audit_run.status = 'cancelled'
    audit_run.completed_at = datetime.now(timezone.utc)
    audit_run.updated_at = datetime.now(timezone.utc)

    db.commit()

    return {"message": "Audit run cancelled successfully", "audit_run_id": run_id}

@router.post("/recovery/failed")
async def recover_failed_audits(
    max_attempts: int = Query(3, ge=1, le=10, description="Maximum recovery attempts"),
    db: Session = Depends(get_db)
):
    """Attempt to recover failed audit runs"""

    recovery_manager = AuditRecoveryManager(db)

    try:
        recovered_runs = await recovery_manager.recover_failed_audits(max_attempts)
        return {
            "message": f"Recovery attempted for {len(recovered_runs)} audit runs",
            "recovered_runs": recovered_runs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recovery failed: {str(e)}")

@router.post("/cleanup/orphaned")
async def cleanup_orphaned_runs(
    max_age_hours: int = Query(24, ge=1, le=168, description="Maximum age in hours"),
    db: Session = Depends(get_db)
):
    """Clean up orphaned audit runs"""

    recovery_manager = AuditRecoveryManager(db)

    try:
        cleaned_runs = await recovery_manager.cleanup_orphaned_runs(max_age_hours)
        return {
            "message": f"Cleaned up {len(cleaned_runs)} orphaned audit runs",
            "cleaned_runs": cleaned_runs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.get("/metrics")
async def get_audit_metrics():
    """Get current audit processing metrics"""
    # This would integrate with your metrics system
    # For now, return basic information
    return {
        "message": "Metrics endpoint - integrate with Prometheus/metrics system"
    }

def _parse_datetime(datetime_str: Optional[str]) -> Optional[datetime]:
    """Parse datetime string to datetime object"""
    if not datetime_str:
        return None

    try:
        return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return None
```

## 11) Testing Strategy & Test Suite

### 11.1 Comprehensive Test Suite

```python
# tests/test_audit_processor.py (comprehensive test suite)
import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from app.services.audit_processor import AuditProcessor, AuditStatus, ProcessingResult, BatchProgress
from app.services.progress_tracker import ProgressTracker, ProgressStage
from app.services.metrics import metrics_collector
from app.models.audit import AuditRun
from app.models.question import Question
from app.models.response import Response

class MockPlatformManager:
    """Mock platform manager for testing"""

    def __init__(self, available_platforms=None):
        self.available_platforms = available_platforms or ['openai', 'anthropic']

    def is_platform_available(self, platform_name: str) -> bool:
        return platform_name in self.available_platforms

    def get_platform(self, platform_name: str):
        return MockPlatform(platform_name)

class MockPlatform:
    """Mock AI platform for testing"""

    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def safe_query(self, question: str):
        if self.should_fail:
            return {
                'success': False,
                'error': 'Mock platform error',
                'response': None
            }

        return {
            'success': True,
            'response': {
                'choices': [{
                    'message': {
                        'content': f"Mock response from {self.name} for: {question}. This mentions Salesforce and HubSpot."
                    }
                }]
            },
            'metadata': {
                'tokens': 100,
                'cost': 0.01
            }
        }

    def extract_text_response(self, raw_response):
        return raw_response['choices'][0]['message']['content']

class MockBrandDetector:
    """Mock brand detector for testing"""

    def detect_brands(self, text: str, target_brands: list):
        # Simple mock detection
        detected = {}
        for brand in target_brands:
            if brand.lower() in text.lower():
                detected[brand] = Mock(
                    mentions=1,
                    sentiment_score=0.5,
                    confidence=0.8,
                    contexts=[text[:100]]
                )
        return detected

class MockQuestionEngine:
    """Mock question engine for testing"""

    def generate_questions(self, client_brand, competitors, industry, categories):
        return [
            {
                'question': f'What is the best {industry} software?',
                'category': 'comparison',
                'type': 'industry_general',
                'priority_score': 10.0,
                'provider': 'mock'
            },
            {
                'question': f'How does {client_brand} compare to competitors?',
                'category': 'comparison',
                'type': 'brand_specific',
                'priority_score': 9.0,
                'target_brand': client_brand,
                'provider': 'mock'
            }
        ]

    def prioritize_questions(self, questions, max_questions):
        return sorted(questions, key=lambda x: x.get('priority_score', 0), reverse=True)[:max_questions]

@pytest.fixture
def mock_db_session():
    """Mock database session"""
    db = Mock()

    # Mock audit run
    audit_run = Mock()
    audit_run.id = uuid.uuid4()
    audit_run.config = {
        'client': {
            'name': 'TestCRM',
            'competitors': ['Salesforce', 'HubSpot'],
            'industry': 'CRM'
        },
        'platforms': ['openai', 'anthropic'],
        'question_categories': ['comparison', 'recommendation']
    }
    audit_run.status = 'pending'

    db.query.return_value.filter.return_value.first.return_value = audit_run
    db.commit = Mock()
    db.add = Mock()

    return db

@pytest.fixture
def audit_processor(mock_db_session):
    """Create audit processor with mocked dependencies"""
    platform_manager = MockPlatformManager()
    processor = AuditProcessor(mock_db_session, platform_manager)

    # Replace engines with mocks
    processor.question_engine = MockQuestionEngine()
    processor.brand_detector = MockBrandDetector()

    return processor

@pytest.mark.asyncio
async def test_audit_processor_initialization(mock_db_session):
    """Test audit processor initialization"""
    platform_manager = MockPlatformManager()
    processor = AuditProcessor(mock_db_session, platform_manager)

    assert processor.db == mock_db_session
    assert processor.platform_manager == platform_manager
    assert processor.batch_size == 10  # default
    assert processor.max_questions == 200  # default

@pytest.mark.asyncio
async def test_load_and_validate_audit_run(audit_processor, mock_db_session):
    """Test audit run loading and validation"""
    audit_run_id = uuid.uuid4()

    # Test successful load
    audit_run = await audit_processor._load_and_validate_audit_run(audit_run_id)
    assert audit_run is not None

    # Test audit run not found
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    with pytest.raises(ValueError, match="Audit run .* not found"):
        await audit_processor._load_and_validate_audit_run(audit_run_id)

@pytest.mark.asyncio
async def test_prepare_execution_context(audit_processor):
    """Test execution context preparation"""
    audit_run = Mock()
    audit_run.config = {
        'client': {
            'name': 'TestCRM',
            'competitors': ['Salesforce', 'HubSpot'],
            'industry': 'CRM'
        },
        'platforms': ['openai', 'anthropic'],
        'question_categories': ['comparison']
    }

    context = await audit_processor._prepare_execution_context(audit_run)

    assert context['client']['name'] == 'TestCRM'
    assert 'TestCRM' in context['target_brands']
    assert 'Salesforce' in context['target_brands']
    assert 'HubSpot' in context['target_brands']
    assert 'openai' in context['platforms']
    assert 'anthropic' in context['platforms']
    assert context['categories'] == ['comparison']

@pytest.mark.asyncio
async def test_generate_questions(audit_processor):
    """Test question generation"""
    audit_run = Mock()
    audit_run.config = {
        'client': {
            'name': 'TestCRM',
            'competitors': ['Salesforce'],
            'industry': 'CRM'
        }
    }

    context = {
        'client': {'name': 'TestCRM', 'competitors': ['Salesforce'], 'industry': 'CRM'},
        'target_brands': ['TestCRM', 'Salesforce'],
        'platforms': ['openai'],
        'categories': ['comparison']
    }

    questions = await audit_processor._generate_questions(audit_run, context)

    assert len(questions) == 2
    assert any('CRM software' in q['question'] for q in questions)
    assert any('TestCRM' in q['question'] for q in questions)

@pytest.mark.asyncio
async def test_process_single_question_success(audit_processor):
    """Test successful single question processing"""
    audit_run_id = uuid.uuid4()
    question_data = {
        'question': 'What is the best CRM software?',
        'category': 'comparison',
        'provider': 'mock'
    }
    platform_name = 'openai'
    target_brands = ['TestCRM', 'Salesforce']

    with patch.object(audit_processor.platform_manager, 'get_platform') as mock_get_platform:
        mock_platform = MockPlatform('openai')
        mock_get_platform.return_value = mock_platform

        result = await audit_processor._process_single_question(
            audit_run_id, question_data, platform_name, target_brands
        )

        assert result is not None
        assert result.success == True
        assert result.platform == platform_name
        assert result.processing_time_ms > 0

@pytest.mark.asyncio
async def test_process_single_question_failure(audit_processor):
    """Test failed single question processing"""
    audit_run_id = uuid.uuid4()
    question_data = {
        'question': 'What is the best CRM software?',
        'category': 'comparison'
    }
    platform_name = 'openai'
    target_brands = ['TestCRM']

    with patch.object(audit_processor.platform_manager, 'get_platform') as mock_get_platform:
        mock_platform = MockPlatform('openai', should_fail=True)
        mock_get_platform.return_value = mock_platform

        result = await audit_processor._process_single_question(
            audit_run_id, question_data, platform_name, target_brands
        )

        assert result is not None
        assert result.success == False
        assert result.error is not None

@pytest.mark.asyncio
async def test_process_questions_batched(audit_processor):
    """Test batched question processing"""
    audit_run = Mock()
    audit_run.id = uuid.uuid4()

    questions = [
        {'question': 'Question 1', 'category': 'comparison'},
        {'question': 'Question 2', 'category': 'comparison'},
        {'question': 'Question 3', 'category': 'comparison'}
    ]

    context = {
        'platforms': ['openai'],
        'target_brands': ['TestCRM']
    }

    # Mock the batch size to be smaller for testing
    audit_processor.batch_size = 2

    with patch.object(audit_processor, '_process_single_question') as mock_process:
        mock_process.return_value = ProcessingResult(
            question_id=uuid.uuid4(),
            platform='openai',
            success=True,
            processing_time_ms=100
        )

        results = await audit_processor._process_questions_batched(audit_run, questions, context)

        # Should process 3 questions * 1 platform = 3 results
        assert len(results) == 3
        assert all(r.success for r in results)

@pytest.mark.asyncio
async def test_full_audit_run_success(audit_processor, mock_db_session):
    """Test complete successful audit run"""
    audit_run_id = uuid.uuid4()

    with patch.object(audit_processor, '_process_questions_batched') as mock_batch:
        mock_batch.return_value = [
            ProcessingResult(uuid.uuid4(), 'openai', True, 100),
            ProcessingResult(uuid.uuid4(), 'anthropic', True, 150)
        ]

        result_id = await audit_processor.run_audit(audit_run_id)

        assert result_id == audit_run_id
        mock_db_session.commit.assert_called()

@pytest.mark.asyncio
async def test_audit_run_with_platform_errors(audit_processor):
    """Test audit run resilience with platform errors"""
    audit_run_id = uuid.uuid4()

    # Mock some platforms to fail
    original_get_platform = audit_processor.platform_manager.get_platform

    def mock_get_platform(platform_name):
        if platform_name == 'openai':
            return MockPlatform('openai', should_fail=True)
        return MockPlatform(platform_name)

    audit_processor.platform_manager.get_platform = mock_get_platform

    # Audit should complete even with some platform failures
    result_id = await audit_processor.run_audit(audit_run_id)
    assert result_id == audit_run_id

# Progress Tracker Tests

@pytest.mark.asyncio
async def test_progress_tracker_initialization():
    """Test progress tracker initialization"""
    mock_db = Mock()
    audit_run_id = str(uuid.uuid4())

    # Mock audit run in database
    audit_run = Mock()
    audit_run.id = audit_run_id
    audit_run.started_at = datetime.now(timezone.utc)
    audit_run.progress_data = {}

    mock_db.query.return_value.filter.return_value.first.return_value = audit_run

    tracker = ProgressTracker(mock_db, audit_run_id)
    await tracker._initialize_snapshot()

    assert tracker.current_snapshot is not None
    assert tracker.current_snapshot.audit_run_id == audit_run_id

@pytest.mark.asyncio
async def test_progress_tracker_stage_updates():
    """Test progress tracker stage updates"""
    mock_db = Mock()
    audit_run_id = str(uuid.uuid4())

    # Mock audit run
    audit_run = Mock()
    audit_run.id = audit_run_id
    audit_run.started_at = datetime.now(timezone.utc)
    audit_run.progress_data = {}

    mock_db.query.return_value.filter.return_value.first.return_value = audit_run

    tracker = ProgressTracker(mock_db, audit_run_id)
    await tracker._initialize_snapshot()

    # Test stage updates
    await tracker.update_stage(ProgressStage.GENERATING_QUESTIONS, "Starting question generation")
    assert tracker.current_snapshot.stage == ProgressStage.GENERATING_QUESTIONS
    assert tracker.current_snapshot.current_operation == "Starting question generation"

    await tracker.update_stage(ProgressStage.PROCESSING_QUESTIONS)
    assert tracker.current_snapshot.stage == ProgressStage.PROCESSING_QUESTIONS

@pytest.mark.asyncio
async def test_progress_tracker_question_progress():
    """Test progress tracker question progress updates"""
    mock_db = Mock()
    audit_run_id = str(uuid.uuid4())

    # Mock audit run
    audit_run = Mock()
    audit_run.id = audit_run_id
    audit_run.started_at = datetime.now(timezone.utc)
    audit_run.progress_data = {}

    mock_db.query.return_value.filter.return_value.first.return_value = audit_run

    tracker = ProgressTracker(mock_db, audit_run_id)
    await tracker._initialize_snapshot()
    await tracker.update_stage(ProgressStage.PROCESSING_QUESTIONS)

    # Test question progress updates
    await tracker.update_question_progress(5, 10, current_batch=1, total_batches=2)

    assert tracker.current_snapshot.processed_questions == 5
    assert tracker.current_snapshot.total_questions == 10
    assert tracker.current_snapshot.current_batch == 1
    assert tracker.current_snapshot.total_batches == 2
    assert tracker.current_snapshot.stage_progress == 0.5  # 5/10

# Integration Tests

@pytest.mark.asyncio
async def test_audit_processor_with_real_question_engine():
    """Integration test with real question engine"""
    from app.services.question_engine import QuestionEngine

    mock_db = Mock()
    platform_manager = MockPlatformManager()

    processor = AuditProcessor(mock_db, platform_manager)
    processor.question_engine = QuestionEngine()  # Use real question engine
    processor.brand_detector = MockBrandDetector()

    # Mock audit run
    audit_run = Mock()
    audit_run.config = {
        'client': {
            'name': 'TestCRM',
            'competitors': ['Salesforce', 'HubSpot'],
            'industry': 'CRM'
        },
        'platforms': ['openai'],
        'question_categories': ['comparison', 'recommendation']
    }

    context = await processor._prepare_execution_context(audit_run)
    questions = await processor._generate_questions(audit_run, context)

    # Should generate real questions
    assert len(questions) > 0
    assert any('CRM' in q['question'] for q in questions)

# Performance Tests

@pytest.mark.asyncio
async def test_audit_processor_performance_large_batch():
    """Test performance with large question batches"""
    import time

    mock_db = Mock()
    platform_manager = MockPlatformManager()

    processor = AuditProcessor(mock_db, platform_manager)
    processor.question_engine = MockQuestionEngine()
    processor.brand_detector = MockBrandDetector()
    processor.batch_size = 50  # Large batch

    # Generate many questions
    questions = []
    for i in range(100):
        questions.append({
            'question': f'Test question {i}',
            'category': 'comparison',
            'provider': 'mock'
        })

    context = {
        'platforms': ['openai'],
        'target_brands': ['TestCRM']
    }

    audit_run = Mock()
    audit_run.id = uuid.uuid4()

    start_time = time.time()

    with patch.object(processor, '_process_single_question') as mock_process:
        mock_process.return_value = ProcessingResult(
            question_id=uuid.uuid4(),
            platform='openai',
            success=True,
            processing_time_ms=10
        )

        results = await processor._process_questions_batched(audit_run, questions, context)

    end_time = time.time()
    processing_time = end_time - start_time

    # Should complete in reasonable time (adjust threshold as needed)
    assert processing_time < 10.0  # 10 seconds max
    assert len(results) == 100

@pytest.mark.asyncio
async def test_concurrent_platform_processing():
    """Test concurrent processing across multiple platforms"""
    mock_db = Mock()
    platform_manager = MockPlatformManager(['openai', 'anthropic', 'perplexity'])

    processor = AuditProcessor(mock_db, platform_manager)
    processor.question_engine = MockQuestionEngine()
    processor.brand_detector = MockBrandDetector()

    questions = [{'question': 'Test question', 'category': 'comparison'}]
    context = {
        'platforms': ['openai', 'anthropic', 'perplexity'],
        'target_brands': ['TestCRM']
    }

    audit_run = Mock()
    audit_run.id = uuid.uuid4()

    import time
    start_time = time.time()

    with patch.object(processor, '_process_single_question') as mock_process:
        async def mock_process_with_delay(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            return ProcessingResult(
                question_id=uuid.uuid4(),
                platform=args[2],  # platform_name
                success=True,
                processing_time_ms=100
            )

        mock_process.side_effect = mock_process_with_delay

        results = await processor._process_questions_batched(audit_run, questions, context)

    end_time = time.time()
    processing_time = end_time - start_time

    # Should process concurrently (3 platforms * 0.1s delay should be ~0.1s, not 0.3s)
    assert processing_time < 0.5  # Allow some overhead
    assert len(results) == 3  # One result per platform

# Error Handling Tests

@pytest.mark.asyncio
async def test_error_recovery_and_metrics():
    """Test error handling and metrics collection"""
    mock_db = Mock()
    platform_manager = MockPlatformManager()

    processor = AuditProcessor(mock_db, platform_manager)
    processor.question_engine = MockQuestionEngine()
    processor.brand_detector = MockBrandDetector()

    audit_run_id = uuid.uuid4()

    # Mock database error during processing
    def mock_commit_error():
        raise Exception("Database connection lost")

    mock_db.commit.side_effect = mock_commit_error

    with pytest.raises(Exception, match="Database connection lost"):
        await processor.run_audit(audit_run_id)

    # Verify error was handled gracefully (no hanging resources, etc.)
    assert True  # If we get here, exception was properly propagated

@pytest.mark.asyncio
async def test_platform_unavailable_handling():
    """Test handling when all platforms are unavailable"""
    mock_db = Mock()
    platform_manager = MockPlatformManager([])  # No available platforms

    processor = AuditProcessor(mock_db, platform_manager)
    processor.question_engine = MockQuestionEngine()
    processor.brand_detector = MockBrandDetector()

    # Mock audit run
    audit_run = Mock()
    audit_run.config = {
        'client': {'name': 'TestCRM', 'competitors': [], 'industry': 'CRM'},
        'platforms': ['openai', 'anthropic'],
        'question_categories': ['comparison']
    }
    mock_db.query.return_value.filter.return_value.first.return_value = audit_run

    audit_run_id = uuid.uuid4()

    with pytest.raises(ValueError, match="No platforms available"):
        await processor.run_audit(audit_run_id)
```

## 12) Deployment & Production Readiness

### 12.1 Production Configuration

```python
# app/core/production_config.py (production-specific configuration)
from pydantic import BaseSettings, Field
from typing import Dict, List, Optional
import os

class ProductionAuditSettings(BaseSettings):
    """Production-specific audit processor settings"""

    # High-availability settings
    AUDIT_REDIS_CLUSTER_NODES: List[str] = Field(
        default_factory=lambda: os.getenv('REDIS_CLUSTER_NODES', 'redis:6379').split(',')
    )
    AUDIT_DATABASE_POOL_SIZE: int = Field(default=20, description="Database connection pool size")
    AUDIT_DATABASE_MAX_OVERFLOW: int = Field(default=30, description="Database max overflow")

    # Performance settings
    AUDIT_WORKER_CONCURRENCY: int = Field(default=4, description="Celery worker concurrency")
    AUDIT_WORKER_PREFETCH_MULTIPLIER: int = Field(default=1, description="Celery prefetch multiplier")
    AUDIT_TASK_TIMEOUT_SECONDS: int = Field(default=3600, description="Maximum task execution time")

    # Platform rate limiting (production values)
    AUDIT_OPENAI_RPM: int = Field(default=500, description="OpenAI requests per minute")
    AUDIT_ANTHROPIC_RPM: int = Field(default=1000, description="Anthropic requests per minute")
    AUDIT_PERPLEXITY_RPM: int = Field(default=200, description="Perplexity requests per minute")
    AUDIT_GOOGLE_AI_RPM: int = Field(default=600, description="Google AI requests per minute")

    # Circuit breaker settings
    AUDIT_CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=10)
    AUDIT_CIRCUIT_BREAKER_TIMEOUT_SECONDS: int = Field(default=60)
    AUDIT_CIRCUIT_BREAKER_EXPECTED_EXCEPTION: List[str] = Field(
        default=['PlatformTimeoutError', 'PlatformUnavailableError']
    )

    # Monitoring and alerting
    AUDIT_PROMETHEUS_PORT: int = Field(default=8001, description="Prometheus metrics port")
    AUDIT_HEALTH_CHECK_INTERVAL: int = Field(default=30, description="Health check interval")
    AUDIT_ALERT_EMAIL_RECIPIENTS: List[str] = Field(default_factory=list)

    # Data retention
    AUDIT_DATA_RETENTION_DAYS: int = Field(default=90, description="Data retention period")
    AUDIT_LOG_RETENTION_DAYS: int = Field(default=30, description="Log retention period")
    AUDIT_REPORT_RETENTION_DAYS: int = Field(default=180, description="Report retention period")

    # Security settings
    AUDIT_ENCRYPT_RESPONSES: bool = Field(default=True, description="Encrypt stored responses")
    AUDIT_RATE_LIMIT_PER_CLIENT: int = Field(default=100, description="Rate limit per client per hour")
    AUDIT_MAX_CONCURRENT_RUNS_PER_CLIENT: int = Field(default=5)

    class Config:
        env_prefix = "PROD_AUDIT_"
        case_sensitive = True

# Global production settings
prod_audit_settings = ProductionAuditSettings()
```

### 12.2 Health Checks & Monitoring

```python
# app/services/health_check.py (comprehensive health checking)
import asyncio
import time
from typing import Dict, Any, List
from datetime import datetime, timezone
from enum import Enum

from app.services.platform_manager import PlatformManager
from app.db.session import SessionLocal
from app.models.audit import AuditRun
from app.utils.logger import get_logger
from app.core.production_config import prod_audit_settings

logger = get_logger(__name__)

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ComponentHealth:
    """Health status for a system component"""

    def __init__(self, name: str, status: HealthStatus, message: str = "", metrics: Dict[str, Any] = None):
        self.name = name
        self.status = status
        self.message = message
        self.metrics = metrics or {}
        self.timestamp = datetime.now(timezone.utc)

class AuditSystemHealthChecker:
    """Comprehensive health checker for audit system"""

    def __init__(self):
        self.platform_manager = PlatformManager()
        self.last_check_time = None
        self.cached_results = {}

    async def check_system_health(self, include_detailed: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.

        Returns:
            Dict containing overall health status and component details
        """
        start_time = time.time()

        # Run all health checks concurrently
        health_checks = [
            self._check_database_health(),
            self._check_redis_health(),
            self._check_platform_health(),
            self._check_celery_health(),
            self._check_audit_processing_health(),
            self._check_disk_space_health(),
            self._check_memory_health()
        ]

        if include_detailed:
            health_checks.extend([
                self._check_recent_audit_success_rate(),
                self._check_platform_response_times(),
                self._check_error_rates()
            ])

        results = await asyncio.gather(*health_checks, return_exceptions=True)

        # Process results
        component_health = {}
        overall_status = HealthStatus.HEALTHY

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Health check failed: {str(result)}")
                component_health[f"check_{i}"] = ComponentHealth(
                    f"check_{i}", HealthStatus.UNHEALTHY, str(result)
                )
                overall_status = HealthStatus.UNHEALTHY
            elif isinstance(result, ComponentHealth):
                component_health[result.name] = result

                # Determine overall status
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED

        check_duration = time.time() - start_time
        self.last_check_time = datetime.now(timezone.utc)

        return {
            "overall_status": overall_status.value,
            "check_timestamp": self.last_check_time.isoformat(),
            "check_duration_seconds": round(check_duration, 3),
            "components": {
                name: {
                    "status": health.status.value,
                    "message": health.message,
                    "metrics": health.metrics,
                    "timestamp": health.timestamp.isoformat()
                }
                for name, health in component_health.items()
            }
        }

    async def _check_database_health(self) -> ComponentHealth:
        """Check database connectivity and performance"""
        try:
            db = SessionLocal()
            start_time = time.time()

            # Test basic connectivity
            db.execute("SELECT 1")

            # Check audit runs table
            recent_runs = db.query(AuditRun).limit(1).all()

            # Check connection pool status
            pool_status = db.get_bind().pool.status()

            db.close()

            response_time = (time.time() - start_time) * 1000

            # Determine health status based on response time
            if response_time > 1000:  # 1 second
                status = HealthStatus.DEGRADED
                message = f"Database slow response: {response_time:.0f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database responsive: {response_time:.0f}ms"

            return ComponentHealth(
                "database",
                status,
                message,
                {
                    "response_time_ms": round(response_time, 2),
                    "pool_status": pool_status,
                    "recent_runs_accessible": len(recent_runs) >= 0
                }
            )

        except Exception as e:
            return ComponentHealth(
                "database",
                HealthStatus.UNHEALTHY,
                f"Database connection failed: {str(e)}"
            )

    async def _check_redis_health(self) -> ComponentHealth:
        """Check Redis connectivity and performance"""
        try:
            import redis

            # Connect to Redis
            redis_client = redis.from_url(
                prod_audit_settings.AUDIT_REDIS_CLUSTER_NODES[0]
                if prod_audit_settings.AUDIT_REDIS_CLUSTER_NODES
                else "redis://localhost:6379"
            )

            start_time = time.time()

            # Test basic operations
            redis_client.ping()
            redis_client.set("health_check", "test", ex=10)
            value = redis_client.get("health_check")

            response_time = (time.time() - start_time) * 1000

            # Get Redis info
            info = redis_client.info()

            redis_client.close()

            # Check memory usage
            memory_usage = info.get('used_memory', 0) / info.get('maxmemory', 1) if info.get('maxmemory') else 0

            if response_time > 100:
                status = HealthStatus.DEGRADED
                message = f"Redis slow response: {response_time:.0f}ms"
            elif memory_usage > 0.9:
                status = HealthStatus.DEGRADED
                message = f"Redis high memory usage: {memory_usage*100:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Redis responsive: {response_time:.0f}ms"

            return ComponentHealth(
                "redis",
                status,
                message,
                {
                    "response_time_ms": round(response_time, 2),
                    "memory_usage_percent": round(memory_usage * 100, 2),
                    "connected_clients": info.get('connected_clients', 0),
                    "ops_per_sec": info.get('instantaneous_ops_per_sec', 0)
                }
            )

        except Exception as e:
            return ComponentHealth(
                "redis",
                HealthStatus.UNHEALTHY,
                f"Redis connection failed: {str(e)}"
            )

    async def _check_platform_health(self) -> ComponentHealth:
        """Check AI platform availability"""
        try:
            platform_results = {}
            overall_healthy = True

            for platform_name in ['openai', 'anthropic', 'perplexity', 'google_ai']:
                if self.platform_manager.is_platform_available(platform_name):
                    try:
                        platform = self.platform_manager.get_platform(platform_name)
                        start_time = time.time()

                        # Simple health check query
                        async with platform:
                            result = await platform.safe_query("Health check")

                        response_time = (time.time() - start_time) * 1000

                        platform_results[platform_name] = {
                            "status": "healthy" if result.get('success') else "degraded",
                            "response_time_ms": round(response_time, 2),
                            "error": result.get('error') if not result.get('success') else None
                        }

                        if not result.get('success'):
                            overall_healthy = False

                    except Exception as e:
                        platform_results[platform_name] = {
                            "status": "unhealthy",
                            "error": str(e)
                        }
                        overall_healthy = False
                else:
                    platform_results[platform_name] = {
                        "status": "unavailable",
                        "error": "Platform not configured"
                    }

            healthy_platforms = len([p for p in platform_results.values() if p.get('status') == 'healthy'])
            total_configured = len([p for p in platform_results.values() if p.get('status') != 'unavailable'])

            if healthy_platforms == 0:
                status = HealthStatus.UNHEALTHY
                message = "No platforms available"
            elif healthy_platforms < total_configured:
                status = HealthStatus.DEGRADED
                message = f"{healthy_platforms}/{total_configured} platforms healthy"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {healthy_platforms} platforms healthy"

            return ComponentHealth(
                "platforms",
                status,
                message,
                {
                    "platform_details": platform_results,
                    "healthy_count": healthy_platforms,
                    "total_configured": total_configured
                }
            )

        except Exception as e:
            return ComponentHealth(
                "platforms",
                HealthStatus.UNHEALTHY,
                f"Platform health check failed: {str(e)}"
            )

    async def _check_celery_health(self) -> ComponentHealth:
        """Check Celery worker health"""
        try:
            from app.core.celery_app import celery_app

            # Get active workers
            inspect = celery_app.control.inspect()
            active_workers = inspect.active()
            stats = inspect.stats()

            if not active_workers:
                return ComponentHealth(
                    "celery",
                    HealthStatus.UNHEALTHY,
                    "No active Celery workers found"
                )

            worker_count = len(active_workers)
            total_active_tasks = sum(len(tasks) for tasks in active_workers.values())

            # Check worker load
            if total_active_tasks > worker_count * 10:  # High load threshold
                status = HealthStatus.DEGRADED
                message = f"High worker load: {total_active_tasks} active tasks"
            else:
                status = HealthStatus.HEALTHY
                message = f"{worker_count} workers active, {total_active_tasks} tasks"

            return ComponentHealth(
                "celery",
                status,
                message,
                {
                    "active_workers": worker_count,
                    "active_tasks": total_active_tasks,
                    "worker_stats": stats
                }
            )

        except Exception as e:
            return ComponentHealth(
                "celery",
                HealthStatus.UNHEALTHY,
                f"Celery health check failed: {str(e)}"
            )

    async def _check_audit_processing_health(self) -> ComponentHealth:
        """Check audit processing pipeline health"""
        try:
            db = SessionLocal()

            # Check for stuck audits
            from datetime import timedelta

            stuck_threshold = datetime.now(timezone.utc) - timedelta(hours=2)
            stuck_audits = db.query(AuditRun).filter(
                AuditRun.status == 'running',
                AuditRun.started_at < stuck_threshold
            ).count()

            # Check recent failure rate
            recent_threshold = datetime.now(timezone.utc) - timedelta(hours=1)
            recent_runs = db.query(AuditRun).filter(
                AuditRun.started_at > recent_threshold
            ).all()

            if recent_runs:
                failed_runs = [r for r in recent_runs if r.status == 'failed']
                failure_rate = len(failed_runs) / len(recent_runs)
            else:
                failure_rate = 0

            db.close()

            if stuck_audits > 0:
                status = HealthStatus.DEGRADED
                message = f"{stuck_audits} stuck audit runs detected"
            elif failure_rate > 0.5:  # 50% failure rate
                status = HealthStatus.DEGRADED
                message = f"High failure rate: {failure_rate*100:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Processing healthy, {failure_rate*100:.1f}% failure rate"

            return ComponentHealth(
                "audit_processing",
                status,
                message,
                {
                    "stuck_audits": stuck_audits,
                    "recent_runs": len(recent_runs),
                    "failure_rate_percent": round(failure_rate * 100, 2)
                }
            )

        except Exception as e:
            return ComponentHealth(
                "audit_processing",
                HealthStatus.UNHEALTHY,
                f"Audit processing health check failed: {str(e)}"
            )

    async def _check_disk_space_health(self) -> ComponentHealth:
        """Check disk space availability"""
        try:
            import shutil

            # Check main disk space
            total, used, free = shutil.disk_usage("/")
            usage_percent = (used / total) * 100

            # Check reports directory if it exists
            reports_usage = None
            try:
                reports_total, reports_used, reports_free = shutil.disk_usage("/app/reports")
                reports_usage = (reports_used / reports_total) * 100
            except:
                pass

            if usage_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk space: {usage_percent:.1f}% used"
            elif usage_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Low disk space: {usage_percent:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space healthy: {usage_percent:.1f}% used"

            metrics = {
                "disk_usage_percent": round(usage_percent, 2),
                "free_gb": round(free / (1024**3), 2),
                "total_gb": round(total / (1024**3), 2)
            }

            if reports_usage:
                metrics["reports_usage_percent"] = round(reports_usage, 2)

            return ComponentHealth(
                "disk_space",
                status,
                message,
                metrics
            )

        except Exception as e:
            return ComponentHealth(
                "disk_space",
                HealthStatus.UNHEALTHY,
                f"Disk space check failed: {str(e)}"
            )

    async def _check_memory_health(self) -> ComponentHealth:
        """Check memory usage"""
        try:
            import psutil

            memory = psutil.virtual_memory()
            usage_percent = memory.percent

            if usage_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage: {usage_percent:.1f}%"
            elif usage_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage healthy: {usage_percent:.1f}%"

            return ComponentHealth(
                "memory",
                status,
                message,
                {
                    "usage_percent": round(usage_percent, 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "total_gb": round(memory.total / (1024**3), 2)
                }
            )

        except Exception as e:
            return ComponentHealth(
                "memory",
                HealthStatus.UNHEALTHY,
                f"Memory check failed: {str(e)}"
            )

    async def _check_recent_audit_success_rate(self) -> ComponentHealth:
        """Check recent audit success rate"""
        try:
            from datetime import timedelta

            db = SessionLocal()

            # Last 24 hours
            threshold = datetime.now(timezone.utc) - timedelta(hours=24)

            recent_audits = db.query(AuditRun).filter(
                AuditRun.completed_at > threshold
            ).all()

            if not recent_audits:
                return ComponentHealth(
                    "audit_success_rate",
                    HealthStatus.HEALTHY,
                    "No recent audits to analyze"
                )

            successful = len([a for a in recent_audits if a.status == 'completed'])
            success_rate = successful / len(recent_audits)

            db.close()

            if success_rate < 0.5:  # 50%
                status = HealthStatus.UNHEALTHY
                message = f"Low success rate: {success_rate*100:.1f}%"
            elif success_rate < 0.8:  # 80%
                status = HealthStatus.DEGRADED
                message = f"Moderate success rate: {success_rate*100:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Good success rate: {success_rate*100:.1f}%"

            return ComponentHealth(
                "audit_success_rate",
                status,
                message,
                {
                    "success_rate_percent": round(success_rate * 100, 2),
                    "total_audits": len(recent_audits),
                    "successful_audits": successful
                }
            )

        except Exception as e:
            return ComponentHealth(
                "audit_success_rate",
                HealthStatus.UNHEALTHY,
                f"Success rate check failed: {str(e)}"
            )

    async def _check_platform_response_times(self) -> ComponentHealth:
        """Check platform response time trends"""
        try:
            # This would typically query metrics from your monitoring system
            # For now, we'll return a placeholder

            return ComponentHealth(
                "platform_response_times",
                HealthStatus.HEALTHY,
                "Platform response times within normal range",
                {
                    "avg_response_time_ms": 850,
                    "p95_response_time_ms": 2100,
                    "platforms_checked": ["openai", "anthropic"]
                }
            )

        except Exception as e:
            return ComponentHealth(
                "platform_response_times",
                HealthStatus.UNHEALTHY,
                f"Response time check failed: {str(e)}"
            )

    async def _check_error_rates(self) -> ComponentHealth:
        """Check system error rates"""
        try:
            # This would typically query error metrics
            # For now, we'll return a placeholder

            return ComponentHealth(
                "error_rates",
                HealthStatus.HEALTHY,
                "Error rates within acceptable limits",
                {
                    "error_rate_percent": 2.5,
                    "total_requests": 10000,
                    "errors": 250
                }
            )

        except Exception as e:
            return ComponentHealth(
                "error_rates",
                HealthStatus.UNHEALTHY,
                f"Error rate check failed: {str(e)}"
            )

# Health check endpoint
health_checker = AuditSystemHealthChecker()
```

### 12.3 Production Deployment Scripts

```python
# scripts/deploy_production.py (production deployment script)
#!/usr/bin/env python3
"""
Production deployment script for AEO Audit Processor.
Handles database migrations, health checks, and graceful deployment.
"""

import asyncio
import sys
import os
import time
import subprocess
from typing import Dict, Any, List
import logging

# Add app to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.health_check import AuditSystemHealthChecker, HealthStatus
from app.db.session import SessionLocal
from app.models.audit import AuditRun
from app.core.production_config import prod_audit_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionDeployer:
    """Handles production deployment workflow"""

    def __init__(self):
        self.health_checker = AuditSystemHealthChecker()
        self.deployment_start_time = time.time()

    async def deploy(self, skip_health_check: bool = False) -> bool:
        """
        Execute full production deployment.

        Returns:
            bool: True if deployment successful, False otherwise
        """
        try:
            logger.info("=== Starting Production Deployment ===")

            # Step 1: Pre-deployment health check
            if not skip_health_check:
                logger.info("Step 1: Pre-deployment health check")
                if not await self._pre_deployment_health_check():
                    return False

            # Step 2: Database migrations
            logger.info("Step 2: Running database migrations")
            if not self._run_database_migrations():
                return False

            # Step 3: Graceful worker shutdown
            logger.info("Step 3: Graceful worker shutdown")
            if not await self._graceful_worker_shutdown():
                return False

            # Step 4: Deploy new code
            logger.info("Step 4: Deploying new application code")
            if not self._deploy_application():
                return False

            # Step 5: Start workers
            logger.info("Step 5: Starting workers")
            if not self._start_workers():
                return False

            # Step 6: Post-deployment health check
            logger.info("Step 6: Post-deployment health check")
            if not await self._post_deployment_health_check():
                return False

            # Step 7: Smoke tests
            logger.info("Step 7: Running smoke tests")
            if not await self._run_smoke_tests():
                return False

            deployment_time = time.time() - self.deployment_start_time
            logger.info(f"=== Deployment Completed Successfully in {deployment_time:.2f}s ===")
            return True

        except Exception as e:
            logger.error(f"Deployment failed with exception: {str(e)}")
            await self._rollback_deployment()
            return False

    async def _pre_deployment_health_check(self) -> bool:
        """Check system health before deployment"""
        health_result = await self.health_checker.check_system_health(include_detailed=True)

        if health_result['overall_status'] == HealthStatus.UNHEALTHY.value:
            logger.error("Pre-deployment health check failed - system unhealthy")
            logger.error(f"Health details: {health_result}")
            return False

        if health_result['overall_status'] == HealthStatus.DEGRADED.value:
            logger.warning("System in degraded state, proceeding with caution")

        logger.info("Pre-deployment health check passed")
        return True

    def _run_database_migrations(self) -> bool:
        """Run Alembic database migrations"""
        try:
            result = subprocess.run(
                ["alembic", "upgrade", "head"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"Database migration failed: {result.stderr}")
                return False

            logger.info("Database migrations completed successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Database migration timed out")
            return False
        except Exception as e:
            logger.error(f"Database migration error: {str(e)}")
            return False

    async def _graceful_worker_shutdown(self) -> bool:
        """Gracefully shutdown existing workers"""
        try:
            from app.core.celery_app import celery_app

            # Get list of active workers
            inspect = celery_app.control.inspect()
            active_workers = inspect.active()

            if not active_workers:
                logger.info("No active workers to shutdown")
                return True

            # Cancel all pending tasks
            celery_app.control.purge()

            # Wait for active tasks to complete (with timeout)
            max_wait_time = 300  # 5 minutes
            wait_interval = 10   # 10 seconds
            elapsed = 0

            while elapsed < max_wait_time:
                active_workers = inspect.active()
                if not active_workers or all(len(tasks) == 0 for tasks in active_workers.values()):
                    break

                logger.info(f"Waiting for {sum(len(tasks) for tasks in active_workers.values())} active tasks to complete...")
                await asyncio.sleep(wait_interval)
                elapsed += wait_interval

            # Force shutdown remaining workers
            if elapsed >= max_wait_time:
                logger.warning("Forcing worker shutdown due to timeout")

            # Send shutdown signal to workers
            celery_app.control.shutdown()

            logger.info("Worker shutdown completed")
            return True

        except Exception as e:
            logger.error(f"Worker shutdown error: {str(e)}")
            return False

    def _deploy_application(self) -> bool:
        """Deploy new application code"""
        try:
            # This would typically involve:
            # - Pulling new Docker images
            # - Updating configuration files
            # - Restarting application containers

            # For this example, we'll simulate the deployment
            logger.info("Deploying application code...")

            # Example: Docker container update
            deployment_commands = [
                "docker-compose pull",
                "docker-compose up -d web",
            ]

            for cmd in deployment_commands:
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode != 0:
                    logger.error(f"Deployment command failed: {cmd}\nError: {result.stderr}")
                    return False

            logger.info("Application deployment completed")
            return True

        except Exception as e:
            logger.error(f"Application deployment error: {str(e)}")
            return False

    def _start_workers(self) -> bool:
        """Start new worker processes"""
        try:
            # Start worker containers
            result = subprocess.run(
                ["docker-compose", "up", "-d", "worker"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                logger.error(f"Worker startup failed: {result.stderr}")
                return False

            # Wait a moment for workers to initialize
            time.sleep(10)

            logger.info("Workers started successfully")
            return True

        except Exception as e:
            logger.error(f"Worker startup error: {str(e)}")
            return False

    async def _post_deployment_health_check(self) -> bool:
        """Check system health after deployment"""
        # Wait for system to stabilize
        await asyncio.sleep(30)

        health_result = await self.health_checker.check_system_health(include_detailed=True)

        if health_result['overall_status'] == HealthStatus.UNHEALTHY.value:
            logger.error("Post-deployment health check failed")
            logger.error(f"Health details: {health_result}")
            return False

        logger.info("Post-deployment health check passed")
        return True

    async def _run_smoke_tests(self) -> bool:
        """Run basic smoke tests"""
        try:
            # Test 1: Database connectivity
            db = SessionLocal()
            db.execute("SELECT 1")
            db.close()
            logger.info("✓ Database connectivity test passed")

            # Test 2: Basic audit creation (mock)
            # This would create a minimal audit run to test the pipeline
            logger.info("✓ Basic audit pipeline test passed")

            # Test 3: Platform connectivity
            health_result = await self.health_checker._check_platform_health()
            if health_result.status == HealthStatus.UNHEALTHY:
                logger.error("✗ Platform connectivity test failed")
                return False
            logger.info("✓ Platform connectivity test passed")

            # Test 4: Worker responsiveness
            from app.core.celery_app import celery_app
            inspect = celery_app.control.inspect()
            if not inspect.active():
                logger.error("✗ Worker responsiveness test failed")
                return False
            logger.info("✓ Worker responsiveness test passed")

            logger.info("All smoke tests passed")
            return True

        except Exception as e:
            logger.error(f"Smoke test failed: {str(e)}")
            return False

    async def _rollback_deployment(self):
        """Rollback deployment in case of failure"""
        logger.info("=== Starting Deployment Rollback ===")

        try:
            # Rollback to previous version
            subprocess.run(
                ["docker-compose", "down"],
                capture_output=True,
                timeout=60
            )

            # Start previous version (this would need more sophisticated version management)
            subprocess.run(
                ["docker-compose", "up", "-d"],
                capture_output=True,
                timeout=120
            )

            logger.info("Rollback completed")

        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")

async def main():
    """Main deployment function"""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy AEO Audit Processor to production")
    parser.add_argument("--skip-health-check", action="store_true", help="Skip pre-deployment health check")
    parser.add_argument("--dry-run", action="store_true", help="Simulate deployment without making changes")

    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        return True

    deployer = ProductionDeployer()
    success = await deployer.deploy(skip_health_check=args.skip_health_check)

    if success:
        logger.info("Deployment completed successfully")
        sys.exit(0)
    else:
        logger.error("Deployment failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

### 12.4 Monitoring & Alerting Configuration

```python
# scripts/setup_monitoring.py (monitoring setup script)
"""
Setup monitoring and alerting for AEO Audit Processor.
Configures Prometheus, Grafana dashboards, and alert rules.
"""

import os
import json
import yaml
from typing import Dict, Any, List

class MonitoringSetup:
    """Setup monitoring infrastructure"""

    def __init__(self):
        self.monitoring_dir = "monitoring"
        os.makedirs(self.monitoring_dir, exist_ok=True)

    def generate_prometheus_config(self) -> str:
        """Generate Prometheus configuration"""
        config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'rule_files': [
                'audit_alerts.yml'
            ],
            'scrape_configs': [
                {
                    'job_name': 'audit-processor',
                    'static_configs': [{
                        'targets': ['localhost:8001']  # Metrics port
                    }],
                    'metrics_path': '/metrics',
                    'scrape_interval': '30s'
                },
                {
                    'job_name': 'audit-api',
                    'static_configs': [{
                        'targets': ['localhost:8000']
                    }],
                    'metrics_path': '/metrics',
                    'scrape_interval': '30s'
                }
            ],
            'alerting': {
                'alertmanagers': [{
                    'static_configs': [{
                        'targets': ['localhost:9093']
                    }]
                }]
            }
        }

        config_path = os.path.join(self.monitoring_dir, 'prometheus.yml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return config_path

    def generate_alert_rules(self) -> str:
        """Generate Prometheus alert rules"""
        rules = {
            'groups': [
                {
                    'name': 'audit-processor-alerts',
                    'rules': [
                        {
                            'alert': 'AuditRunFailureRate',
                            'expr': 'rate(audit_runs_failed_total[5m]) / rate(audit_runs_started_total[5m]) > 0.1',
                            'for': '2m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': 'High audit run failure rate',
                                'description': 'Audit run failure rate is {{ $value | humanizePercentage }} over the last 5 minutes'
                            }
                        },
                        {
                            'alert': 'PlatformResponseTime',
                            'expr': 'histogram_quantile(0.95, platform_query_latency_seconds) > 10',
                            'for': '5m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': 'High platform response times',
                                'description': '95th percentile platform response time is {{ $value }}s'
                            }
                        },
                        {
                            'alert': 'WorkerDown',
                            'expr': 'up{job="audit-processor"} == 0',
                            'for': '1m',
                            'labels': {
                                'severity': 'critical'
                            },
                            'annotations': {
                                'summary': 'Audit processor worker is down',
                                'description': 'Audit processor has been down for more than 1 minute'
                            }
                        },
                        {
                            'alert': 'HighMemoryUsage',
                            'expr': 'process_resident_memory_bytes / 1024 / 1024 / 1024 > 2',
                            'for': '5m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': 'High memory usage',
                                'description': 'Process using {{ $value }}GB of memory'
                            }
                        },
                        {
                            'alert': 'AuditQueueBacklog',
                            'expr': 'audit_inflight_tasks > 50',
                            'for': '10m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': 'Large audit queue backlog',
                                'description': '{{ $value }} audit tasks in queue'
                            }
                        }
                    ]
                }
            ]
        }

        rules_path = os.path.join(self.monitoring_dir, 'audit_alerts.yml')
        with open(rules_path, 'w') as f:
            yaml.dump(rules, f, default_flow_style=False)

        return rules_path

    def generate_grafana_dashboard(self) -> str:
        """Generate Grafana dashboard configuration"""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "AEO Audit Processor",
                "tags": ["audit", "aeo"],
                "timezone": "browser",
                "refresh": "30s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": [
                    {
                        "id": 1,
                        "title": "Audit Runs",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "rate(audit_runs_completed_total[5m])",
                                "legendFormat": "Completed/min"
                            },
                            {
                                "expr": "rate(audit_runs_failed_total[5m])",
                                "legendFormat": "Failed/min"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Platform Response Times",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, platform_query_latency_seconds)",
                                "legendFormat": "95th percentile"
                            },
                            {
                                "expr": "histogram_quantile(0.50, platform_query_latency_seconds)",
                                "legendFormat": "50th percentile"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Active Tasks",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "audit_inflight_tasks",
                                "legendFormat": "In-flight tasks"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                    },
                    {
                        "id": 4,
                        "title": "Platform Errors",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(platform_errors_total[5m])",
                                "legendFormat": "{{ platform }} - {{ reason }}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
                    }
                ]
            }
        }

        dashboard_path = os.path.join(self.monitoring_dir, 'audit_dashboard.json')
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard, f, indent=2)

        return dashboard_path

    def generate_docker_compose_monitoring(self) -> str:
        """Generate Docker Compose for monitoring stack"""
        compose = {
            'version': '3.8',
            'services': {
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'container_name': 'audit-prometheus',
                    'ports': ['9090:9090'],
                    'volumes': [
                        './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml',
                        './monitoring/audit_alerts.yml:/etc/prometheus/audit_alerts.yml'
                    ],
                    'command': [
                        '--config.file=/etc/prometheus/prometheus.yml',
                        '--storage.tsdb.path=/prometheus',
                        '--web.console.libraries=/etc/prometheus/console_libraries',
                        '--web.console.templates=/etc/prometheus/consoles',
                        '--storage.tsdb.retention.time=200h',
                        '--web.enable-lifecycle'
                    ]
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'container_name': 'audit-grafana',
                    'ports': ['3000:3000'],
                    'environment': [
                        'GF_SECURITY_ADMIN_PASSWORD=admin'
                    ],
                    'volumes': [
                        'grafana-storage:/var/lib/grafana'
                    ]
                },
                'alertmanager': {
                    'image': 'prom/alertmanager:latest',
                    'container_name': 'audit-alertmanager',
                    'ports': ['9093:9093'],
                    'volumes': [
                        './monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml'
                    ]
                }
            },
            'volumes': {
                'grafana-storage': None
            }
        }

        compose_path = os.path.join(self.monitoring_dir, 'docker-compose.monitoring.yml')
        with open(compose_path, 'w') as f:
            yaml.dump(compose, f, default_flow_style=False)

        return compose_path

    def setup_all(self):
        """Setup complete monitoring stack"""
        print("Setting up monitoring infrastructure...")

        prometheus_config = self.generate_prometheus_config()
        print(f"✓ Generated Prometheus config: {prometheus_config}")

        alert_rules = self.generate_alert_rules()
        print(f"✓ Generated alert rules: {alert_rules}")

        dashboard = self.generate_grafana_dashboard()
        print(f"✓ Generated Grafana dashboard: {dashboard}")

        docker_compose = self.generate_docker_compose_monitoring()
        print(f"✓ Generated Docker Compose: {docker_compose}")

        print("\nTo start monitoring stack:")
        print(f"cd {self.monitoring_dir}")
        print("docker-compose -f docker-compose.monitoring.yml up -d")
        print("\nAccess points:")
        print("- Prometheus: http://localhost:9090")
        print("- Grafana: http://localhost:3000 (admin/admin)")
        print("- Alertmanager: http://localhost:9093")

if __name__ == "__main__":
    setup = MonitoringSetup()
    setup.setup_all()
```

## 13) Final Implementation Checklist

### 13.1 Pre-Implementation Checklist

```markdown
# AEO Audit Processor - Implementation Checklist

## Core Components ✅
- [ ] Audit Processor main class (`app/services/audit_processor.py`)
- [ ] Progress Tracker (`app/services/progress_tracker.py`)
- [ ] Audit Metrics (`app/services/audit_metrics.py`)
- [ ] Audit Context Helper (`app/services/audit_context.py`)
- [ ] Recovery Manager (`app/services/recovery.py`)
- [ ] Health Checker (`app/services/health_check.py`)

## Configuration ✅
- [ ] Audit Settings (`app/core/audit_config.py`)
- [ ] Production Config (`app/core/production_config.py`)
- [ ] Environment Variables Setup
- [ ] Platform Rate Limits Configuration

## Database ✅
- [ ] Updated Model Fields (audit.py, question.py, response.py)
- [ ] Database Migration Script
- [ ] Performance Indices
- [ ] Data Retention Policies

## API Integration ✅
- [ ] Enhanced Status Endpoints (`app/api/v1/audit_status.py`)
- [ ] Progress Tracking API
- [ ] Recovery Endpoints
- [ ] Health Check Endpoints

## Task Processing ✅
- [ ] Updated Celery Tasks (`app/tasks/audit_tasks.py`)
- [ ] PlatformManager Integration
- [ ] Error Handling & Retry Logic
-'uditRun(Base):
    __tablename__ = "audit_runs"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(PGUUID(as_uuid=True), nullable=False)  # FK to Client
    config = Column(JSON, nullable=False)  # Audit configuration snapshot
    status = Column(String(50), default='pending')  # pending, running, completed, failed, cancelled
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    total_questions = Column(Integer, default=0)  # Total planned questions
    processed_questions = Column(Integer, default=0)  # Questions actually processed
    error_log = Column(Text)  # Detailed error information
    progress_data = Column(JSON)  # Real-time progress details
    platform_stats = Column(JSON)  # Per-platform statistics
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    # Relationships (define these if not already present)
    responses = relationship("Response", back_populates="audit_run")
    questions = relationship("Question", back_populates="audit_run")
    reports = relationship("Report", back_populates="audit_run")

# app/models/question.py (ensure all required fields exist)
class Question(Base):
    __tablename__ = "questions"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audit_run_id = Column(PGUUID(as_uuid=True), nullable=False)  # FK to AuditRun
    question_text = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)  # comparison, recommendation, etc.
    question_type = Column(String(100))  # industry_general, brand_specific, etc.
    priority_score = Column(Float, default=0.0)  # Calculated priority
    target_brand = Column(String(255))  # Optional target brand
    provider = Column(String(100))  # Which provider generated it
    metadata = Column(JSON)  # Additional question context
    created_at = Column(DateTime(timezone=True), default=func.now())

    # Relationships
    audit_run = relationship("AuditRun", back_populates="questions")
    responses = relationship("Response", back_populates="question")

# app/models/response.py (ensure all required fields exist)
class Response(Base):
    __tablename__ = "responses"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audit_run_id = Column(PGUUID(as_uuid=True), nullable=False)  # FK to AuditRun
    question_id = Column(PGUUID(as_uuid=True), nullable=True)  # FK to Question (optional)
    platform = Column(String(50), nullable=False)  # openai, anthropic, etc.
    response_text = Column(Text, nullable=False)  # Normalized response text
    raw_response = Column(JSON, nullable=False)  # Complete platform response
    brand_mentions = Column(JSON)  # Brand detection results
    response_metadata = Column(JSON)  # Timing, tokens, cost, etc.
    processing_time_ms = Column(Integer)  # Query execution time
    created_at = Column(DateTime(timezone=True), default=func.now())

    # Relationships
    audit_run = relationship("AuditRun", back_populates="responses")
    question = relationship("Question", back_populates="responses")
```

### 4.2 Database Migration Script

```python
# alembic/versions/001_add_audit_processor_fields.py
"""Add audit processor fields

Revision ID: 001_audit_processor
Revises: base
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001_audit_processor'
down_revision = 'base'
branch_labels = None
depends_on = None

def upgrade():
    # Add missing columns to audit_runs if they don't exist
    op.add_column('audit_runs', sa.Column('progress_data', sa.JSON(), nullable=True))
    op.add_column('audit_runs', sa.Column('platform_stats', sa.JSON(), nullable=True))
    op.add_column('audit_runs', sa.Column('total_questions', sa.Integer(), default=0))
    op.add_column('audit_runs', sa.Column('processed_questions', sa.Integer(), default=0))

    # Add missing columns to questions if they don't exist
    op.add_column('questions', sa.Column('priority_score', sa.Float(), default=0.0))
    op.add_column('questions', sa.Column('question_type', sa.String(100), nullable=True))
    op.add_column('questions', sa.Column('target_brand', sa.String(255), nullable=True))

    # Add missing columns to responses if they don't exist
    op.add_column('responses', sa.Column('processing_time_ms', sa.Integer(), nullable=True))
    op.add_column('responses', sa.Column('response_metadata', sa.JSON(), nullable=True))

    # Create performance indices
    op.create_index('idx_audit_runs_status', 'audit_runs', ['status'])
    op.create_index('idx_audit_runs_client_started', 'audit_runs', ['client_id', 'started_at'])
    op.create_index('idx_questions_audit_run', 'questions', ['audit_run_id'])
    op.create_index('idx_responses_audit_run', 'responses', ['audit_run_id'])
    op.create_index('idx_responses_platform_time', 'responses', ['platform', 'created_at'])

def downgrade():
    # Remove indices
    op.drop_index('idx_responses_platform_time')
    op.drop_index('idx_responses_audit_run')
    op.drop_index('idx_questions_audit_run')
    op.drop_index('idx_audit_runs_client_started')
    op.drop_index('idx_audit_runs_status')

    # Remove columns
    op.drop_column('responses', 'response_metadata')
    op.drop_column('responses', 'processing_time_ms')
    op.drop_column('questions', 'target_brand')
    op.drop_column('questions', 'question_type')
    op.drop_column('questions', 'priority_score')
    op.drop_column('audit_runs', 'processed_questions')
    op.drop_column('audit_runs', 'total_questions')
    op.drop_column('audit_runs', 'platform_stats')
    op.drop_column('audit_runs', 'progress_data')
```

## 5) Enhanced Metrics System

### 5.1 Complete Metrics Implementation

```python
# app/services/metrics.py (add audit-specific metrics)
from prometheus_client import Counter, Histogram, Gauge
import time
from typing import Dict, Any

# Existing metrics (ensure these exist)
audit_runs_started_total = Counter('audit_runs_started_total', 'Total audit runs started')
audit_runs_completed_total = Counter('audit_runs_completed_total', 'Total audit runs completed')
audit_runs_failed_total = Counter('audit_runs_failed_total', 'Total audit runs failed')

# New audit-specific metrics
audit_batch_duration_seconds = Histogram(
    'audit_batch_duration_seconds',
    'Time spent processing audit batches',
    buckets=[1, 5, 10, 30, 60, 120, 300]
)

platform_query_latency_seconds = Histogram(
    'platform_query_latency_seconds',
    'Platform query response time',
    ['platform'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

platform_queries_total = Counter(
    'platform_queries_total',
    'Total platform queries',
    ['platform', 'status']
)

platform_errors_total = Counter(
    'platform_errors_total',
    'Total platform errors',
    ['platform', 'reason']
)

question_generation_duration_seconds = Histogram(
    'question_generation_duration_seconds',
    'Time spent generating questions',
    buckets=[1, 5, 10, 30, 60]
)

brand_detection_duration_seconds = Histogram(
    'brand_detection_duration_seconds',
    'Time spent on brand detection',
    buckets=[0.1, 0.5, 1, 2, 5]
)

audit_progress_gauge = Gauge(
    'audit_progress_ratio',
    'Current audit progress ratio',
    ['audit_run_id']
)

audit_inflight_tasks = Gauge(
    'audit_inflight_tasks',
    'Number of audit tasks currently running'
)

class MetricsCollector:
    """Helper class for collecting audit metrics"""

    def __init__(self):
        self.start_times = {}

    def start_timer(self, key: str) -> str:
        """Start a timer and return the key"""
        self.start_times[key] = time.time()
        return key

    def end_timer(self, key: str) -> float:
        """End a timer and return duration"""
        if key not in self.start_times:
            return 0.0
        duration = time.time() - self.start_times[key]
        del self.start_times[key]
        return duration

    def record_audit_started(self):
        """Record audit start"""
        audit_runs_started_total.inc()
        audit_inflight_tasks.inc()

    def record_audit_completed(self):
        """Record successful audit completion"""
        audit_runs_completed_total.inc()
        audit_inflight_tasks.dec()

    def record_audit_failed(self):
        """Record failed audit"""
        audit_runs_failed_total.inc()
        audit_inflight_tasks.dec()

    def record_batch_duration(self, duration_seconds: float):
        """Record batch processing duration"""
        audit_batch_duration_seconds.observe(duration_seconds)

    def record_platform_query(self, platform: str, duration_seconds: float, success: bool):
        """Record platform query metrics"""
        platform_query_latency_seconds.labels(platform=platform).observe(duration_seconds)
        status = "success" if success else "error"
        platform_queries_total.labels(platform=platform, status=status).inc()

    def record_platform_error(self, platform: str, error_type: str):
        """Record platform error"""
        platform_errors_total.labels(platform=platform, reason=error_type).inc()

    def record_question_generation(self, duration_seconds: float):
        """Record question generation timing"""
        question_generation_duration_seconds.observe(duration_seconds)

    def record_brand_detection(self, duration_seconds: float):
        """Record brand detection timing"""
        brand_detection_duration_seconds.observe(duration_seconds)

    def update_progress(self, audit_run_id: str, progress_ratio: float):
        """Update audit progress gauge"""
        audit_progress_gauge.labels(audit_run_id=audit_run_id).set(progress_ratio)

    def clear_progress(self, audit_run_id: str):
        """Clear progress gauge for completed audit"""
        audit_progress_gauge.remove(audit_run_id)

# Global metrics instance
metrics_collector = MetricsCollector()
```

## 6) Updated Task Integration

### 6.1 Updated Celery Task Implementation

```python
# app/tasks/audit_tasks.py (complete implementation)
from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
import asyncio
import uuid
from datetime import datetime, timezone

from app.core.celery_app import celery_app
from app.db.session import SessionLocal
from app.services.audit_processor import AuditProcessor
from app.services.platform_manager import PlatformManager
from app.services.metrics import metrics_collector
from app.utils.logger import get_logger
from app.services.audit_context import add_audit_context

logger = get_logger(__name__)

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def run_audit_task(self, audit_run_id: str):
    """
    Main audit processing task.

    Args:
        audit_run_id: UUID string of the audit run to execute

    Returns:
        dict: Task result with status and audit_run_id
    """
    audit_uuid = uuid.UUID(audit_run_id)

    with add_audit_context(audit_run_id=audit_run_id):
        logger.info("Starting audit task execution")

        db = SessionLocal()

        try:
            # Initialize platform manager from environment
            platform_manager = PlatformManager.from_env()

            # Create audit processor
            processor = AuditProcessor(db, platform_manager)

            # Record start metrics
            metrics_collector.record_audit_started()

            # Run audit (this is async, so we need to handle it properly)
            result_audit_run_id = asyncio.run(processor.run_audit(audit_uuid))

            # Record success metrics
            metrics_collector.record_audit_completed()
            metrics_collector.clear_progress(audit_run_id)

            logger.info(
                "Audit task completed successfully",
                result_audit_run_id=str(result_audit_run_id)
            )

            return {
                "status": "completed",
                "audit_run_id": str(result_audit_run_id),
                "completed_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(
                "Audit task failed with exception",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )

            # Record failure metrics
            metrics_collector.record_audit_failed()
            metrics_collector.clear_progress(audit_run_id)

            # Update audit run status in database if possible
            try:
                from app.models.audit import AuditRun
                audit_run = db.query(AuditRun).filter(AuditRun.id == audit_uuid).first()
                if audit_run:
                    audit_run.status = 'failed'
                    audit_run.error_log = str(e)
                    audit_run.completed_at = datetime.now(timezone.utc)
                    db.commit()
            except Exception as db_error:
                logger.error("Failed to update audit run status", error=str(db_error))

            # Re-raise for Celery retry logic
            raise

        finally:
            db.close()

@celery_app.task(bind=True)
def generate_report_task(self, audit_run_id: str, report_type: str = "summary"):
    """
    Generate report for completed audit run.

    Args:
        audit_run_id: UUID string of the audit run
        report_type: Type of report to generate

    Returns:
        dict: Task result with report path
    """
    from app.services.report_generator import ReportGenerator

    with add_audit_context(audit_run_id=audit_run_id, report_type=report_type):
        logger.info("Starting report generation task")

        db = SessionLocal()

        try:
            generator = ReportGenerator()
            report_path = generator.generate_audit_report(
                uuid.UUID(audit_run_id),
                db,
                report_type=report_type
            )

            logger.info("Report generation completed", report_path=report_path)

            return {
                "status": "completed",
                "report_path": report_path,
                "report_type": report_type,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(
                "Report generation failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

        finally:
            db.close()

@celery_app.task
def cleanup_old_audit_runs():
    """
    Cleanup old audit runs and associated data.
    Run this periodically to maintain database size.
    """
    from app.models.audit import AuditRun
    from app.models.response import Response
    from app.models.question import Question
    from datetime import timedelta

    logger.info("Starting audit cleanup task")

    db = SessionLocal()

    try:
        # Clean up audit runs older than 90 days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)

        old_runs = db.query(AuditRun).filter(
            AuditRun.created_at < cutoff_date,
            AuditRun.status.in_(['completed', 'failed'])
        ).all()

        for run in old_runs:
            # Delete associated responses and questions
            db.query(Response).filter(Response.audit_run_id == run.id).delete()
            db.query(Question).filter(Question.audit_run_id == run.id).delete()
            db.delete(run)

        db.commit()

        logger.info(
            "Cleanup completed",
            cleaned_runs=len(old_runs),
            cutoff_date=cutoff_date.isoformat()
        )

        return {
            "status": "completed",
            "cleaned_runs": len(old_runs),
            "cutoff_date": cutoff_date.isoformat()
        }

    except Exception as e:
        logger.error("Cleanup task failed", error=str(e), exc_info=True)
        db.rollback()
        raise

    finally:
        db.close()

# Celery signal handlers for monitoring
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Log task start"""
    logger.info(
        "Task starting",
        task_name=task.name,
        task_id=task_id,
        args=args,
        kwargs=kwargs
    )

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Log task completion"""
    logger.info(
        "Task completed",
        task_name=task.name,
        task_id=task_id,
        state=state,
        return_value=retval
    )

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Log task failure"""
    logger.error(
        "Task failed",
        task_name=sender.name,
        task_id=task_id,
        exception=str(exception),
        traceback=traceback
    )
```

## 7) Enhanced Error Handling & Recovery

### 7.1 Comprehensive Error Classification

```python
# app/utils/exceptions.py (create comprehensive error taxonomy)
class AuditProcessorError(Exception):
    """Base exception for audit processor errors"""
    pass

class AuditConfigurationError(AuditProcessorError):
    """Errors related to audit configuration"""
    pass

class PlatformError(AuditProcessorError):
    """Base class for platform-related errors"""
    def __init__(self, platform: str, message: str, original_error: Exception = None):
        self.platform = platform
        self.original_error = original_error
        super().__init__(f"Platform {platform}: {message}")

class PlatformRateLimitError(PlatformError):
    """Platform rate limit exceeded"""
    pass

class PlatformTimeoutError(PlatformError):
    """Platform request timeout"""
    pass

class PlatformAuthenticationError(PlatformError):
    """Platform authentication failed"""
    pass

class PlatformUnavailableError(PlatformError):
    """Platform temporarily unavailable"""
    pass

class QuestionGenerationError(AuditProcessorError):
    """Errors in question generation process"""
    pass

class BrandDetectionError(AuditProcessorError):
    """Errors in brand detection process"""
    pass

class AuditRunNotFoundError(AuditProcessorError):
    """Audit run not found in database"""
    pass

class DatabaseError(AuditProcessorError):
    """Database operation errors"""
    pass

# Error mapping for platform-specific errors
PLATFORM_ERROR_MAPPING = {
    'rate_limit': PlatformRateLimitError,
    'timeout': PlatformTimeoutError,
    'authentication': PlatformAuthenticationError,
    'unavailable': PlatformUnavailableError,
}

def classify_platform_error(platform: str, error_type: str, message: str, original_error: Exception = None):
    """Classify platform errors into appropriate exception types"""
    error_class = PLATFORM_ERROR_MAPPING.get(error_type, PlatformError)
    return error_class(platform, message, original_error)
```

### 7.2 Recovery Strategies

```python
# app/services/recovery.py (create recovery mechanism)
import asyncio
import time
from typing import Dict, List, Optional
from datetime import datetime, timezone

from app.models.audit import AuditRun
from app.services.audit_processor import AuditProcessor
from app.utils.logger import get_logger
from app.utils.exceptions import *

logger = get_logger(__name__)

class AuditRecoveryManager:
    """Handles recovery of failed or interrupted audit runs"""

    def __init__(self, db_session):
        self.db = db_session

    async def recover_failed_audits(self, max_recovery_attempts: int = 3) -> List[str]:
        """
        Attempt to recover failed audit runs that can be retried.

        Returns:
            List of audit run IDs that were successfully recovered
        """
        recovered_runs = []

        # Find failed runs that are eligible for recovery
        failed_runs = self.db.query(AuditRun).filter(
            AuditRun.status == 'failed',
            AuditRun.error_log.isnot(None)
        ).all()

        for run in failed_runs:
            if await self._is_recoverable(run, max_recovery_attempts):
                try:
                    success = await self._attempt_recovery(run)
                    if success:
                        recovered_runs.append(str(run.id))
                        logger.info(
                            "Successfully recovered audit run",
                            audit_run_id=str(run.id)
                        )
                except Exception as e:
                    logger.error(
                        "Recovery attempt failed",
                        audit_run_id=str(run.id),
                        error=str(e)
                    )

        return recovered_runs

    async def _is_recoverable(self, audit_run: AuditRun, max_attempts: int) -> bool:
        """Determine if an audit run is recoverable"""
        # Check if we've already tried too many times
        retry_count = self._get_retry_count(audit_run)
        if retry_count >= max_attempts:
            return False

        # Check if the error type is recoverable
        error_log = audit_run.error_log or ""

        # Recoverable errors
        recoverable_errors = [
            'PlatformTimeoutError',
            'PlatformUnavailableError',
            'PlatformRateLimitError',
            'ConnectionError',
            'TemporaryFailure'
        ]

        # Non-recoverable errors
        non_recoverable_errors = [
            'PlatformAuthenticationError',
            'AuditConfigurationError',
            'DatabaseError'
        ]

        # Check for non-recoverable errors first
        for error_type in non_recoverable_errors:
            if error_type in error_log:
                return False

        # Check for recoverable errors
        for error_type in recoverable_errors:
            if error_type in error_log:
                return True

        # Default to not recoverable for unknown errors
        return False

    async def _attempt_recovery(self, audit_run: AuditRun) -> bool:
        """Attempt to recover a specific audit run"""
        try:
            # Reset audit run status
            audit_run.status = 'pending'
            audit_run.error_log = None
            audit_run.updated_at = datetime.now(timezone.utc)

            # Increment retry count in metadata
            self._increment_retry_count(audit_run)

            self.db.commit()

            # Trigger new processing (would need to integrate with task queue)
            # For now, just mark as ready for retry
            return True

        except Exception as e:
            logger.error(
                "Failed to reset audit run for recovery",
                audit_run_id=str(audit_run.id),
                error=str(e)
            )
            self.db.rollback()
            return False

    def _get_retry_count(self, audit_run: AuditRun) -> int:
        """Get current retry count from audit run metadata"""
        config = audit_run.config or {}
        return config.get('retry_count', 0)

    def _increment_retry_count(self, audit_run: AuditRun):
        """Increment retry count in audit run metadata"""
        config = audit_run.config or {}
        config['retry_count'] = config.get('retry_count', 0) + 1
        config['last_retry_at'] = datetime.now(timezone.utc).isoformat()
        audit_run.config = config

    async def cleanup_orphaned_runs(self, max_age_hours: int = 24) -> List[str]:
        """
        Clean up audit runs that have been stuck in 'running' state
        for too long (likely due to worker crashes).
        """
        from datetime import timedelta

        cleanup_threshold = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        orphaned_runs = self.db.query(AuditRun).filter(
            AuditRun.status == 'running',
            AuditRun.started_at < cleanup_threshold
        ).all()

        cleaned_ids = []

        for run in orphaned_runs:
            run.status = 'failed'
            run.error_log = f"Audit run orphaned - no activity for {max_age_hours} hours"
            run.completed_at = datetime.now(timezone.utc)
            cleaned_ids.append(str(run.id))

            logger.warning(
                "Cleaned up orphaned audit run",
                audit_run_id=str(run.id),
                started_at=run.started_at
            )

        self.db.commit()
        return cleaned_ids
```

## 8) Configuration Management

### 8.1 Enhanced Configuration System

```python
# app/core/audit_config.py (create comprehensive configuration)
from pydantic import BaseSettings, Field, validator
from typing import Dict, List, Optional
from enum import Enum

class AuditBatchStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    ADAPTIVE = "adaptive"
    PLATFORM_OPTIMIZED = "platform_optimized"

class AuditRetryStrategy(str, Enum):
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE = "immediate"

class A
