"""
Memory Consolidation for Organic Intelligence.

Manages learning memory, consolidates experiences, and maintains
long-term knowledge for the intelligent system.
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import threading

from app.utils.logger import get_logger
from app.organism.control.master_switch import get_organic_control, FeatureCategory
from app.organism.control.decorators import register_organic_feature

logger = get_logger(__name__)


class MemoryType(str, Enum):
    """Types of memory stored in the system"""
    EXPERIENCE = "experience"
    PATTERN = "pattern"
    DECISION = "decision"
    INSIGHT = "insight"
    PERFORMANCE = "performance"
    ERROR = "error"
    ADAPTATION = "adaptation"


class MemoryPriority(str, Enum):
    """Priority levels for memory retention"""
    CRITICAL = "critical"      # Never delete
    HIGH = "high"             # Keep for extended periods
    MEDIUM = "medium"         # Standard retention
    LOW = "low"               # Can be cleaned up
    TEMPORARY = "temporary"   # Short-term only


@dataclass
class MemoryItem:
    """Individual memory item"""
    id: str
    memory_type: MemoryType
    priority: MemoryPriority
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    accessed_count: int = 0
    last_accessed: Optional[datetime] = None
    relevance_score: float = 1.0
    tags: Set[str] = field(default_factory=set)
    associations: Set[str] = field(default_factory=set)  # IDs of related memories


@dataclass
class LearningSession:
    """A learning session containing multiple experiences"""
    id: str
    start_time: datetime
    end_time: Optional[datetime]
    experiences: List[str]  # Memory IDs
    insights_generated: List[str]
    performance_delta: float
    context: Dict[str, Any]


@register_organic_feature("memory_consolidation", FeatureCategory.LEARNING)
class MemoryConsolidator:
    """
    Memory consolidation system for organic intelligence.

    Manages short-term and long-term memory, consolidates experiences
    into patterns and insights, and maintains system knowledge.
    """

    def __init__(self):
        if not self.is_organic_enabled():
            return

        # Memory storage
        self._memory_db_path = "organic_memory.db"
        self._memory_cache: Dict[str, MemoryItem] = {}
        self._learning_sessions: Dict[str, LearningSession] = {}

        # Memory limits
        self._max_cache_size = 10000
        self._max_memory_items = 100000

        # Consolidation settings
        self._consolidation_interval = 300  # 5 minutes
        self._last_consolidation = time.time()

        # Threading
        self._consolidation_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.RLock()

        # Metrics
        self._metrics = {
            "total_memories": 0,
            "consolidations_performed": 0,
            "patterns_discovered": 0,
            "memory_cache_hits": 0,
            "memory_cache_misses": 0
        }

        logger.info("Memory Consolidator initialized")

    async def initialize(self):
        """Initialize memory consolidation system"""
        if not self.is_organic_enabled():
            return

        try:
            # Initialize database
            await self._initialize_database()

            # Load recent memories into cache
            await self._load_memory_cache()

            # Start consolidation thread
            self._running = True
            self._consolidation_thread = threading.Thread(
                target=self._consolidation_loop,
                daemon=True,
                name="MemoryConsolidation"
            )
            self._consolidation_thread.start()

            logger.info("Memory Consolidator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Memory Consolidator: {e}")
            raise

    async def _initialize_database(self):
        """Initialize SQLite database for persistent memory storage"""
        try:
            with sqlite3.connect(self._memory_db_path) as conn:
                cursor = conn.cursor()

                # Create memories table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        memory_type TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        accessed_count INTEGER DEFAULT 0,
                        last_accessed TIMESTAMP,
                        relevance_score REAL DEFAULT 1.0,
                        tags TEXT,
                        associations TEXT
                    )
                """)

                # Create learning sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS learning_sessions (
                        id TEXT PRIMARY KEY,
                        start_time TIMESTAMP NOT NULL,
                        end_time TIMESTAMP,
                        experiences TEXT,
                        insights_generated TEXT,
                        performance_delta REAL,
                        context TEXT
                    )
                """)

                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_priority ON memories(priority)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_relevance_score ON memories(relevance_score)")

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to initialize memory database: {e}")
            raise

    async def _load_memory_cache(self):
        """Load recent and high-priority memories into cache"""
        try:
            with sqlite3.connect(self._memory_db_path) as conn:
                cursor = conn.cursor()

                # Load recent high-priority memories
                cursor.execute("""
                    SELECT * FROM memories
                    WHERE priority IN ('critical', 'high')
                    OR created_at > datetime('now', '-1 day')
                    ORDER BY relevance_score DESC, created_at DESC
                    LIMIT ?
                """, (self._max_cache_size,))

                rows = cursor.fetchall()
                for row in rows:
                    memory_item = self._row_to_memory_item(row)
                    self._memory_cache[memory_item.id] = memory_item

                logger.info(f"Loaded {len(self._memory_cache)} memories into cache")

        except Exception as e:
            logger.error(f"Failed to load memory cache: {e}")

    def _row_to_memory_item(self, row: tuple) -> MemoryItem:
        """Convert database row to MemoryItem"""
        return MemoryItem(
            id=row[0],
            memory_type=MemoryType(row[1]),
            priority=MemoryPriority(row[2]),
            content=json.loads(row[3]),
            metadata=json.loads(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            accessed_count=row[6] or 0,
            last_accessed=datetime.fromisoformat(row[7]) if row[7] else None,
            relevance_score=row[8] or 1.0,
            tags=set(json.loads(row[9])) if row[9] else set(),
            associations=set(json.loads(row[10])) if row[10] else set()
        )

    def _consolidation_loop(self):
        """Main consolidation processing loop"""
        logger.info("Memory consolidation loop started")

        while self._running:
            try:
                time.sleep(60)  # Check every minute

                if not self._running:
                    break

                current_time = time.time()
                if current_time - self._last_consolidation >= self._consolidation_interval:
                    asyncio.run(self._perform_consolidation())
                    self._last_consolidation = current_time

            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}")
                time.sleep(10)

        logger.info("Memory consolidation loop stopped")

    async def _perform_consolidation(self):
        """Perform memory consolidation"""
        try:
            consolidation_start = time.time()

            # 1. Identify patterns in recent memories
            patterns = await self._identify_patterns()

            # 2. Consolidate related experiences
            consolidated_experiences = await self._consolidate_experiences()

            # 3. Update memory relevance scores
            await self._update_relevance_scores()

            # 4. Clean up old low-priority memories
            cleaned_count = await self._cleanup_old_memories()

            # 5. Update cache
            await self._refresh_memory_cache()

            consolidation_time = time.time() - consolidation_start
            self._metrics["consolidations_performed"] += 1
            self._metrics["patterns_discovered"] += len(patterns)

            logger.debug(
                f"Memory consolidation completed in {consolidation_time:.2f}s: "
                f"{len(patterns)} patterns, {len(consolidated_experiences)} experiences, "
                f"{cleaned_count} memories cleaned"
            )

        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")

    async def _identify_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns in recent memories"""
        try:
            patterns = []

            # Get recent memories by type
            memory_groups = {}
            for memory in self._memory_cache.values():
                if memory.memory_type not in memory_groups:
                    memory_groups[memory.memory_type] = []
                memory_groups[memory.memory_type].append(memory)

            # Look for patterns within each memory type
            for memory_type, memories in memory_groups.items():
                if len(memories) < 3:  # Need at least 3 items to identify patterns
                    continue

                type_patterns = await self._analyze_memory_group_patterns(memory_type, memories)
                patterns.extend(type_patterns)

            return patterns

        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")
            return []

    async def _analyze_memory_group_patterns(
        self,
        memory_type: MemoryType,
        memories: List[MemoryItem]
    ) -> List[Dict[str, Any]]:
        """Analyze patterns within a group of memories of the same type"""
        try:
            patterns = []

            if memory_type == MemoryType.PERFORMANCE:
                # Look for performance trends
                performance_data = [
                    (m.created_at, m.content.get("metric_value", 0))
                    for m in sorted(memories, key=lambda x: x.created_at)
                ]

                if len(performance_data) >= 5:
                    # Simple trend analysis
                    values = [d[1] for d in performance_data]
                    if self._is_trending_up(values):
                        patterns.append({
                            "type": "performance_improvement",
                            "confidence": 0.8,
                            "evidence": performance_data[-5:],
                            "description": "Performance metrics showing upward trend"
                        })
                    elif self._is_trending_down(values):
                        patterns.append({
                            "type": "performance_degradation",
                            "confidence": 0.8,
                            "evidence": performance_data[-5:],
                            "description": "Performance metrics showing downward trend"
                        })

            elif memory_type == MemoryType.ERROR:
                # Look for error patterns
                error_types = {}
                for memory in memories:
                    error_type = memory.content.get("error_type", "unknown")
                    if error_type not in error_types:
                        error_types[error_type] = []
                    error_types[error_type].append(memory)

                for error_type, error_memories in error_types.items():
                    if len(error_memories) >= 3:
                        patterns.append({
                            "type": "recurring_error",
                            "confidence": 0.9,
                            "evidence": [m.id for m in error_memories],
                            "description": f"Recurring {error_type} errors detected"
                        })

            elif memory_type == MemoryType.DECISION:
                # Look for decision outcome patterns
                successful_decisions = [m for m in memories if m.content.get("success", False)]
                if len(successful_decisions) > len(memories) * 0.8:
                    patterns.append({
                        "type": "high_decision_success_rate",
                        "confidence": 0.7,
                        "evidence": [m.id for m in successful_decisions],
                        "description": "High decision success rate observed"
                    })

            return patterns

        except Exception as e:
            logger.error(f"Error analyzing memory group patterns: {e}")
            return []

    def _is_trending_up(self, values: List[float], threshold: float = 0.1) -> bool:
        """Check if values show an upward trend"""
        if len(values) < 3:
            return False

        recent_avg = sum(values[-3:]) / 3
        earlier_avg = sum(values[:-3]) / max(1, len(values) - 3)

        return recent_avg > earlier_avg * (1 + threshold)

    def _is_trending_down(self, values: List[float], threshold: float = 0.1) -> bool:
        """Check if values show a downward trend"""
        if len(values) < 3:
            return False

        recent_avg = sum(values[-3:]) / 3
        earlier_avg = sum(values[:-3]) / max(1, len(values) - 3)

        return recent_avg < earlier_avg * (1 - threshold)

    async def _consolidate_experiences(self) -> List[str]:
        """Consolidate related experiences into higher-level insights"""
        try:
            consolidated = []

            # Group experiences by context similarity
            experience_groups = await self._group_similar_experiences()

            for group in experience_groups:
                if len(group) >= 3:  # Need multiple experiences for consolidation
                    consolidated_insight = await self._create_consolidated_insight(group)
                    if consolidated_insight:
                        consolidated.append(consolidated_insight)

            return consolidated

        except Exception as e:
            logger.error(f"Error consolidating experiences: {e}")
            return []

    async def _group_similar_experiences(self) -> List[List[MemoryItem]]:
        """Group similar experiences together"""
        try:
            experience_memories = [
                m for m in self._memory_cache.values()
                if m.memory_type == MemoryType.EXPERIENCE
            ]

            # Simple grouping by shared tags
            groups = []
            used_memories = set()

            for memory in experience_memories:
                if memory.id in used_memories:
                    continue

                # Find memories with similar tags
                similar_memories = [memory]
                used_memories.add(memory.id)

                for other_memory in experience_memories:
                    if (other_memory.id not in used_memories and
                        len(memory.tags.intersection(other_memory.tags)) >= 2):
                        similar_memories.append(other_memory)
                        used_memories.add(other_memory.id)

                if len(similar_memories) >= 2:
                    groups.append(similar_memories)

            return groups

        except Exception as e:
            logger.error(f"Error grouping similar experiences: {e}")
            return []

    async def _create_consolidated_insight(self, experiences: List[MemoryItem]) -> Optional[str]:
        """Create a consolidated insight from multiple experiences"""
        try:
            # Generate insight from experiences
            common_tags = set.intersection(*[exp.tags for exp in experiences])
            shared_context = {}

            # Find common context elements
            for exp in experiences:
                for key, value in exp.content.items():
                    if key in shared_context:
                        if shared_context[key] != value:
                            shared_context[key] = "varied"
                    else:
                        shared_context[key] = value

            # Create insight memory
            insight_id = f"insight_{int(time.time())}"
            insight_content = {
                "type": "consolidated_insight",
                "source_experiences": [exp.id for exp in experiences],
                "common_patterns": list(common_tags),
                "shared_context": shared_context,
                "confidence": min(1.0, len(common_tags) / 5.0)
            }

            await self.store_memory(
                memory_id=insight_id,
                memory_type=MemoryType.INSIGHT,
                priority=MemoryPriority.HIGH,
                content=insight_content,
                metadata={"consolidation_source": True},
                tags=common_tags
            )

            return insight_id

        except Exception as e:
            logger.error(f"Error creating consolidated insight: {e}")
            return None

    async def _update_relevance_scores(self):
        """Update relevance scores for memories based on usage and age"""
        try:
            current_time = time.time()

            for memory in self._memory_cache.values():
                age_days = (current_time - memory.created_at.timestamp()) / 86400

                # Base score decay over time
                age_factor = max(0.1, 1.0 - (age_days / 365))  # Decay over a year

                # Usage factor
                usage_factor = min(2.0, 1.0 + (memory.accessed_count / 100))

                # Priority factor
                priority_factors = {
                    MemoryPriority.CRITICAL: 2.0,
                    MemoryPriority.HIGH: 1.5,
                    MemoryPriority.MEDIUM: 1.0,
                    MemoryPriority.LOW: 0.7,
                    MemoryPriority.TEMPORARY: 0.3
                }
                priority_factor = priority_factors.get(memory.priority, 1.0)

                # Calculate new relevance score
                new_score = age_factor * usage_factor * priority_factor
                memory.relevance_score = min(2.0, max(0.1, new_score))

            # Update database
            await self._update_relevance_scores_in_db()

        except Exception as e:
            logger.error(f"Error updating relevance scores: {e}")

    async def _update_relevance_scores_in_db(self):
        """Update relevance scores in database"""
        try:
            with sqlite3.connect(self._memory_db_path) as conn:
                cursor = conn.cursor()

                for memory in self._memory_cache.values():
                    cursor.execute(
                        "UPDATE memories SET relevance_score = ? WHERE id = ?",
                        (memory.relevance_score, memory.id)
                    )

                conn.commit()

        except Exception as e:
            logger.error(f"Error updating relevance scores in database: {e}")

    async def _cleanup_old_memories(self) -> int:
        """Clean up old, low-relevance memories"""
        try:
            cleaned_count = 0

            # Get memory count
            with sqlite3.connect(self._memory_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memories")
                total_memories = cursor.fetchone()[0]

                if total_memories <= self._max_memory_items:
                    return 0

                # Calculate how many to remove
                to_remove = total_memories - self._max_memory_items

                # Remove lowest relevance, oldest memories (except critical)
                cursor.execute("""
                    DELETE FROM memories
                    WHERE priority != 'critical'
                    AND id IN (
                        SELECT id FROM memories
                        WHERE priority != 'critical'
                        ORDER BY relevance_score ASC, created_at ASC
                        LIMIT ?
                    )
                """, (to_remove,))

                cleaned_count = cursor.rowcount
                conn.commit()

            # Remove from cache as well
            memory_ids_to_remove = []
            for memory_id, memory in self._memory_cache.items():
                if memory.priority != MemoryPriority.CRITICAL and memory.relevance_score < 0.3:
                    memory_ids_to_remove.append(memory_id)

            for memory_id in memory_ids_to_remove:
                del self._memory_cache[memory_id]

            return cleaned_count

        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")
            return 0

    async def _refresh_memory_cache(self):
        """Refresh memory cache with most relevant items"""
        try:
            # Clear current cache
            self._memory_cache.clear()

            # Reload from database
            await self._load_memory_cache()

        except Exception as e:
            logger.error(f"Error refreshing memory cache: {e}")

    async def store_memory(
        self,
        memory_id: str,
        memory_type: MemoryType,
        priority: MemoryPriority,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        associations: Optional[Set[str]] = None
    ) -> bool:
        """
        Store a new memory item.

        Args:
            memory_id: Unique identifier for the memory
            memory_type: Type of memory
            priority: Priority level
            content: Memory content
            metadata: Optional metadata
            tags: Optional tags for categorization
            associations: Optional associated memory IDs

        Returns:
            True if storage successful
        """
        try:
            memory_item = MemoryItem(
                id=memory_id,
                memory_type=memory_type,
                priority=priority,
                content=content,
                metadata=metadata or {},
                created_at=datetime.now(timezone.utc),
                tags=tags or set(),
                associations=associations or set()
            )

            # Store in cache
            with self._lock:
                self._memory_cache[memory_id] = memory_item

                # Limit cache size
                if len(self._memory_cache) > self._max_cache_size:
                    # Remove least relevant items
                    items_to_remove = sorted(
                        self._memory_cache.items(),
                        key=lambda x: (x[1].relevance_score, x[1].created_at)
                    )[:len(self._memory_cache) - self._max_cache_size + 1]

                    for item_id, _ in items_to_remove:
                        del self._memory_cache[item_id]

            # Store in database
            await self._store_memory_in_db(memory_item)

            self._metrics["total_memories"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to store memory {memory_id}: {e}")
            return False

    async def _store_memory_in_db(self, memory: MemoryItem):
        """Store memory item in database"""
        try:
            with sqlite3.connect(self._memory_db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO memories
                    (id, memory_type, priority, content, metadata, created_at,
                     accessed_count, last_accessed, relevance_score, tags, associations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id,
                    memory.memory_type.value,
                    memory.priority.value,
                    json.dumps(memory.content),
                    json.dumps(memory.metadata),
                    memory.created_at.isoformat(),
                    memory.accessed_count,
                    memory.last_accessed.isoformat() if memory.last_accessed else None,
                    memory.relevance_score,
                    json.dumps(list(memory.tags)),
                    json.dumps(list(memory.associations))
                ))

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to store memory in database: {e}")
            raise

    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a specific memory item"""
        try:
            # Check cache first
            if memory_id in self._memory_cache:
                memory = self._memory_cache[memory_id]
                memory.accessed_count += 1
                memory.last_accessed = datetime.now(timezone.utc)
                self._metrics["memory_cache_hits"] += 1
                return memory

            # Check database
            with sqlite3.connect(self._memory_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
                row = cursor.fetchone()

                if row:
                    memory = self._row_to_memory_item(row)
                    memory.accessed_count += 1
                    memory.last_accessed = datetime.now(timezone.utc)

                    # Update access count in database
                    cursor.execute(
                        "UPDATE memories SET accessed_count = ?, last_accessed = ? WHERE id = ?",
                        (memory.accessed_count, memory.last_accessed.isoformat(), memory_id)
                    )
                    conn.commit()

                    # Add to cache
                    self._memory_cache[memory_id] = memory
                    self._metrics["memory_cache_misses"] += 1
                    return memory

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None

    async def search_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[Set[str]] = None,
        content_keywords: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[MemoryItem]:
        """Search for memories based on criteria"""
        try:
            conditions = []
            params = []

            if memory_type:
                conditions.append("memory_type = ?")
                params.append(memory_type.value)

            query = "SELECT * FROM memories"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY relevance_score DESC, created_at DESC LIMIT ?"
            params.append(limit)

            with sqlite3.connect(self._memory_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()

                memories = [self._row_to_memory_item(row) for row in rows]

                # Filter by tags if specified
                if tags:
                    memories = [m for m in memories if tags.intersection(m.tags)]

                # Filter by content keywords if specified
                if content_keywords:
                    filtered_memories = []
                    for memory in memories:
                        content_str = json.dumps(memory.content).lower()
                        if any(keyword.lower() in content_str for keyword in content_keywords):
                            filtered_memories.append(memory)
                    memories = filtered_memories

                return memories

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []

    async def consolidate_recent_experiences(self):
        """Consolidate recent experiences (called by central intelligence)"""
        try:
            if time.time() - self._last_consolidation >= self._consolidation_interval:
                await self._perform_consolidation()

        except Exception as e:
            logger.error(f"Error in recent experience consolidation: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            stats = self._metrics.copy()

            # Add current cache stats
            stats["cache_size"] = len(self._memory_cache)
            stats["cache_hit_rate"] = (
                self._metrics["memory_cache_hits"] /
                max(1, self._metrics["memory_cache_hits"] + self._metrics["memory_cache_misses"])
            )

            # Memory type distribution
            type_distribution = {}
            for memory in self._memory_cache.values():
                type_name = memory.memory_type.value
                type_distribution[type_name] = type_distribution.get(type_name, 0) + 1

            stats["memory_type_distribution"] = type_distribution

            # Priority distribution
            priority_distribution = {}
            for memory in self._memory_cache.values():
                priority_name = memory.priority.value
                priority_distribution[priority_name] = priority_distribution.get(priority_name, 0) + 1

            stats["priority_distribution"] = priority_distribution

            return stats

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}

    async def shutdown(self):
        """Shutdown memory consolidation system"""
        try:
            logger.info("Shutting down Memory Consolidator...")

            # Stop consolidation thread
            self._running = False
            if self._consolidation_thread and self._consolidation_thread.is_alive():
                self._consolidation_thread.join(timeout=5.0)

            # Perform final consolidation
            await self._perform_consolidation()

            logger.info("Memory Consolidator shutdown complete")

        except Exception as e:
            logger.error(f"Error during Memory Consolidator shutdown: {e}")