"""
Pattern Recognition Engine for Organic Intelligence.

Identifies patterns across all system components including performance patterns,
user behavior patterns, error patterns, and system health patterns.
"""

import asyncio
import time
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from app.utils.logger import get_logger
from app.organism.control.decorators import register_organic_feature
from app.organism.control.master_switch import FeatureCategory

logger = get_logger(__name__)


class PatternType(str, Enum):
    """Types of patterns that can be recognized"""
    PERFORMANCE = "performance"
    ERROR = "error"
    USER_BEHAVIOR = "user_behavior"
    SYSTEM_HEALTH = "system_health"
    USAGE = "usage"
    TEMPORAL = "temporal"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"


class PatternConfidence(str, Enum):
    """Confidence levels for pattern recognition"""
    LOW = "low"          # < 0.3
    MEDIUM = "medium"    # 0.3 - 0.7
    HIGH = "high"        # 0.7 - 0.9
    VERY_HIGH = "very_high"  # > 0.9


@dataclass
class DataPoint:
    """Single data point for pattern analysis"""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    component: str = "unknown"
    category: str = "general"


@dataclass
class Pattern:
    """Recognized pattern"""
    id: str
    name: str
    type: PatternType
    confidence: float
    description: str
    discovered_at: datetime
    data_points: List[DataPoint]
    characteristics: Dict[str, Any]
    prediction: Optional[Dict[str, Any]] = None
    suggested_actions: List[str] = field(default_factory=list)


@dataclass
class PatternTrend:
    """Trend analysis for a pattern"""
    direction: str  # "increasing", "decreasing", "stable", "cyclical"
    magnitude: float
    velocity: float
    confidence: float
    forecast: Optional[List[float]] = None


@register_organic_feature("pattern_recognition", FeatureCategory.LEARNING)
class PatternRecognizer:
    """
    Advanced pattern recognition engine for organic intelligence.

    Identifies various types of patterns across system components and
    provides insights for system improvement and adaptation.
    """

    def __init__(self):
        if not self.is_organic_enabled():
            return

        # Data storage
        self._data_streams: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._recognized_patterns: Dict[str, Pattern] = {}
        self._pattern_history: List[Pattern] = []

        # Configuration
        self._min_data_points = 10
        self._confidence_threshold = 0.6
        self._max_pattern_age = timedelta(hours=24)

        # Pattern detection algorithms
        self._detectors = {
            PatternType.PERFORMANCE: self._detect_performance_patterns,
            PatternType.ERROR: self._detect_error_patterns,
            PatternType.USER_BEHAVIOR: self._detect_user_behavior_patterns,
            PatternType.SYSTEM_HEALTH: self._detect_system_health_patterns,
            PatternType.USAGE: self._detect_usage_patterns,
            PatternType.TEMPORAL: self._detect_temporal_patterns,
            PatternType.ANOMALY: self._detect_anomaly_patterns,
            PatternType.CORRELATION: self._detect_correlation_patterns,
        }

        # Statistics
        self._stats = {
            "patterns_detected": 0,
            "patterns_validated": 0,
            "predictions_made": 0,
            "accuracy_score": 0.0
        }

        logger.info("Pattern Recognition Engine initialized")

    async def initialize(self):
        """Initialize pattern recognition engine"""
        if not self.is_organic_enabled():
            return

        logger.info("Initializing Pattern Recognition Engine")
        # Start background pattern detection
        asyncio.create_task(self._pattern_detection_loop())

    def add_data_point(self, stream_name: str, value: float, metadata: Optional[Dict] = None):
        """
        Add a data point to a stream for pattern analysis.

        Args:
            stream_name: Name of the data stream
            value: Numeric value
            metadata: Additional metadata about the data point
        """
        if not self.is_organic_enabled():
            return

        data_point = DataPoint(
            timestamp=time.time(),
            value=value,
            metadata=metadata or {},
            component=metadata.get("component", "unknown") if metadata else "unknown",
            category=metadata.get("category", "general") if metadata else "general"
        )

        self._data_streams[stream_name].append(data_point)

        # Trigger immediate pattern analysis for this stream if enough data
        if len(self._data_streams[stream_name]) >= self._min_data_points:
            asyncio.create_task(self._analyze_stream(stream_name))

    async def _pattern_detection_loop(self):
        """Background loop for continuous pattern detection"""
        while self.is_organic_enabled():
            try:
                await self._run_pattern_detection_cycle()
                await asyncio.sleep(30)  # Run every 30 seconds
            except Exception as e:
                logger.error(f"Error in pattern detection loop: {e}")
                await asyncio.sleep(60)

    async def _run_pattern_detection_cycle(self):
        """Run a single pattern detection cycle"""
        # Clean up old patterns
        await self._cleanup_old_patterns()

        # Analyze all data streams
        for stream_name in list(self._data_streams.keys()):
            if len(self._data_streams[stream_name]) >= self._min_data_points:
                await self._analyze_stream(stream_name)

        # Look for cross-stream correlations
        await self._analyze_correlations()

        # Update pattern trends
        await self._update_pattern_trends()

    async def _analyze_stream(self, stream_name: str):
        """Analyze a specific data stream for patterns"""
        try:
            data_points = list(self._data_streams[stream_name])

            for pattern_type, detector in self._detectors.items():
                patterns = await detector(stream_name, data_points)
                for pattern in patterns:
                    await self._register_pattern(pattern)

        except Exception as e:
            logger.error(f"Error analyzing stream {stream_name}: {e}")

    async def _detect_performance_patterns(self, stream_name: str, data_points: List[DataPoint]) -> List[Pattern]:
        """Detect performance-related patterns"""
        patterns = []

        if "performance" not in stream_name.lower() and "latency" not in stream_name.lower():
            return patterns

        try:
            values = [dp.value for dp in data_points]
            timestamps = [dp.timestamp for dp in data_points]

            # Detect performance degradation
            if len(values) >= 20:
                recent_avg = statistics.mean(values[-10:])
                historical_avg = statistics.mean(values[:-10])

                if recent_avg > historical_avg * 1.5:  # 50% degradation
                    pattern = Pattern(
                        id=f"perf_degradation_{stream_name}_{int(time.time())}",
                        name=f"Performance Degradation in {stream_name}",
                        type=PatternType.PERFORMANCE,
                        confidence=min(0.9, (recent_avg / historical_avg - 1.0)),
                        description=f"Performance degraded by {((recent_avg / historical_avg - 1.0) * 100):.1f}%",
                        discovered_at=datetime.now(timezone.utc),
                        data_points=data_points[-20:],
                        characteristics={
                            "degradation_factor": recent_avg / historical_avg,
                            "recent_avg": recent_avg,
                            "historical_avg": historical_avg
                        },
                        suggested_actions=[
                            "Investigate recent changes",
                            "Check resource utilization",
                            "Review error logs"
                        ]
                    )
                    patterns.append(pattern)

            # Detect performance spikes
            if len(values) >= 10:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0

                spikes = [v for v in values[-5:] if v > mean_val + 2 * std_val]
                if len(spikes) >= 2:
                    pattern = Pattern(
                        id=f"perf_spikes_{stream_name}_{int(time.time())}",
                        name=f"Performance Spikes in {stream_name}",
                        type=PatternType.PERFORMANCE,
                        confidence=0.8,
                        description=f"Detected {len(spikes)} performance spikes",
                        discovered_at=datetime.now(timezone.utc),
                        data_points=data_points[-10:],
                        characteristics={
                            "spike_count": len(spikes),
                            "spike_magnitude": max(spikes) / mean_val if mean_val > 0 else 0
                        },
                        suggested_actions=[
                            "Identify spike triggers",
                            "Implement load balancing",
                            "Scale resources"
                        ]
                    )
                    patterns.append(pattern)

        except Exception as e:
            logger.error(f"Error in performance pattern detection: {e}")

        return patterns

    async def _detect_error_patterns(self, stream_name: str, data_points: List[DataPoint]) -> List[Pattern]:
        """Detect error-related patterns"""
        patterns = []

        if "error" not in stream_name.lower() and "failure" not in stream_name.lower():
            return patterns

        try:
            # Group errors by time windows
            time_windows = defaultdict(int)
            window_size = 300  # 5 minutes

            for dp in data_points:
                window = int(dp.timestamp // window_size) * window_size
                time_windows[window] += dp.value

            # Detect error bursts
            values = list(time_windows.values())
            if len(values) >= 5:
                mean_errors = statistics.mean(values)
                std_errors = statistics.stdev(values) if len(values) > 1 else 0

                burst_threshold = mean_errors + 2 * std_errors
                bursts = [v for v in values[-3:] if v > burst_threshold]

                if len(bursts) >= 2:
                    pattern = Pattern(
                        id=f"error_burst_{stream_name}_{int(time.time())}",
                        name=f"Error Burst in {stream_name}",
                        type=PatternType.ERROR,
                        confidence=0.9,
                        description=f"Error burst detected: {len(bursts)} windows above threshold",
                        discovered_at=datetime.now(timezone.utc),
                        data_points=data_points[-20:],
                        characteristics={
                            "burst_count": len(bursts),
                            "threshold": burst_threshold,
                            "max_errors": max(bursts)
                        },
                        suggested_actions=[
                            "Investigate error causes",
                            "Implement circuit breakers",
                            "Review error handling"
                        ]
                    )
                    patterns.append(pattern)

        except Exception as e:
            logger.error(f"Error in error pattern detection: {e}")

        return patterns

    async def _detect_user_behavior_patterns(self, stream_name: str, data_points: List[DataPoint]) -> List[Pattern]:
        """Detect user behavior patterns"""
        patterns = []

        if "user" not in stream_name.lower() and "request" not in stream_name.lower():
            return patterns

        # Implement user behavior pattern detection
        # This would analyze request patterns, usage patterns, etc.

        return patterns

    async def _detect_system_health_patterns(self, stream_name: str, data_points: List[DataPoint]) -> List[Pattern]:
        """Detect system health patterns"""
        patterns = []

        if "health" not in stream_name.lower() and "resource" not in stream_name.lower():
            return patterns

        # Implement system health pattern detection
        # This would analyze CPU, memory, disk usage patterns

        return patterns

    async def _detect_usage_patterns(self, stream_name: str, data_points: List[DataPoint]) -> List[Pattern]:
        """Detect usage patterns"""
        patterns = []

        # Implement usage pattern detection
        # This would analyze feature usage, API call patterns, etc.

        return patterns

    async def _detect_temporal_patterns(self, stream_name: str, data_points: List[DataPoint]) -> List[Pattern]:
        """Detect time-based patterns"""
        patterns = []

        try:
            # Look for cyclical patterns (hourly, daily, weekly)
            timestamps = [dp.timestamp for dp in data_points]
            values = [dp.value for dp in data_points]

            if len(data_points) >= 50:
                # Check for hourly patterns
                hourly_values = defaultdict(list)
                for dp in data_points:
                    hour = datetime.fromtimestamp(dp.timestamp).hour
                    hourly_values[hour].append(dp.value)

                # Calculate coefficient of variation for each hour
                hour_variations = {}
                for hour, vals in hourly_values.items():
                    if len(vals) >= 3:
                        mean_val = statistics.mean(vals)
                        std_val = statistics.stdev(vals)
                        if mean_val > 0:
                            hour_variations[hour] = std_val / mean_val

                # Check if there's a clear pattern
                if len(hour_variations) >= 10:
                    pattern_strength = 1.0 - statistics.mean(hour_variations.values())
                    if pattern_strength > 0.7:
                        pattern = Pattern(
                            id=f"hourly_pattern_{stream_name}_{int(time.time())}",
                            name=f"Hourly Pattern in {stream_name}",
                            type=PatternType.TEMPORAL,
                            confidence=pattern_strength,
                            description=f"Strong hourly pattern detected (strength: {pattern_strength:.2f})",
                            discovered_at=datetime.now(timezone.utc),
                            data_points=data_points[-50:],
                            characteristics={
                                "pattern_type": "hourly",
                                "pattern_strength": pattern_strength,
                                "hour_variations": hour_variations
                            },
                            suggested_actions=[
                                "Optimize for peak hours",
                                "Implement predictive scaling",
                                "Schedule maintenance during low-usage hours"
                            ]
                        )
                        patterns.append(pattern)

        except Exception as e:
            logger.error(f"Error in temporal pattern detection: {e}")

        return patterns

    async def _detect_anomaly_patterns(self, stream_name: str, data_points: List[DataPoint]) -> List[Pattern]:
        """Detect anomalous patterns"""
        patterns = []

        try:
            if len(data_points) >= 20:
                values = [dp.value for dp in data_points]
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0

                # Detect outliers
                outliers = []
                for i, dp in enumerate(data_points[-10:]):
                    if abs(dp.value - mean_val) > 3 * std_val:
                        outliers.append((i, dp))

                if len(outliers) >= 2:
                    pattern = Pattern(
                        id=f"anomaly_{stream_name}_{int(time.time())}",
                        name=f"Anomalies in {stream_name}",
                        type=PatternType.ANOMALY,
                        confidence=0.8,
                        description=f"Detected {len(outliers)} anomalous values",
                        discovered_at=datetime.now(timezone.utc),
                        data_points=[outlier[1] for outlier in outliers],
                        characteristics={
                            "outlier_count": len(outliers),
                            "threshold": 3 * std_val,
                            "mean_value": mean_val
                        },
                        suggested_actions=[
                            "Investigate anomaly causes",
                            "Implement anomaly detection",
                            "Review data quality"
                        ]
                    )
                    patterns.append(pattern)

        except Exception as e:
            logger.error(f"Error in anomaly pattern detection: {e}")

        return patterns

    async def _detect_correlation_patterns(self) -> List[Pattern]:
        """Detect correlation patterns between different streams"""
        patterns = []

        try:
            # This would implement cross-stream correlation analysis
            # For now, return empty list
            pass

        except Exception as e:
            logger.error(f"Error in correlation pattern detection: {e}")

        return patterns

    async def _analyze_correlations(self):
        """Analyze correlations between different data streams"""
        # Implement correlation analysis between streams
        pass

    async def _register_pattern(self, pattern: Pattern):
        """Register a newly detected pattern"""
        try:
            # Check if this pattern already exists
            existing_pattern = None
            for existing in self._recognized_patterns.values():
                if (existing.name == pattern.name and
                    existing.type == pattern.type and
                    (datetime.now(timezone.utc) - existing.discovered_at).total_seconds() < 3600):
                    existing_pattern = existing
                    break

            if existing_pattern:
                # Update existing pattern
                existing_pattern.confidence = max(existing_pattern.confidence, pattern.confidence)
                existing_pattern.data_points.extend(pattern.data_points)
                # Keep only recent data points
                existing_pattern.data_points = existing_pattern.data_points[-50:]
            else:
                # Add new pattern
                self._recognized_patterns[pattern.id] = pattern
                self._pattern_history.append(pattern)
                self._stats["patterns_detected"] += 1

                logger.info(f"New pattern detected: {pattern.name} (confidence: {pattern.confidence:.2f})")

        except Exception as e:
            logger.error(f"Error registering pattern: {e}")

    async def _cleanup_old_patterns(self):
        """Clean up old patterns"""
        try:
            current_time = datetime.now(timezone.utc)
            old_patterns = []

            for pattern_id, pattern in self._recognized_patterns.items():
                if current_time - pattern.discovered_at > self._max_pattern_age:
                    old_patterns.append(pattern_id)

            for pattern_id in old_patterns:
                del self._recognized_patterns[pattern_id]

        except Exception as e:
            logger.error(f"Error cleaning up old patterns: {e}")

    async def _update_pattern_trends(self):
        """Update trends for existing patterns"""
        # Implement trend analysis for patterns
        pass

    async def analyze_current_patterns(self) -> List[Pattern]:
        """Get current recognized patterns"""
        return list(self._recognized_patterns.values())

    def get_pattern_by_id(self, pattern_id: str) -> Optional[Pattern]:
        """Get a specific pattern by ID"""
        return self._recognized_patterns.get(pattern_id)

    def get_patterns_by_type(self, pattern_type: PatternType) -> List[Pattern]:
        """Get patterns of a specific type"""
        return [p for p in self._recognized_patterns.values() if p.type == pattern_type]

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get pattern recognition statistics"""
        return {
            **self._stats,
            "active_patterns": len(self._recognized_patterns),
            "data_streams": len(self._data_streams),
            "total_data_points": sum(len(stream) for stream in self._data_streams.values())
        }

    async def shutdown(self):
        """Shutdown pattern recognition engine"""
        logger.info("Shutting down Pattern Recognition Engine")
        # Clean up resources
        self._data_streams.clear()
        self._recognized_patterns.clear()