# Brand Detection Engine - Complete Build Script

This single markdown file contains everything needed to build the brand detection engine from scratch. Execute each code block in order.

## Step 1: Environment Setup and Dependencies

```bash
# Create project structure
mkdir -p app/services/brand_detection/core
mkdir -p app/services/brand_detection/market_adapters
mkdir -p app/services/brand_detection/models
mkdir -p app/services/brand_detection/utils
mkdir -p app/services/brand_detection/tests/unit
mkdir -p app/services/brand_detection/tests/integration
mkdir -p app/services/brand_detection/tests/performance

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core dependencies
fastapi==0.104.1
uvicorn==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.12.1
celery==5.3.4
redis==5.0.1
pydantic==2.5.0
python-dotenv==1.0.0

# AI and NLP
openai==1.3.8
anthropic==0.7.7
spacy==3.7.2
transformers==4.35.2
sentence-transformers==2.2.2
torch==2.1.0
vaderSentiment==3.3.2

# Text processing
rapidfuzz==3.5.2
nltk==3.8.1

# Data processing
pandas==2.1.3
numpy==1.25.2

# Utilities
structlog==23.2.0
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-benchmark==4.0.0
requests==2.31.0
EOF

# Install dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

echo "✅ Environment setup complete"
```

## Step 2: Core Data Models

```python
# Create: app/services/brand_detection/models/__init__.py
cat > app/services/brand_detection/models/__init__.py << 'EOF'
"""Brand detection data models"""
EOF
```

```python
# Create: app/services/brand_detection/models/brand_mention.py
cat > app/services/brand_detection/models/brand_mention.py << 'EOF'
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
import uuid
from datetime import datetime

class SentimentPolarity(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class DetectionMethod(Enum):
    NER = "named_entity_recognition"
    FUZZY = "fuzzy_matching"
    SEMANTIC = "semantic_similarity"
    REGEX = "regex_pattern"
    HYBRID = "hybrid_approach"

@dataclass
class BrandContext:
    """Context surrounding a brand mention"""
    text: str
    start_position: int
    end_position: int
    sentence: str
    surrounding_entities: List[str] = field(default_factory=list)
    competitive_mentions: List[str] = field(default_factory=list)

class BrandMention:
    """Enhanced brand mention with comprehensive metadata"""

    def __init__(self,
                 brand: str,
                 original_text: str,
                 confidence: float,
                 detection_method: DetectionMethod,
                 language: str = "en"):
        self.id = str(uuid.uuid4())
        self.brand = brand
        self.original_text = original_text
        self.confidence = confidence
        self.detection_method = detection_method
        self.language = language
        self.created_at = datetime.utcnow()

        # Analytics data
        self.contexts: List[BrandContext] = []
        self.sentiment_score: float = 0.0
        self.sentiment_polarity: SentimentPolarity = SentimentPolarity.NEUTRAL
        self.mention_count: int = 0
        self.competitive_context: Dict[str, int] = {}
        self.market_specific_data: Dict[str, any] = {}

    def add_context(self, context: BrandContext):
        """Add context with validation"""
        if not isinstance(context, BrandContext):
            raise ValueError("Context must be BrandContext instance")
        self.contexts.append(context)
        self.mention_count = len(self.contexts)

    def calculate_relevance_score(self) -> float:
        """Calculate overall relevance based on multiple factors"""
        base_score = self.confidence

        # Boost for multiple mentions
        mention_boost = min(0.2, len(self.contexts) * 0.05)

        # Boost for competitive context
        competitive_boost = 0.1 if self.competitive_context else 0.0

        # Sentiment influence
        sentiment_influence = abs(self.sentiment_score) * 0.1

        return min(1.0, base_score + mention_boost + competitive_boost + sentiment_influence)

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "brand": self.brand,
            "mention_count": self.mention_count,
            "confidence": self.confidence,
            "sentiment_score": self.sentiment_score,
            "sentiment_polarity": self.sentiment_polarity.value,
            "detection_method": self.detection_method.value,
            "language": self.language,
            "relevance_score": self.calculate_relevance_score(),
            "contexts": [
                {
                    "text": ctx.text,
                    "sentence": ctx.sentence,
                    "competitive_mentions": ctx.competitive_mentions
                }
                for ctx in self.contexts[:3]  # Limit for API response size
            ],
            "competitive_context": self.competitive_context,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class DetectionResult:
    """Result container for brand detection operations"""
    text_analyzed: str
    language: str
    processing_time_ms: float
    mentions: List[BrandMention]
    total_brands_found: int
    confidence_threshold: float
    market_adapter_used: str

    def __post_init__(self):
        self.total_brands_found = len(self.mentions)

    def get_top_mentions(self, limit: int = 10) -> List[BrandMention]:
        """Get top mentions by relevance score"""
        return sorted(
            self.mentions,
            key=lambda x: x.calculate_relevance_score(),
            reverse=True
        )[:limit]

    def get_mentions_by_brand(self, brand_name: str) -> List[BrandMention]:
        """Get all mentions for specific brand"""
        return [m for m in self.mentions if m.brand.lower() == brand_name.lower()]

    def to_summary_dict(self) -> Dict:
        """Summary for logging and monitoring"""
        return {
            "total_mentions": self.total_brands_found,
            "processing_time_ms": self.processing_time_ms,
            "language": self.language,
            "market_adapter": self.market_adapter_used,
            "top_brands": [m.brand for m in self.get_top_mentions(5)],
            "avg_confidence": sum(m.confidence for m in self.mentions) / len(self.mentions) if self.mentions else 0
        }
EOF
```

## Step 3: Market Adapters Base

```python
# Create: app/services/brand_detection/market_adapters/__init__.py
cat > app/services/brand_detection/market_adapters/__init__.py << 'EOF'
"""Market-specific brand detection adapters"""
EOF
```

```python
# Create: app/services/brand_detection/market_adapters/base.py
cat > app/services/brand_detection/market_adapters/base.py << 'EOF'
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Optional, Tuple
import re
from ..models.brand_mention import BrandMention, DetectionMethod

class BaseMarketAdapter(ABC):
    """Abstract base class for market-specific brand detection logic"""

    def __init__(self, market_code: str, language_code: str):
        self.market_code = market_code  # 'DE', 'US'
        self.language_code = language_code  # 'de', 'en'
        self.company_suffixes = self._get_company_suffixes()
        self.business_keywords = self._get_business_keywords()
        self.industry_patterns = self._get_industry_patterns()

        # Performance tracking
        self._cache_hits = 0
        self._cache_misses = 0

    @abstractmethod
    def _get_company_suffixes(self) -> List[str]:
        """Get market-specific company suffixes"""
        pass

    @abstractmethod
    def _get_business_keywords(self) -> List[str]:
        """Get market-specific business context keywords"""
        pass

    @abstractmethod
    def _get_industry_patterns(self) -> Dict[str, List[str]]:
        """Get market-specific industry patterns"""
        pass

    @abstractmethod
    def normalize_brand_name(self, brand: str) -> Set[str]:
        """Generate market-specific brand variations"""
        pass

    @abstractmethod
    def calculate_market_confidence(self,
                                  original_text: str,
                                  brand: str,
                                  context: str) -> float:
        """Calculate market-specific confidence score"""
        pass

    @abstractmethod
    def extract_business_context(self, text: str) -> Dict[str, any]:
        """Extract market-specific business context"""
        pass

    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if not text or not isinstance(text, str):
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Market-specific preprocessing
        return self._market_specific_preprocessing(text)

    @abstractmethod
    def _market_specific_preprocessing(self, text: str) -> str:
        """Market-specific text preprocessing"""
        pass

    def validate_brand_mention(self,
                             brand: str,
                             context: str,
                             detection_method: DetectionMethod) -> bool:
        """Validate if brand mention is legitimate"""

        # Basic validation
        if not brand or len(brand.strip()) < 2:
            return False

        # Check for common false positives
        false_positive_patterns = [
            r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b',
            r'\b\d+\b',  # Pure numbers
            r'^[^\w\s]+$'  # Only special characters
        ]

        for pattern in false_positive_patterns:
            if re.match(pattern, brand.lower().strip()):
                return False

        return self._market_specific_validation(brand, context, detection_method)

    @abstractmethod
    def _market_specific_validation(self,
                                  brand: str,
                                  context: str,
                                  detection_method: DetectionMethod) -> bool:
        """Market-specific validation logic"""
        pass

    def get_performance_stats(self) -> Dict[str, any]:
        """Get adapter performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "market_code": self.market_code,
            "language_code": self.language_code,
            "cache_hit_rate": cache_hit_rate,
            "total_requests": total_requests
        }

class MarketAdapterError(Exception):
    """Custom exception for market adapter errors"""

    def __init__(self, message: str, market_code: str, error_code: str = None):
        self.market_code = market_code
        self.error_code = error_code
        super().__init__(f"Market {market_code}: {message}")

class MarketAdapterFactory:
    """Factory for creating market-specific adapters"""

    _adapters = {}

    @classmethod
    def register_adapter(cls, market_code: str, adapter_class):
        """Register a market adapter"""
        cls._adapters[market_code.upper()] = adapter_class

    @classmethod
    def create_adapter(cls, market_code: str) -> BaseMarketAdapter:
        """Create market adapter instance"""
        market_code = market_code.upper()

        if market_code not in cls._adapters:
            raise MarketAdapterError(
                f"No adapter registered for market {market_code}",
                market_code,
                "ADAPTER_NOT_FOUND"
            )

        try:
            return cls._adapters[market_code]()
        except Exception as e:
            raise MarketAdapterError(
                f"Failed to create adapter: {str(e)}",
                market_code,
                "ADAPTER_CREATION_FAILED"
            )

    @classmethod
    def get_available_markets(cls) -> List[str]:
        """Get list of available market codes"""
        return list(cls._adapters.keys())
EOF
```

## Step 4: German Market Adapter

```python
# Create: app/services/brand_detection/market_adapters/german_adapter.py
cat > app/services/brand_detection/market_adapters/german_adapter.py << 'EOF'
import re
from typing import List, Dict, Set, Optional
from .base import BaseMarketAdapter, MarketAdapterFactory
from ..models.brand_mention import DetectionMethod

class GermanMarketAdapter(BaseMarketAdapter):
    """German market-specific brand detection logic"""

    def __init__(self):
        super().__init__("DE", "de")

        # German-specific character mappings
        self.umlaut_mappings = {
            'ä': ['ae', 'a'], 'ö': ['oe', 'o'], 'ü': ['ue', 'u'],
            'Ä': ['Ae', 'AE', 'A'], 'Ö': ['Oe', 'OE', 'O'],
            'Ü': ['Ue', 'UE', 'U'], 'ß': ['ss', 's']
        }

        # German compound word patterns
        self.compound_patterns = [
            r'([A-ZÄÖÜ][a-zäöüß]+)([A-ZÄÖÜ][a-zäöüß]+)',  # CamelCase compounds
            r'([a-zäöüß]+)-([a-zäöüß]+)',  # Hyphenated compounds
        ]

        # German business context indicators
        self.business_context_indicators = [
            'unternehmen', 'firma', 'gesellschaft', 'konzern', 'gruppe',
            'software', 'lösung', 'anbieter', 'hersteller', 'dienstleister'
        ]

    def _get_company_suffixes(self) -> List[str]:
        """German company legal forms"""
        return [
            'GmbH', 'AG', 'KG', 'OHG', 'mbH', 'eV', 'UG',
            'GmbH & Co. KG', 'SE', 'KGaA', 'eG',
            # Variations
            'gmbh', 'ag', 'kg', 'ohg', 'mbh', 'ev', 'ug'
        ]

    def _get_business_keywords(self) -> List[str]:
        """German business context keywords"""
        return [
            'unternehmen', 'firma', 'gesellschaft', 'konzern', 'gruppe',
            'anbieter', 'hersteller', 'dienstleister', 'entwickler',
            'software', 'lösung', 'system', 'plattform', 'tool',
            'service', 'dienst', 'produkt', 'marke', 'brand'
        ]

    def _get_industry_patterns(self) -> Dict[str, List[str]]:
        """German industry-specific patterns"""
        return {
            'software': [
                'softwareunternehmen', 'softwareanbieter', 'softwarehersteller',
                'it-unternehmen', 'technologieunternehmen'
            ],
            'automotive': [
                'automobilhersteller', 'autobauer', 'fahrzeughersteller',
                'autozulieferer', 'automobilkonzern'
            ],
            'finance': [
                'bank', 'sparkasse', 'versicherung', 'finanzdienstleister',
                'kreditinstitut', 'geldinstitut'
            ],
            'retail': [
                'einzelhändler', 'handelskette', 'kaufhaus', 'supermarkt',
                'onlineshop', 'e-commerce'
            ]
        }

    def normalize_brand_name(self, brand: str) -> Set[str]:
        """Generate German-specific brand variations"""
        variations = {brand, brand.lower(), brand.upper()}

        # Handle umlauts
        for original, replacements in self.umlaut_mappings.items():
            if original in brand:
                for replacement in replacements:
                    variations.add(brand.replace(original, replacement))
                    variations.add(brand.replace(original, replacement).lower())

        # Handle company suffixes
        brand_without_suffix = self._remove_company_suffix(brand)
        if brand_without_suffix != brand:
            variations.add(brand_without_suffix)
            variations.add(brand_without_suffix.lower())

            # Add variations with different suffixes
            for suffix in self.company_suffixes[:5]:  # Top 5 most common
                variations.add(f"{brand_without_suffix} {suffix}")

        # Handle compound words
        compound_parts = self._split_compound_word(brand)
        if len(compound_parts) > 1:
            # Add individual parts
            variations.update(compound_parts)
            # Add different combinations
            variations.add(' '.join(compound_parts))
            variations.add('-'.join(compound_parts))

        # Handle acronyms (German companies often use them)
        if len(brand.split()) > 1:
            acronym = ''.join(word[0].upper() for word in brand.split() if word)
            variations.add(acronym)
            variations.add(acronym.lower())

        # Clean up variations
        return {v.strip() for v in variations if v.strip() and len(v.strip()) >= 2}

    def _remove_company_suffix(self, brand: str) -> str:
        """Remove German company suffixes"""
        for suffix in self.company_suffixes:
            patterns = [
                f" {suffix}$", f" {suffix.lower()}$",
                f"-{suffix}$", f"-{suffix.lower()}$"
            ]
            for pattern in patterns:
                brand = re.sub(pattern, "", brand, flags=re.IGNORECASE)
        return brand.strip()

    def _split_compound_word(self, word: str) -> List[str]:
        """Split German compound words"""
        parts = [word]

        for pattern in self.compound_patterns:
            matches = re.finditer(pattern, word)
            for match in matches:
                parts.extend(match.groups())

        # Remove duplicates and original word if split occurred
        unique_parts = list(set(parts))
        if len(unique_parts) > 1 and word in unique_parts:
            unique_parts.remove(word)

        return [part for part in unique_parts if len(part) >= 2]

    def calculate_market_confidence(self,
                                  original_text: str,
                                  brand: str,
                                  context: str) -> float:
        """Calculate German market-specific confidence"""
        base_confidence = 0.5

        # Boost for German business context
        german_context_score = self._calculate_german_context_score(context)
        base_confidence += german_context_score * 0.3

        # Boost for proper German capitalization
        if self._has_proper_german_capitalization(brand):
            base_confidence += 0.1

        # Boost for company suffix presence
        if self._has_german_company_suffix(brand):
            base_confidence += 0.15

        # Boost for compound word structure
        if self._is_german_compound_structure(brand):
            base_confidence += 0.1

        # Penalty for non-German characteristics
        if self._has_non_german_patterns(brand):
            base_confidence -= 0.2

        return min(1.0, max(0.0, base_confidence))

    def _calculate_german_context_score(self, context: str) -> float:
        """Calculate how German the context appears"""
        if not context:
            return 0.0

        context_lower = context.lower()
        german_indicator_count = 0

        # Check for German business keywords
        for keyword in self.business_context_indicators:
            if keyword in context_lower:
                german_indicator_count += 1

        # Check for German articles and prepositions
        german_words = ['der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'mit', 'für', 'von']
        for word in german_words:
            if f' {word} ' in context_lower:
                german_indicator_count += 0.5

        # Normalize score
        return min(1.0, german_indicator_count / 5)

    def _has_proper_german_capitalization(self, brand: str) -> bool:
        """Check if brand follows German capitalization rules"""
        if not brand:
            return False

        # German nouns are capitalized
        words = brand.split()
        if len(words) == 1:
            return words[0][0].isupper() if words[0] else False

        # For multi-word brands, major words should be capitalized
        capitalized_count = sum(1 for word in words if word and word[0].isupper())
        return capitalized_count / len(words) >= 0.5

    def _has_german_company_suffix(self, brand: str) -> bool:
        """Check if brand has German company suffix"""
        brand_lower = brand.lower()
        return any(suffix.lower() in brand_lower for suffix in self.company_suffixes)

    def _is_german_compound_structure(self, brand: str) -> bool:
        """Check if brand has German compound word structure"""
        return any(re.search(pattern, brand) for pattern in self.compound_patterns)

    def _has_non_german_patterns(self, brand: str) -> bool:
        """Check for patterns that are unlikely in German brands"""
        non_german_patterns = [
            r'\b(Inc|Corp|LLC|Ltd)\b',  # English suffixes
            r'^[a-z]+$',  # All lowercase (unusual for German)
            r'[xqy]{2,}',  # Letter combinations rare in German
        ]

        return any(re.search(pattern, brand, re.IGNORECASE) for pattern in non_german_patterns)

    def extract_business_context(self, text: str) -> Dict[str, any]:
        """Extract German business context"""
        context = {
            'industry_indicators': [],
            'business_relationships': [],
            'company_actions': [],
            'market_position': [],
            'german_specific': {
                'legal_forms': [],
                'compound_words': [],
                'industry_terms': []
            }
        }

        text_lower = text.lower()

        # Extract industry indicators
        for industry, terms in self.industry_patterns.items():
            found_terms = [term for term in terms if term in text_lower]
            if found_terms:
                context['industry_indicators'].append({
                    'industry': industry,
                    'terms': found_terms
                })

        # Extract legal forms
        legal_forms = [suffix for suffix in self.company_suffixes if suffix.lower() in text_lower]
        context['german_specific']['legal_forms'] = legal_forms

        # Extract compound words
        compound_matches = []
        for pattern in self.compound_patterns:
            matches = re.finditer(pattern, text)
            compound_matches.extend([match.group() for match in matches])
        context['german_specific']['compound_words'] = list(set(compound_matches))

        return context

    def _market_specific_preprocessing(self, text: str) -> str:
        """German-specific text preprocessing"""
        if not text:
            return ""

        # Normalize quotation marks (German uses different quotes)
        text = re.sub(r'[„""]', '"', text)
        text = re.sub(r'[‚'']', "'", text)

        # Handle German-specific punctuation
        text = re.sub(r'(?<=[a-zäöüß])\.(?=[A-ZÄÖÜ])', '. ', text)  # Add space after period

        # Normalize umlauts for consistent processing
        # (Keep originals but also create normalized versions)
        return text

    def _market_specific_validation(self,
                                  brand: str,
                                  context: str,
                                  detection_method: DetectionMethod) -> bool:
        """German-specific validation"""

        # Reject obvious non-German patterns
        if self._has_non_german_patterns(brand):
            return False

        # For fuzzy matching, be more strict
        if detection_method == DetectionMethod.FUZZY:
            if not self._has_proper_german_capitalization(brand):
                return False

        # Validate compound words make sense
        if self._is_german_compound_structure(brand):
            parts = self._split_compound_word(brand)
            if any(len(part) < 2 for part in parts):
                return False

        return True

# Register the adapter
MarketAdapterFactory.register_adapter("DE", GermanMarketAdapter)
EOF
```

## Step 5: Utility Classes

```python
# Create: app/services/brand_detection/utils/__init__.py
cat > app/services/brand_detection/utils/__init__.py << 'EOF'
"""Utility modules for brand detection"""
EOF
```

```python
# Create: app/services/brand_detection/utils/performance.py
cat > app/services/brand_detection/utils/performance.py << 'EOF'
import time
import functools
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    operation: str
    duration_ms: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

class PerformanceMonitor:
    """Thread-safe performance monitoring for brand detection operations"""

    def __init__(self, max_metrics_per_operation: int = 1000):
        self.max_metrics_per_operation = max_metrics_per_operation
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_per_operation))
        self._lock = threading.RLock()
        self._operation_counts = defaultdict(int)

    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric thread-safely"""
        with self._lock:
            self._metrics[metric.operation].append(metric)
            self._operation_counts[metric.operation] += 1

    @contextmanager
    def measure_operation(self,
                         operation: str,
                         metadata: Dict[str, Any] = None,
                         log_slow_operations: bool = True,
                         slow_threshold_ms: float = 1000.0):
        """Context manager for measuring operation performance"""
        start_time = time.perf_counter()
        timestamp = time.time()
        error_message = None
        success = True

        try:
            yield
        except Exception as e:
            error_message = str(e)
            success = False
            logger.error(f"Operation {operation} failed: {error_message}")
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            metric = PerformanceMetric(
                operation=operation,
                duration_ms=duration_ms,
                timestamp=timestamp,
                metadata=metadata or {},
                success=success,
                error_message=error_message
            )

            self.record_metric(metric)

            # Log slow operations
            if log_slow_operations and duration_ms > slow_threshold_ms:
                logger.warning(
                    f"Slow operation detected: {operation} took {duration_ms:.2f}ms "
                    f"(threshold: {slow_threshold_ms}ms)"
                )

    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation"""
        with self._lock:
            metrics = list(self._metrics.get(operation, []))

            if not metrics:
                return {
                    "operation": operation,
                    "total_calls": 0,
                    "avg_duration_ms": 0,
                    "min_duration_ms": 0,
                    "max_duration_ms": 0,
                    "success_rate": 0,
                    "error_count": 0
                }

            durations = [m.duration_ms for m in metrics]
            successes = [m for m in metrics if m.success]

            return {
                "operation": operation,
                "total_calls": len(metrics),
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "median_duration_ms": sorted(durations)[len(durations) // 2],
                "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)],
                "success_rate": len(successes) / len(metrics) * 100,
                "error_count": len(metrics) - len(successes),
                "recent_errors": [
                    m.error_message for m in metrics[-10:]
                    if not m.success and m.error_message
                ]
            }

def performance_monitor(operation: str = None,
                       metadata: Dict[str, Any] = None,
                       monitor_instance: PerformanceMonitor = None):
    """Decorator for monitoring function performance"""

    def decorator(func: Callable) -> Callable:
        op_name = operation or f"{func.__module__}.{func.__qualname__}"
        monitor = monitor_instance or _global_monitor

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with monitor.measure_operation(op_name, metadata):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with monitor.measure_operation(op_name, metadata):
                return await func(*args, **kwargs)

        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

class PerformanceProfiler:
    """Detailed profiler for complex operations"""

    def __init__(self, name: str):
        self.name = name
        self.checkpoints: List[Dict[str, Any]] = []
        self.start_time = None
        self.metadata = {}

    def start(self, metadata: Dict[str, Any] = None):
        """Start profiling"""
        self.start_time = time.perf_counter()
        self.metadata = metadata or {}
        self.checkpoints = []

    def checkpoint(self, label: str, metadata: Dict[str, Any] = None):
        """Add a checkpoint"""
        if self.start_time is None:
            raise ValueError("Profiler not started")

        current_time = time.perf_counter()
        elapsed_ms = (current_time - self.start_time) * 1000

        self.checkpoints.append({
            "label": label,
            "elapsed_ms": elapsed_ms,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })

    def finish(self) -> Dict[str, Any]:
        """Finish profiling and return results"""
        if self.start_time is None:
            raise ValueError("Profiler not started")

        total_time = (time.perf_counter() - self.start_time) * 1000

        # Calculate intervals between checkpoints
        intervals = []
        prev_time = 0
        for checkpoint in self.checkpoints:
            interval_ms = checkpoint["elapsed_ms"] - prev_time
            intervals.append({
                "label": checkpoint["label"],
                "interval_ms": interval_ms,
                "cumulative_ms": checkpoint["elapsed_ms"],
                "metadata": checkpoint["metadata"]
            })
            prev_time = checkpoint["elapsed_ms"]

        result = {
            "operation": self.name,
            "total_duration_ms": total_time,
            "checkpoints": intervals,
            "metadata": self.metadata,
            "timestamp": time.time()
        }

        # Record in global monitor
        metric = PerformanceMetric(
            operation=f"profile_{self.name}",
            duration_ms=total_time,
            timestamp=time.time(),
            metadata={"profile_data": result}
        )
        _global_monitor.record_metric(metric)

        return result

@contextmanager
def profile_operation(name: str, metadata: Dict[str, Any] = None):
    """Context manager for detailed profiling"""
    profiler = PerformanceProfiler(name)
    profiler.start(metadata)

    try:
        yield profiler
    finally:
        result = profiler.finish()
        logger.info(f"Profile complete for {name}: {result['total_duration_ms']:.2f}ms")

# Global performance monitor instance
_global_monitor = PerformanceMonitor()

def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return _global_monitor
EOF
```

```python
# Create: app/services/brand_detection/utils/cache_manager.py
cat > app/services/brand_detection/utils/cache_manager.py << 'EOF'
import json
import hashlib
import time
from typing import Any, Optional, Dict, List, Callable, Union
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class CacheKeyGenerator:
    """Generate consistent cache keys for different operations"""

    @staticmethod
    def brand_normalization_key(brand: str, market_code: str) -> str:
        """Generate key for brand normalization cache"""
        content = f"norm:{market_code}:{brand.lower()}"
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def detection_result_key(text: str, brands: List[str], market_code: str,
                           confidence_threshold: float) -> str:
        """Generate key for detection result cache"""
        # Create stable hash from input parameters
        brands_str = ",".join(sorted(b.lower() for b in brands))
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        content = f"detect:{market_code}:{confidence_threshold}:{brands_str}:{text_hash}"
        return hashlib.md5(content.encode()).hexdigest()

class BrandDetectionCache:
    """Simple in-memory caching system for brand detection operations"""

    def __init__(self, default_ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.key_generator = CacheKeyGenerator()

        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_errors = 0

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        return time.time() > entry.get("expires_at", 0)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if key in self.cache:
                entry = self.cache[key]
                if not self._is_expired(entry):
                    self.cache_hits += 1
                    return entry["value"]
                else:
                    # Remove expired entry
                    del self.cache[key]

            self.cache_misses += 1
            return None

        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.cache_errors += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl

            self.cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": time.time()
            }
            return True

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.cache_errors += 1
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            self.cache_errors += 1
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if key in self.cache:
                return not self._is_expired(self.cache[key])
            return False
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            self.cache_errors += 1
            return False

    def clear_expired(self):
        """Clear expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time > entry.get("expires_at", 0)
        ]
        for key in expired_keys:
            del self.cache[key]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_errors": self.cache_errors,
            "hit_rate_percent": hit_rate,
            "total_requests": total_requests,
            "cache_size": len(self.cache)
        }

# Global cache instance
_global_cache: Optional[BrandDetectionCache] = None

def initialize_global_cache(default_ttl: int = 3600):
    """Initialize the global cache instance"""
    global _global_cache
    _global_cache = BrandDetectionCache(default_ttl)

def get_global_cache() -> Optional[BrandDetectionCache]:
    """Get the global cache instance"""
    if _global_cache is None:
        initialize_global_cache()
    return _global_cache
EOF
```

## Step 6: Core Detection Components

```python
# Create: app/services/brand_detection/core/__init__.py
cat > app/services/brand_detection/core/__init__.py << 'EOF'
"""Core brand detection components"""
EOF
```

```python
# Create: app/services/brand_detection/core/similarity.py
cat > app/services/brand_detection/core/similarity.py << 'EOF'
import asyncio
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from rapidfuzz import fuzz
import openai

logger = logging.getLogger(__name__)

@dataclass
class SimilarityResult:
    """Result of similarity comparison"""
    score: float
    method: str
    confidence: float
    metadata: Dict[str, any] = None

class SimilarityEngine:
    """Multi-modal similarity engine for brand matching"""

    def __init__(self, openai_api_key: str, similarity_threshold: float = 0.7):
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = 85

    async def get_openai_embedding(self, text: str) -> Optional[List[float]]:
        """Get OpenAI embedding for text"""
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text.strip()
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return None

    def calculate_fuzzy_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """Calculate fuzzy string similarity"""

        # Multiple fuzzy matching algorithms
        ratio_score = fuzz.ratio(text1.lower(), text2.lower()) / 100
        partial_ratio = fuzz.partial_ratio(text1.lower(), text2.lower()) / 100
        token_sort = fuzz.token_sort_ratio(text1.lower(), text2.lower()) / 100
        token_set = fuzz.token_set_ratio(text1.lower(), text2.lower()) / 100

        # Weighted combination
        weights = [0.3, 0.2, 0.25, 0.25]
        scores = [ratio_score, partial_ratio, token_sort, token_set]
        final_score = sum(w * s for w, s in zip(weights, scores))

        # Confidence based on score consistency
        import numpy as np
        score_variance = np.var(scores)
        confidence = max(0.0, 1.0 - score_variance)

        return SimilarityResult(
            score=final_score,
            method="fuzzy_matching",
            confidence=confidence,
            metadata={
                "ratio": ratio_score,
                "partial_ratio": partial_ratio,
                "token_sort": token_sort,
                "token_set": token_set,
                "variance": score_variance
            }
        )

    async def calculate_semantic_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """Calculate semantic similarity using OpenAI embeddings"""
        try:
            emb1 = await self.get_openai_embedding(text1)
            emb2 = await self.get_openai_embedding(text2)

            if emb1 and emb2:
                cosine_sim = self._cosine_similarity(emb1, emb2)
                return SimilarityResult(
                    score=cosine_sim,
                    method="openai_embedding",
                    confidence=0.9,  # High confidence for OpenAI embeddings
                    metadata={"embedding_model": "text-embedding-3-small"}
                )
        except Exception as e:
            logger.error(f"Semantic similarity failed: {e}")

        # Fallback to fuzzy matching
        return self.calculate_fuzzy_similarity(text1, text2)

    async def calculate_hybrid_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """Calculate hybrid similarity using multiple methods"""

        # Get fuzzy similarity (fast)
        fuzzy_result = self.calculate_fuzzy_similarity(text1, text2)

        # If fuzzy similarity is very low, skip expensive semantic analysis
        if fuzzy_result.score < 0.3:
            return fuzzy_result

        # Get semantic similarity
        semantic_result = await self.calculate_semantic_similarity(text1, text2)

        # Combine results intelligently
        if semantic_result.score > 0:
            # Weight semantic similarity higher for closer matches
            semantic_weight = 0.7 if fuzzy_result.score > 0.5 else 0.5
            fuzzy_weight = 1 - semantic_weight

            combined_score = (semantic_weight * semantic_result.score +
                            fuzzy_weight * fuzzy_result.score)

            confidence = (semantic_result.confidence + fuzzy_result.confidence) / 2
            method = f"hybrid({semantic_result.method}+{fuzzy_result.method})"

            metadata = {
                "semantic_score": semantic_result.score,
                "fuzzy_score": fuzzy_result.score,
                "semantic_weight": semantic_weight,
                "fuzzy_weight": fuzzy_weight
            }
        else:
            # Fall back to fuzzy only
            combined_score = fuzzy_result.score
            confidence = fuzzy_result.confidence * 0.8  # Lower confidence without semantic
            method = f"fuzzy_fallback({fuzzy_result.method})"
            metadata = fuzzy_result.metadata

        return SimilarityResult(
            score=combined_score,
            method=method,
            confidence=confidence,
            metadata=metadata
        )

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            return float(dot_product / (norm_v1 * norm_v2))
        except Exception as e:
            logger.error(f"Cosine similarity calculation error: {e}")
            return 0.0

    def get_cache_stats(self) -> Dict[str, any]:
        """Get similarity engine statistics"""
        return {
            "similarity_threshold": self.similarity_threshold,
            "fuzzy_threshold": self.fuzzy_threshold
        }
EOF
```

```python
# Create: app/services/brand_detection/core/normalizer.py
cat > app/services/brand_detection/core/normalizer.py << 'EOF'
import re
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import unicodedata
import logging
from ..market_adapters.base import BaseMarketAdapter, MarketAdapterFactory

logger = logging.getLogger(__name__)

@dataclass
class NormalizationResult:
    """Result of brand name normalization"""
    original: str
    normalized_forms: Set[str]
    primary_form: str
    confidence: float
    market_specific_forms: Dict[str, Set[str]]
    metadata: Dict[str, any]

class BrandNormalizer:
    """Advanced brand name normalization engine"""

    def __init__(self):
        self.common_abbreviations = {
            'corporation': ['corp', 'co'],
            'company': ['co', 'comp'],
            'incorporated': ['inc', 'incorp'],
            'limited': ['ltd', 'lim'],
            'gesellschaft': ['ges'],
            'aktiengesellschaft': ['ag'],
            'gmbh': ['gesellschaft mit beschränkter haftung'],
            'international': ['intl', 'int\'l'],
            'technologies': ['tech', 'technologies'],
            'software': ['sw', 'soft'],
            'systems': ['sys', 'syst']
        }

        # Unicode normalization patterns
        self.unicode_patterns = [
            # Smart quotes
            (r'["""]', '"'),
            (r'[''']', "'"),
            # Dashes
            (r'[‒–—―]', '-'),
            # Spaces
            (r'[\u00A0\u2000-\u200B\u2028\u2029\u202F\u205F\u3000]', ' '),
        ]

        # Common brand name patterns
        self.brand_patterns = [
            # Camel case splitting
            (r'([a-z])([A-Z])', r'\1 \2'),
            # Number-letter boundaries
            (r'(\d)([A-Za-z])', r'\1 \2'),
            (r'([A-Za-z])(\d)', r'\1 \2'),
        ]

    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters in brand name"""
        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)

        # Apply pattern replacements
        for pattern, replacement in self.unicode_patterns:
            text = re.sub(pattern, replacement, text)

        return text.strip()

    def normalize_basic(self, brand: str) -> Set[str]:
        """Basic normalization - case, spacing, punctuation"""
        if not brand:
            return set()

        variations = set()

        # Original
        variations.add(brand)

        # Unicode normalized
        normalized = self.normalize_unicode(brand)
        variations.add(normalized)

        # Case variations
        variations.add(brand.lower())
        variations.add(brand.upper())
        variations.add(brand.title())

        # Remove punctuation variations
        no_punct = re.sub(r'[^\w\s]', '', brand)
        if no_punct != brand:
            variations.add(no_punct)
            variations.add(no_punct.lower())

        # Normalize whitespace
        normalized_space = re.sub(r'\s+', ' ', brand).strip()
        variations.add(normalized_space)

        # Remove extra characters
        clean = re.sub(r'[^\w\s-]', '', brand)
        variations.add(clean)

        return {v for v in variations if v and v.strip()}

    def normalize_comprehensive(self, brand: str, market_code: str = "DE") -> NormalizationResult:
        """Comprehensive brand normalization"""

        if not brand or not brand.strip():
            return NormalizationResult(
                original="",
                normalized_forms=set(),
                primary_form="",
                confidence=0.0,
                market_specific_forms={},
                metadata={"error": "Empty brand name"}
            )

        all_variations = set()
        market_specific = {}

        try:
            # Get market adapter
            adapter = MarketAdapterFactory.create_adapter(market_code)

            # Basic normalization
            basic_forms = self.normalize_basic(brand)
            all_variations.update(basic_forms)

            # Market-specific normalization
            market_forms = adapter.normalize_brand_name(brand)
            all_variations.update(market_forms)
            market_specific[market_code] = market_forms

            # Clean up variations
            clean_variations = {
                v.strip() for v in all_variations
                if v and v.strip() and len(v.strip()) >= 2
            }

            # Determine primary form (most common or original)
            primary_form = self._determine_primary_form(brand, clean_variations)

            # Calculate confidence based on consistency
            confidence = self._calculate_normalization_confidence(brand, clean_variations)

            metadata = {
                "total_variations": len(clean_variations),
                "market_code": market_code,
                "has_market_specific": len(market_forms) > 0,
                "normalization_methods": [
                    "basic", "market_specific"
                ]
            }

            return NormalizationResult(
                original=brand,
                normalized_forms=clean_variations,
                primary_form=primary_form,
                confidence=confidence,
                market_specific_forms=market_specific,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Normalization failed for brand '{brand}': {e}")
            return NormalizationResult(
                original=brand,
                normalized_forms={brand},
                primary_form=brand,
                confidence=0.5,
                market_specific_forms={},
                metadata={"error": str(e)}
            )

    def _determine_primary_form(self, original: str, variations: Set[str]) -> str:
        """Determine the primary normalized form"""
        if not variations:
            return original

        # Preference order:
        # 1. Original if it exists in variations
        if original in variations:
            return original

        # 2. Title case version
        title_versions = [v for v in variations if v.istitle()]
        if title_versions:
            return min(title_versions, key=len)  # Shortest title case

        # 3. First uppercase version
        upper_versions = [v for v in variations if v[0].isupper()]
        if upper_versions:
            return min(upper_versions, key=len)

        # 4. Shortest variation
        return min(variations, key=len)

    def _calculate_normalization_confidence(self, original: str, variations: Set[str]) -> float:
        """Calculate confidence in normalization quality"""
        if not variations:
            return 0.0

        base_confidence = 0.5

        # More variations generally mean better coverage
        variation_boost = min(0.3, len(variations) * 0.02)
        base_confidence += variation_boost

        # Prefer if original is preserved
        if original in variations:
            base_confidence += 0.1

        # Check for reasonable variation distribution
        length_variance = len(set(len(v) for v in variations))
        if length_variance > 1:  # Good variation in lengths
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def get_normalization_stats(self) -> Dict[str, any]:
        """Get normalization performance statistics"""
        return {
            "abbreviation_mappings": len(self.common_abbreviations),
            "unicode_patterns": len(self.unicode_patterns),
            "brand_patterns": len(self.brand_patterns)
        }
EOF
```

```python
# Create: app/services/brand_detection/core/sentiment.py
cat > app/services/brand_detection/core/sentiment.py << 'EOF'
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ..models.brand_mention import SentimentPolarity

logger = logging.getLogger(__name__)

class SentimentMethod(Enum):
    VADER = "vader"
    CUSTOM_BUSINESS = "custom_business"
    HYBRID = "hybrid"

@dataclass
class SentimentAnalysisResult:
    """Result of sentiment analysis"""
    polarity: SentimentPolarity
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    method: SentimentMethod
    context_type: str
    metadata: Dict[str, Any]

class BusinessContextAnalyzer:
    """Analyze business-specific context around brand mentions"""

    def __init__(self):
        # Business context patterns
        self.positive_business_patterns = [
            # Recommendation patterns
            r'\b(recommend|suggests?|advise|endorse)\b.*?{brand}',
            r'{brand}.*?\b(excellent|outstanding|superior|leading|top|best)\b',
            r'\b(choose|select|opt for|go with)\b.*?{brand}',
            r'{brand}.*?\b(solution|winner|leader|choice)\b',

            # Quality patterns
            r'{brand}.*?\b(high[- ]quality|reliable|trusted|proven)\b',
            r'\b(award[- ]winning|certified|approved)\b.*?{brand}',
            r'{brand}.*?\b(industry[- ]standard|market[- ]leader)\b',
        ]

        self.negative_business_patterns = [
            # Problem patterns
            r'{brand}.*?\b(problem|issue|bug|error|fault|fail)\b',
            r'\b(avoid|skip|ignore)\b.*?{brand}',
            r'{brand}.*?\b(disappointing|terrible|awful|poor)\b',
            r'\b(regret|mistake)\b.*?{brand}',

            # Comparison disadvantages
            r'{brand}.*?\b(worse|inferior|behind|lacking)\b',
            r'\b(better|superior)\b.*?than.*?{brand}',
            r'{brand}.*?\b(outdated|obsolete|deprecated)\b',
        ]

    def analyze_business_context(self, text: str, brand: str) -> Dict[str, Any]:
        """Analyze business context around brand mention"""

        # Escape brand name for regex
        brand_escaped = re.escape(brand)

        context_analysis = {
            "positive_signals": 0,
            "negative_signals": 0,
            "context_type": "neutral",
            "specific_patterns": []
        }

        # Check positive patterns
        for pattern in self.positive_business_patterns:
            regex_pattern = pattern.format(brand=brand_escaped)
            matches = re.finditer(regex_pattern, text, re.IGNORECASE)
            for match in matches:
                context_analysis["positive_signals"] += 1
                context_analysis["specific_patterns"].append({
                    "type": "positive",
                    "pattern": pattern,
                    "match": match.group(),
                    "position": match.span()
                })

        # Check negative patterns
        for pattern in self.negative_business_patterns:
            regex_pattern = pattern.format(brand=brand_escaped)
            matches = re.finditer(regex_pattern, text, re.IGNORECASE)
            for match in matches:
                context_analysis["negative_signals"] += 1
                context_analysis["specific_patterns"].append({
                    "type": "negative",
                    "pattern": pattern,
                    "match": match.group(),
                    "position": match.span()
                })

        # Determine overall context type
        total_signals = context_analysis["positive_signals"] + context_analysis["negative_signals"]
        if total_signals > 0:
            positive_ratio = context_analysis["positive_signals"] / total_signals
            if positive_ratio > 0.6:
                context_analysis["context_type"] = "positive_business"
            elif positive_ratio < 0.4:
                context_analysis["context_type"] = "negative_business"
            else:
                context_analysis["context_type"] = "mixed_business"

        return context_analysis

class SentimentAnalyzer:
    """Multi-modal sentiment analysis for brand mentions"""

    def __init__(self):
        self.business_analyzer = BusinessContextAnalyzer()

        # Initialize VADER (lightweight, always available)
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment_vader(self, text: str) -> SentimentAnalysisResult:
        """Analyze sentiment using VADER"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            compound_score = scores['compound']

            # Convert to our polarity enum
            if compound_score >= 0.05:
                polarity = SentimentPolarity.POSITIVE
            elif compound_score <= -0.05:
                polarity = SentimentPolarity.NEGATIVE
            else:
                polarity = SentimentPolarity.NEUTRAL

            # Calculate confidence based on score magnitude
            confidence = min(1.0, abs(compound_score) * 2)

            return SentimentAnalysisResult(
                polarity=polarity,
                score=compound_score,
                confidence=confidence,
                method=SentimentMethod.VADER,
                context_type="general",
                metadata={
                    "positive": scores['pos'],
                    "negative": scores['neg'],
                    "neutral": scores['neu'],
                    "compound": scores['compound']
                }
            )

        except Exception as e:
            logger.error(f"VADER sentiment analysis failed: {e}")
            return SentimentAnalysisResult(
                polarity=SentimentPolarity.NEUTRAL,
                score=0.0,
                confidence=0.0,
                method=SentimentMethod.VADER,
                context_type="error",
                metadata={"error": str(e)}
            )

    def analyze_business_sentiment(self, text: str, brand: str, language: str = "en") -> SentimentAnalysisResult:
        """Analyze sentiment with business context awareness"""

        # Get business context analysis
        business_context = self.business_analyzer.analyze_business_context(text, brand)

        # Get base sentiment analysis
        base_sentiment = self.analyze_sentiment_vader(text)

        # Adjust sentiment based on business context
        adjusted_score = base_sentiment.score
        adjusted_confidence = base_sentiment.confidence

        # Business context adjustments
        positive_signals = business_context["positive_signals"]
        negative_signals = business_context["negative_signals"]
        total_signals = positive_signals + negative_signals

        if total_signals > 0:
            business_sentiment_bias = (positive_signals - negative_signals) / total_signals

            # Weighted combination of linguistic and business sentiment
            linguistic_weight = 0.7
            business_weight = 0.3

            adjusted_score = (linguistic_weight * base_sentiment.score +
                            business_weight * business_sentiment_bias)

            # Increase confidence if business signals align with linguistic sentiment
            if (business_sentiment_bias > 0 and base_sentiment.score > 0) or \
               (business_sentiment_bias < 0 and base_sentiment.score < 0):
                adjusted_confidence = min(1.0, adjusted_confidence + 0.2)

        # Determine final polarity
        if adjusted_score >= 0.1:
            final_polarity = SentimentPolarity.POSITIVE
        elif adjusted_score <= -0.1:
            final_polarity = SentimentPolarity.NEGATIVE


```python
        elif adjusted_score <= -0.1:
            final_polarity = SentimentPolarity.NEGATIVE
        else:
            final_polarity = SentimentPolarity.NEUTRAL

        # Handle mixed signals
        if total_signals > 2 and abs(positive_signals - negative_signals) <= 1:
            final_polarity = SentimentPolarity.MIXED

        return SentimentAnalysisResult(
            polarity=final_polarity,
            score=adjusted_score,
            confidence=adjusted_confidence,
            method=SentimentMethod.CUSTOM_BUSINESS,
            context_type=business_context["context_type"],
            metadata={
                "base_sentiment": base_sentiment.__dict__,
                "business_context": business_context,
                "adjustment_applied": abs(adjusted_score - base_sentiment.score) > 0.05,
                "linguistic_weight": linguistic_weight,
                "business_weight": business_weight
            }
        )

    async def analyze_sentiment_hybrid(self, text: str, brand: str, language: str = "en") -> SentimentAnalysisResult:
        """Hybrid sentiment analysis using multiple methods"""

        # Run multiple analyses
        analyses = {}

        # VADER (fast, baseline)
        try:
            analyses['vader'] = self.analyze_sentiment_vader(text)
        except Exception as e:
            logger.error(f"VADER analysis failed: {e}")

        # Business context (domain-specific)
        try:
            analyses['business'] = self.analyze_business_sentiment(text, brand, language)
        except Exception as e:
            logger.error(f"Business sentiment analysis failed: {e}")

        if not analyses:
            # All methods failed
            return SentimentAnalysisResult(
                polarity=SentimentPolarity.NEUTRAL,
                score=0.0,
                confidence=0.0,
                method=SentimentMethod.HYBRID,
                context_type="error",
                metadata={"error": "All sentiment analysis methods failed"}
            )

        # Combine results intelligently
        combined_result = self._combine_sentiment_results(analyses)
        combined_result.method = SentimentMethod.HYBRID

        return combined_result

    def _combine_sentiment_results(self, analyses: Dict[str, SentimentAnalysisResult]) -> SentimentAnalysisResult:
        """Combine multiple sentiment analysis results"""

        if not analyses:
            return SentimentAnalysisResult(
                polarity=SentimentPolarity.NEUTRAL,
                score=0.0,
                confidence=0.0,
                method=SentimentMethod.HYBRID,
                context_type="error",
                metadata={"error": "No analyses to combine"}
            )

        # Weight different methods
        method_weights = {
            'vader': 0.4,
            'business': 0.6
        }

        weighted_score = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        polarity_votes = {
            SentimentPolarity.POSITIVE: 0,
            SentimentPolarity.NEGATIVE: 0,
            SentimentPolarity.NEUTRAL: 0,
            SentimentPolarity.MIXED: 0
        }

        for method, result in analyses.items():
            weight = method_weights.get(method, 0.1)
            total_weight += weight

            weighted_score += result.score * weight
            weighted_confidence += result.confidence * weight
            polarity_votes[result.polarity] += weight

        # Normalize weights
        if total_weight > 0:
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_score = 0.0
            final_confidence = 0.0

        # Determine final polarity by voting
        final_polarity = max(polarity_votes.keys(), key=lambda k: polarity_votes[k])

        # Override polarity if score suggests otherwise
        if final_score >= 0.15:
            final_polarity = SentimentPolarity.POSITIVE
        elif final_score <= -0.15:
            final_polarity = SentimentPolarity.NEGATIVE
        elif abs(final_score) < 0.05:
            final_polarity = SentimentPolarity.NEUTRAL

        # Determine context type
        context_types = [result.context_type for result in analyses.values()]
        if "positive_business" in context_types:
            context_type = "positive_business"
        elif "negative_business" in context_types:
            context_type = "negative_business"
        else:
            context_type = "general"

        return SentimentAnalysisResult(
            polarity=final_polarity,
            score=final_score,
            confidence=final_confidence,
            method=SentimentMethod.HYBRID,
            context_type=context_type,
            metadata={
                "individual_results": {k: v.__dict__ for k, v in analyses.items()},
                "method_weights": method_weights,
                "polarity_votes": {k.value: v for k, v in polarity_votes.items()},
                "total_methods": len(analyses)
            }
        )
EOF
```

## Step 7: Main Detection Engine

```python
# Create: app/services/brand_detection/core/detector.py
cat > app/services/brand_detection/core/detector.py << 'EOF'
import asyncio
import time
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from ..models.brand_mention import BrandMention, BrandContext, DetectionMethod, DetectionResult
from ..market_adapters.base import BaseMarketAdapter, MarketAdapterFactory
from ..core.similarity import SimilarityEngine
from ..core.normalizer import BrandNormalizer, NormalizationResult
from ..core.sentiment import SentimentAnalyzer
from ..utils.performance import performance_monitor
from ..utils.cache_manager import get_global_cache

logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    """Configuration for brand detection operations"""
    confidence_threshold: float = 0.7
    similarity_threshold: float = 0.8
    max_context_window: int = 200
    enable_fuzzy_matching: bool = True
    enable_semantic_similarity: bool = True
    enable_sentiment_analysis: bool = True
    market_code: str = "DE"
    language_code: str = "de"
    max_mentions_per_brand: int = 50
    enable_caching: bool = True
    cache_ttl: int = 1800  # 30 minutes

class BrandDetectionEngine:
    """Advanced multi-modal brand detection engine"""

    def __init__(self, openai_api_key: str, config: DetectionConfig = None):
        self.config = config or DetectionConfig()

        # Initialize components
        self.similarity_engine = SimilarityEngine(openai_api_key)
        self.normalizer = BrandNormalizer()
        self.sentiment_analyzer = SentimentAnalyzer()

        # Performance tracking
        self.detection_stats = {
            "total_detections": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

    @performance_monitor()
    async def detect_brands(self,
                          text: str,
                          target_brands: List[str],
                          config: DetectionConfig = None) -> DetectionResult:
        """Main brand detection method"""

        detection_config = config or self.config
        start_time = time.perf_counter()

        # Input validation
        if not text or not target_brands:
            return DetectionResult(
                text_analyzed="",
                language=detection_config.language_code,
                processing_time_ms=0,
                mentions=[],
                total_brands_found=0,
                confidence_threshold=detection_config.confidence_threshold,
                market_adapter_used=detection_config.market_code
            )

        # Check cache
        cache_key = self._generate_cache_key(text, target_brands, detection_config)
        if detection_config.enable_caching:
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.detection_stats["cache_hits"] += 1
                return cached_result
            self.detection_stats["cache_misses"] += 1

        try:
            # Get market adapter
            market_adapter = MarketAdapterFactory.create_adapter(detection_config.market_code)

            # Preprocess text
            preprocessed_text = market_adapter.preprocess_text(text)

            # Normalize all target brands
            normalized_brands = await self._normalize_brands(target_brands, market_adapter)

            # Run detection methods in parallel
            detection_tasks = []

            if detection_config.enable_fuzzy_matching:
                detection_tasks.append(
                    self._detect_with_fuzzy_matching(preprocessed_text, normalized_brands, market_adapter)
                )

            if detection_config.enable_semantic_similarity:
                detection_tasks.append(
                    self._detect_with_semantic_similarity(preprocessed_text, normalized_brands, market_adapter)
                )

            # Execute detection methods
            detection_results = await asyncio.gather(*detection_tasks, return_exceptions=True)

            # Merge and deduplicate results
            all_mentions = []
            for result in detection_results:
                if isinstance(result, list):
                    all_mentions.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Detection method failed: {result}")

            # Deduplicate mentions
            unique_mentions = self._deduplicate_mentions(all_mentions, detection_config)

            # Enhance with sentiment analysis
            if detection_config.enable_sentiment_analysis:
                enhanced_mentions = await self._enhance_with_sentiment(
                    unique_mentions, preprocessed_text, detection_config
                )
            else:
                enhanced_mentions = unique_mentions

            # Filter by confidence threshold
            filtered_mentions = [
                mention for mention in enhanced_mentions
                if mention.confidence >= detection_config.confidence_threshold
            ]

            # Limit mentions per brand
            limited_mentions = self._limit_mentions_per_brand(
                filtered_mentions, detection_config.max_mentions_per_brand
            )

            # Create result
            processing_time = (time.perf_counter() - start_time) * 1000

            result = DetectionResult(
                text_analyzed=text,
                language=detection_config.language_code,
                processing_time_ms=processing_time,
                mentions=limited_mentions,
                total_brands_found=len(limited_mentions),
                confidence_threshold=detection_config.confidence_threshold,
                market_adapter_used=detection_config.market_code
            )

            # Cache result
            if detection_config.enable_caching:
                self._cache_result(cache_key, result, detection_config.cache_ttl)

            self.detection_stats["successful_detections"] += 1
            self.detection_stats["total_detections"] += 1

            return result

        except Exception as e:
            logger.error(f"Brand detection failed: {e}")
            self.detection_stats["failed_detections"] += 1
            self.detection_stats["total_detections"] += 1

            processing_time = (time.perf_counter() - start_time) * 1000
            return DetectionResult(
                text_analyzed=text,
                language=detection_config.language_code,
                processing_time_ms=processing_time,
                mentions=[],
                total_brands_found=0,
                confidence_threshold=detection_config.confidence_threshold,
                market_adapter_used=detection_config.market_code
            )

    async def _normalize_brands(self, brands: List[str], market_adapter: BaseMarketAdapter) -> Dict[str, NormalizationResult]:
        """Normalize all target brands"""
        normalized = {}

        for brand in brands:
            try:
                result = self.normalizer.normalize_comprehensive(brand, market_adapter.market_code)
                normalized[brand] = result
            except Exception as e:
                logger.error(f"Normalization failed for brand {brand}: {e}")
                # Fallback normalization
                normalized[brand] = NormalizationResult(
                    original=brand,
                    normalized_forms={brand, brand.lower()},
                    primary_form=brand,
                    confidence=0.5,
                    market_specific_forms={},
                    metadata={"error": str(e)}
                )

        return normalized

    @performance_monitor()
    async def _detect_with_fuzzy_matching(self,
                                        text: str,
                                        normalized_brands: Dict[str, NormalizationResult],
                                        market_adapter: BaseMarketAdapter) -> List[BrandMention]:
        """Detect brands using fuzzy string matching"""

        mentions = []

        try:
            # Extract potential brand candidates from text
            candidates = self._extract_brand_candidates(text, market_adapter)

            for brand, norm_result in normalized_brands.items():
                all_brand_forms = norm_result.normalized_forms

                for candidate in candidates:
                    best_match = None
                    best_score = 0.0

                    # Compare candidate against all normalized forms
                    for brand_form in all_brand_forms:
                        similarity = self.similarity_engine.calculate_fuzzy_similarity(
                            candidate['text'], brand_form
                        )

                        if similarity.score > best_score:
                            best_score = similarity.score
                            best_match = (brand_form, similarity)

                    # Check if match is good enough
                    if best_match and best_score >= self.config.similarity_threshold:
                        # Additional validation
                        if market_adapter.validate_brand_mention(
                            candidate['text'],
                            candidate['context'],
                            DetectionMethod.FUZZY
                        ):
                            mention = BrandMention(
                                brand=brand,
                                original_text=candidate['text'],
                                confidence=best_score,
                                detection_method=DetectionMethod.FUZZY,
                                language=market_adapter.language_code
                            )

                            # Add context
                            context = BrandContext(
                                text=candidate['context'],
                                start_position=candidate['start'],
                                end_position=candidate['end'],
                                sentence=candidate['sentence']
                            )
                            mention.add_context(context)

                            mentions.append(mention)

        except Exception as e:
            logger.error(f"Fuzzy matching detection failed: {e}")

        return mentions

    @performance_monitor()
    async def _detect_with_semantic_similarity(self,
                                             text: str,
                                             normalized_brands: Dict[str, NormalizationResult],
                                             market_adapter: BaseMarketAdapter) -> List[BrandMention]:
        """Detect brands using semantic similarity"""

        mentions = []

        try:
            # Extract potential brand candidates
            candidates = self._extract_brand_candidates(text, market_adapter)

            # Batch process for efficiency
            batch_size = 10
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]

                # Process each brand
                for brand, norm_result in normalized_brands.items():
                    brand_forms = list(norm_result.normalized_forms)[:5]  # Limit for performance

                    for candidate in batch:
                        # Find best semantic match
                        best_similarity = None
                        best_score = 0.0

                        for brand_form in brand_forms:
                            similarity = await self.similarity_engine.calculate_hybrid_similarity(
                                candidate['text'], brand_form
                            )

                            if similarity.score > best_score:
                                best_score = similarity.score
                                best_similarity = similarity

                        # Check threshold and validate
                        if (best_similarity and
                            best_score >= self.config.similarity_threshold and
                            market_adapter.validate_brand_mention(
                                candidate['text'],
                                candidate['context'],
                                DetectionMethod.SEMANTIC
                            )):

                            mention = BrandMention(
                                brand=brand,
                                original_text=candidate['text'],
                                confidence=best_score,
                                detection_method=DetectionMethod.SEMANTIC,
                                language=market_adapter.language_code
                            )

                            # Add context with semantic metadata
                            context = BrandContext(
                                text=candidate['context'],
                                start_position=candidate['start'],
                                end_position=candidate['end'],
                                sentence=candidate['sentence']
                            )
                            mention.add_context(context)

                            # Store semantic similarity metadata
                            mention.market_specific_data['semantic_similarity'] = {
                                'method': best_similarity.method,
                                'confidence': best_similarity.confidence,
                                'metadata': best_similarity.metadata
                            }

                            mentions.append(mention)

        except Exception as e:
            logger.error(f"Semantic similarity detection failed: {e}")

        return mentions

    def _extract_brand_candidates(self, text: str, market_adapter: BaseMarketAdapter) -> List[Dict[str, Any]]:
        """Extract potential brand candidates from text"""
        candidates = []

        # Pattern-based extraction
        patterns = [
            # Capitalized words (potential brand names)
            r'\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b',
            # Words with specific suffixes
            r'\b\w+(?:GmbH|AG|Inc|Corp|LLC|Ltd)\b',
            # Quoted strings (often brand names)
            r'"([^"]+)"',
            r"'([^']+)'",
            # CamelCase words
            r'\b[a-z]+[A-Z][a-zA-Z]*\b'
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                candidate_text = match.group()

                # Skip if too short or too long
                if len(candidate_text.strip()) < 2 or len(candidate_text) > 50:
                    continue

                # Extract context
                start = match.start()
                end = match.end()
                context_start = max(0, start - self.config.max_context_window // 2)
                context_end = min(len(text), end + self.config.max_context_window // 2)
                context = text[context_start:context_end]

                # Extract sentence
                sentence = self._extract_sentence(text, start, end)

                candidates.append({
                    'text': candidate_text.strip(),
                    'start': start,
                    'end': end,
                    'context': context,
                    'sentence': sentence
                })

        # Deduplicate candidates
        unique_candidates = []
        seen_texts = set()

        for candidate in candidates:
            if candidate['text'].lower() not in seen_texts:
                unique_candidates.append(candidate)
                seen_texts.add(candidate['text'].lower())

        return unique_candidates

    def _extract_sentence(self, text: str, start: int, end: int) -> str:
        """Extract the sentence containing the brand mention"""
        # Find sentence boundaries
        sentence_start = start
        sentence_end = end

        # Look backwards for sentence start
        for i in range(start - 1, -1, -1):
            if text[i] in '.!?':
                sentence_start = i + 1
                break

        # Look forwards for sentence end
        for i in range(end, len(text)):
            if text[i] in '.!?':
                sentence_end = i + 1
                break

        return text[sentence_start:sentence_end].strip()

    def _deduplicate_mentions(self, mentions: List[BrandMention], config: DetectionConfig) -> List[BrandMention]:
        """Remove duplicate brand mentions"""
        if not mentions:
            return []

        # Group by brand and original text
        groups = {}
        for mention in mentions:
            key = (mention.brand, mention.original_text.lower())
            if key not in groups:
                groups[key] = []
            groups[key].append(mention)

        deduplicated = []
        for mention_group in groups.values():
            # Keep the mention with highest confidence
            best_mention = max(mention_group, key=lambda m: m.confidence)

            # Merge contexts from all mentions
            all_contexts = []
            for mention in mention_group:
                all_contexts.extend(mention.contexts)

            # Deduplicate contexts
            unique_contexts = []
            seen_positions = set()
            for context in all_contexts:
                pos_key = (context.start_position, context.end_position)
                if pos_key not in seen_positions:
                    unique_contexts.append(context)
                    seen_positions.add(pos_key)

            best_mention.contexts = unique_contexts
            best_mention.mention_count = len(unique_contexts)

            deduplicated.append(best_mention)

        return deduplicated

    async def _enhance_with_sentiment(self, mentions: List[BrandMention], text: str, config: DetectionConfig) -> List[BrandMention]:
        """Enhance mentions with sentiment analysis"""

        enhanced_mentions = []

        for mention in mentions:
            try:
                # Analyze sentiment for each context
                context_sentiments = []

                for context in mention.contexts:
                    sentiment_result = await self.sentiment_analyzer.analyze_sentiment_hybrid(
                        context.text, mention.brand, config.language_code
                    )
                    context_sentiments.append(sentiment_result)

                # Calculate overall sentiment
                if context_sentiments:
                    avg_score = sum(s.score for s in context_sentiments) / len(context_sentiments)

                    # Determine dominant polarity
                    polarity_counts = {}
                    for sentiment in context_sentiments:
                        polarity_counts[sentiment.polarity] = polarity_counts.get(sentiment.polarity, 0) + 1

                    dominant_polarity = max(polarity_counts.keys(), key=lambda k: polarity_counts[k])

                    mention.sentiment_score = avg_score
                    mention.sentiment_polarity = dominant_polarity

                    # Store detailed sentiment data
                    mention.market_specific_data['sentiment_analysis'] = {
                        'individual_contexts': [s.__dict__ for s in context_sentiments],
                        'polarity_distribution': {k.value: v for k, v in polarity_counts.items()}
                    }

                enhanced_mentions.append(mention)

            except Exception as e:
                logger.error(f"Sentiment enhancement failed for mention {mention.brand}: {e}")
                enhanced_mentions.append(mention)  # Keep original mention

        return enhanced_mentions

    def _limit_mentions_per_brand(self, mentions: List[BrandMention], max_per_brand: int) -> List[BrandMention]:
        """Limit number of mentions per brand"""
        brand_counts = {}
        limited_mentions = []

        # Sort by relevance score (best first)
        sorted_mentions = sorted(mentions, key=lambda m: m.calculate_relevance_score(), reverse=True)

        for mention in sorted_mentions:
            brand_count = brand_counts.get(mention.brand, 0)

            if brand_count < max_per_brand:
                limited_mentions.append(mention)
                brand_counts[mention.brand] = brand_count + 1

        return limited_mentions

    def _generate_cache_key(self, text: str, brands: List[str], config: DetectionConfig) -> str:
        """Generate cache key for detection result"""
        import hashlib

        # Create stable hash from inputs
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        brands_str = ",".join(sorted(b.lower() for b in brands))
        config_str = f"{config.confidence_threshold}:{config.similarity_threshold}:{config.market_code}"

        content = f"detect:{config_str}:{brands_str}:{text_hash}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[DetectionResult]:
        """Get cached detection result"""
        cache = get_global_cache()
        if cache:
            return cache.get(cache_key)
        return None

    def _cache_result(self, cache_key: str, result: DetectionResult, ttl: int):
        """Cache detection result"""
        cache = get_global_cache()
        if cache:
            cache.set(cache_key, result, ttl)

    async def detect_brands_batch(self, texts: List[str], target_brands: List[str], config: DetectionConfig = None) -> List[DetectionResult]:
        """Detect brands in multiple texts efficiently"""

        if not texts or not target_brands:
            return []

        detection_config = config or self.config

        # Create detection tasks
        tasks = [
            self.detect_brands(text, target_brands, detection_config)
            for text in texts
        ]

        # Execute in batches to avoid overwhelming the system
        batch_size = 5
        results = []

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, DetectionResult):
                    results.append(result)
                else:
                    logger.error(f"Batch detection failed: {result}")
                    # Add empty result for failed detection
                    results.append(DetectionResult(
                        text_analyzed="",
                        language=detection_config.language_code,
                        processing_time_ms=0,
                        mentions=[],
                        total_brands_found=0,
                        confidence_threshold=detection_config.confidence_threshold,
                        market_adapter_used=detection_config.market_code
                    ))

        return results

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection engine statistics"""
        total = self.detection_stats["total_detections"]
        success_rate = (self.detection_stats["successful_detections"] / total * 100) if total > 0 else 0
        cache_hit_rate = (self.detection_stats["cache_hits"] / (self.detection_stats["cache_hits"] + self.detection_stats["cache_misses"]) * 100) if (self.detection_stats["cache_hits"] + self.detection_stats["cache_misses"]) > 0 else 0

        return {
            "detection_stats": self.detection_stats,
            "success_rate_percent": success_rate,
            "cache_hit_rate_percent": cache_hit_rate,
            "similarity_engine_stats": self.similarity_engine.get_cache_stats(),
            "normalizer_stats": self.normalizer.get_normalization_stats()
        }

    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("Brand detection engine cleaned up")

class BrandDetectionOrchestrator:
    """High-level orchestrator for brand detection operations"""

    def __init__(self, openai_api_key: str):
        self.detection_engine = BrandDetectionEngine(openai_api_key)
        self.configs = {
            "high_precision": DetectionConfig(
                confidence_threshold=0.8,
                similarity_threshold=0.85,
                enable_fuzzy_matching=True,
                enable_semantic_similarity=True,
                max_mentions_per_brand=20
            ),
            "high_recall": DetectionConfig(
                confidence_threshold=0.6,
                similarity_threshold=0.75,
                enable_fuzzy_matching=True,
                enable_semantic_similarity=True,
                max_mentions_per_brand=50
            ),
            "fast": DetectionConfig(
                confidence_threshold=0.7,
                similarity_threshold=0.8,
                enable_fuzzy_matching=True,
                enable_semantic_similarity=False,  # Faster without semantic
                max_mentions_per_brand=30,
                enable_caching=True
            )
        }

    async def detect_with_profile(self, text: str, target_brands: List[str], profile: str = "high_precision", market_code: str = "DE") -> DetectionResult:
        """Detect brands using predefined profile"""

        if profile not in self.configs:
            profile = "high_precision"

        config = self.configs[profile]
        config.market_code = market_code
        config.language_code = "de" if market_code == "DE" else "en"

        return await self.detection_engine.detect_brands(text, target_brands, config)

    async def detect_competitive_analysis(self, text: str, client_brand: str, competitors: List[str], market_code: str = "DE") -> Dict[str, Any]:
        """Perform competitive brand analysis"""

        all_brands = [client_brand] + competitors

        # Use high precision for competitive analysis
        result = await self.detect_with_profile(text, all_brands, "high_precision", market_code)

        # Analyze competitive context
        competitive_analysis = {
            "client_brand": client_brand,
            "competitors": competitors,
            "total_mentions": result.total_brands_found,
            "brand_breakdown": {},
            "competitive_insights": [],
            "sentiment_comparison": {}
        }

        # Break down by brand
        for brand in all_brands:
            brand_mentions = [m for m in result.mentions if m.brand == brand]

            if brand_mentions:
                avg_sentiment = sum(m.sentiment_score for m in brand_mentions) / len(brand_mentions)

                competitive_analysis["brand_breakdown"][brand] = {
                    "mention_count": len(brand_mentions),
                    "average_confidence": sum(m.confidence for m in brand_mentions) / len(brand_mentions),
                    "average_sentiment": avg_sentiment,
                    "detection_methods": list(set(m.detection_method.value for m in brand_mentions)),
                    "contexts": [m.contexts[0].text[:100] + "..." if m.contexts else "" for m in brand_mentions[:3]]
                }

                competitive_analysis["sentiment_comparison"][brand] = avg_sentiment

        # Generate insights
        client_mentions = len([m for m in result.mentions if m.brand == client_brand])
        competitor_mentions = len([m for m in result.mentions if m.brand in competitors])

        if client_mentions + competitor_mentions > 0:
            client_share = client_mentions / (client_mentions + competitor_mentions) * 100
            competitive_analysis["competitive_insights"].append(
                f"{client_brand} has {client_share:.1f}% of brand mentions"
            )

        # Find top competitor
        competitor_counts = {
            comp: len([m for m in result.mentions if m.brand == comp])
            for comp in competitors
        }

        if competitor_counts:
            top_competitor = max(competitor_counts.keys(), key=lambda k: competitor_counts[k])
            competitive_analysis["competitive_insights"].append(
                f"Top competitor: {top_competitor} with {competitor_counts[top_competitor]} mentions"
            )

        return competitive_analysis

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "detection_engine_stats": self.detection_engine.get_detection_statistics(),
            "available_profiles": list(self.configs.keys()),
            "profile_configs": {
                name: {
                    "confidence_threshold": config.confidence_threshold,
                    "similarity_threshold": config.similarity_threshold,
                    "max_mentions_per_brand": config.max_mentions_per_brand,
                    "enabled_methods": [
                        method for method, enabled in [
                            ("fuzzy_matching", config.enable_fuzzy_matching),
                            ("semantic_similarity", config.enable_semantic_similarity)
                        ] if enabled
                    ]
                }
                for name, config in self.configs.items()
            }
        }
EOF
```

## Step 8: Package Integration

```python
# Create: app/services/brand_detection/__init__.py
cat > app/services/brand_detection/__init__.py << 'EOF'
"""
Brand Detection Engine for AEO/GEO Audit Tool

This module provides comprehensive brand detection capabilities optimized for
German and US markets, featuring multi-modal detection methods, advanced
sentiment analysis, and robust error handling.
"""

from .core.detector import (
    BrandDetectionEngine,
    BrandDetectionOrchestrator,
    DetectionConfig
)
from .models.brand_mention import (
    BrandMention,
    BrandContext,
    DetectionResult,
    SentimentPolarity,
    DetectionMethod
)
from .market_adapters.base import MarketAdapterFactory
from .market_adapters.german_adapter import GermanMarketAdapter
from .utils.cache_manager import initialize_global_cache, get_global_cache
from .utils.performance import get_global_monitor

__version__ = "1.0.0"
__author__ = "AEO Audit Tool Team"

# Package-level configuration
DEFAULT_CONFIG = DetectionConfig(
    confidence_threshold=0.7,
    similarity_threshold=0.8,
    market_code="DE",
    language_code="de",
    enable_caching=True,
    cache_ttl=1800
)

# Initialize global components
def initialize_brand_detection(**kwargs):
    """Initialize brand detection system with global components"""

    # Initialize caching
    initialize_global_cache()

    # Register market adapters
    MarketAdapterFactory.register_adapter("DE", GermanMarketAdapter)

    return True

# Convenience functions
def create_detection_engine(openai_api_key: str, config: DetectionConfig = None) -> BrandDetectionEngine:
    """Create a configured brand detection engine"""
    return BrandDetectionEngine(openai_api_key, config or DEFAULT_CONFIG)

def create_orchestrator(openai_api_key: str) -> BrandDetectionOrchestrator:
    """Create a brand detection orchestrator"""
    return BrandDetectionOrchestrator(openai_api_key)

# Export main classes and functions
__all__ = [
    # Core classes
    "BrandDetectionEngine",
    "BrandDetectionOrchestrator",
    "DetectionConfig",

    # Data models
    "BrandMention",
    "BrandContext",
    "DetectionResult",
    "SentimentPolarity",
    "DetectionMethod",

    # Market adapters
    "MarketAdapterFactory",
    "GermanMarketAdapter",

    # Utilities
    "initialize_brand_detection",
    "create_detection_engine",
    "create_orchestrator",
    "get_global_cache",
    "get_global_monitor",

    # Configuration
    "DEFAULT_CONFIG"
]
EOF
```

## Step 9: Basic Tests

```python
# Create: app/services/brand_detection/tests/__init__.py
cat > app/services/brand_detection/tests/__init__.py << 'EOF'
"""Test suite for brand detection engine"""
EOF
```

```python
# Create: app/services/brand_detection/tests/test_basic.py
cat > app/services/brand_detection/tests/test_basic.py << 'EOF'
import pytest
import asyncio
from unittest.mock import Mock, patch
from typing import List

from ..core.detector import BrandDetectionEngine, DetectionConfig
from ..models.brand_mention import BrandMention, DetectionMethod
from ..market_adapters.german_adapter import GermanMarketAdapter

class TestBrandDetectionBasic:
    """Basic tests for brand detection engine"""

    @pytest.fixture
    def detection_config(self):
        """Test configuration"""
        return DetectionConfig(
            confidence_threshold=0.7,
            similarity_threshold=0.8,
            market_code="DE",
            language_code="de",
            enable_caching=False  # Disable for testing
        )

    @pytest.fixture
    def sample_german_text(self):
        """Sample German text for testing"""
        return """
        Salesforce ist eine führende CRM-Software, die von vielen Unternehmen verwendet wird.
        HubSpot bietet auch gute Marketing-Automation-Funktionen.
        Microsoft Dynamics ist eine weitere Alternative.
        """

    @pytest.fixture
    def target_brands(self):
        """Target brands for testing"""
        return ["Salesforce", "HubSpot", "Microsoft Dynamics"]

    def test_detection_config_creation(self, detection_config):
        """Test detection configuration creation"""
        assert detection_config.confidence_threshold == 0.7
        assert detection_config.market_code == "DE"
        assert detection_config.language_code == "de"

    def test_german_adapter_creation(self):
        """Test German market adapter creation"""
        adapter = GermanMarketAdapter()
        assert adapter.market_code == "DE"
        assert adapter.language_code == "de"
        assert "GmbH" in adapter.company_suffixes
        assert "unternehmen" in adapter.business_keywords

    def test_brand_normalization(self):
        """Test brand name normalization"""
        adapter = GermanMarketAdapter()
        variations = adapter.normalize_brand_name("SAP AG")

        assert "SAP AG" in variations
        assert "sap ag" in variations
        assert "SAP" in variations
        assert len(variations) > 3

    def test_german_preprocessing(self):
        """Test German text preprocessing"""
        adapter = GermanMarketAdapter()
        text = "Salesforce  ist   eine    gute  Software."
        processed = adapter.preprocess_text(text)

        # Should normalize whitespace
        assert "  " not in processed
        assert processed.strip() == processed

    def test_brand_mention_creation(self):
        """Test brand mention object creation"""
        mention = BrandMention(
            brand="Salesforce",
            original_text="Salesforce",
            confidence=0.9,
            detection_method=DetectionMethod.FUZZY,
            language="de"
        )

        assert mention.brand == "Salesforce"
        assert mention.confidence == 0.9
        assert mention.detection_method == DetectionMethod.FUZZY
        assert mention.mention_count == 0  # No contexts added yet

    def test_relevance_score_calculation(self):
        """Test relevance score calculation"""
        mention = BrandMention(
            brand="Salesforce",
            original_text="Salesforce",
            confidence=0.8,
            detection_method=DetectionMethod.FUZZY,
            language="de"
        )

        # Base relevance should be close to confidence
        relevance = mention.calculate_relevance_score()
        assert 0.7 <= relevance <= 1.0

    @pytest.mark.asyncio
    async def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        with patch('openai.AsyncOpenAI'):
            engine = BrandDetectionEngine("fake_api_key")

            # Empty text
            result = await engine.detect_brands("", ["Salesforce"])
            assert result.total_brands_found == 0

            # Empty brands list
            result = await engine.detect_brands("Some text", [])
            assert result.total_brands_found == 0

    def test_cache_key_generation(self):
        """Test cache key generation"""
        with patch('openai.AsyncOpenAI'):
            engine = BrandDetectionEngine("fake_api_key")
            config = DetectionConfig()

            key1 = engine._generate_cache_key("test text", ["brand1"], config)
            key2 = engine._generate_cache_key("test text", ["brand1"], config)
            key3 = engine._generate_cache_key("different text", ["brand1"], config)

            # Same inputs should generate same key
            assert key1 == key2
            # Different inputs should generate different keys
            assert key1 != key3

    def test_brand_candidate_extraction(self):
        """Test brand candidate extraction"""
        with patch('openai.AsyncOpenAI'):
            engine = BrandDetectionEngine("fake_api_key")
            adapter = GermanMarketAdapter()

            text = "Salesforce und Microsoft sind beide große Softwareunternehmen."
            candidates = engine._extract_brand_candidates(text, adapter)

            # Should find capitalized words
            candidate_texts = [c['text'] for c in candidates]
            assert "Salesforce" in candidate_texts
            assert "Microsoft" in candidate_texts

    def test_sentence_extraction(self):
        """Test sentence extraction around mentions"""
        with patch('openai.AsyncOpenAI'):
            engine = BrandDetectionEngine("fake_api_key")

            text = "This is first sentence. Salesforce is mentioned here. This is third sentence."
            start = text.find("Salesforce")
            end = start + len("Salesforce")

            sentence = engine._extract_sentence(text, start, end)
            assert "Salesforce is mentioned here" in sentence

    def test_statistics_collection(self):
        """Test statistics collection"""
        with patch('openai.AsyncOpenAI'):
            engine = BrandDetectionEngine("fake_api_key")

            stats = engine.get_detection_statistics()

            assert "detection_stats" in stats
            assert "success_rate_percent" in stats
            assert isinstance(stats["detection_stats"]["total_detections"], int)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF
```

## Step 10: Environment Configuration

```bash
# Create .env file template
cat > .env.template << 'EOF'
# API Keys
OPENAI_API_KEY=your_openai_api_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/aeo_audit

# Redis (optional, will use in-memory cache if not provided)
REDIS_URL=redis://localhost:6379

# Brand Detection Configuration
BRAND_DETECTION_CACHE_TTL=1800
BRAND_DETECTION_MAX_RETRIES=3
BRAND_DETECTION_ENABLE_DEBUG=false

# Logging
LOG_LEVEL=INFO
DEBUG=false
EOF

echo "✅ Created .env template. Copy to .env and fill in your API keys."
```

## Step 11: Basic Usage Example

```python
# Create: example_usage.py
cat > example_usage.py << 'EOF'
#!/usr/bin/env python3
"""
Example usage of the Brand Detection Engine
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the brand detection engine
from app.services.brand_detection import (
    create_detection_engine,
    create_orchestrator,
    DetectionConfig,
    initialize_brand_detection
)

async def main():
    """Main example function"""

    # Initialize the brand detection system
    initialize_brand_detection()

    # Get API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: Please set OPENAI_API_KEY in your .env file")
        return

    # Create detection engine
    print("🚀 Initializing Brand Detection Engine...")
    engine = create_detection_engine(openai_api_key)

    # Sample German text
    sample_text = """
    Salesforce ist eine führende CRM-Software, die von vielen Unternehmen weltweit verwendet wird.
    Die Plattform bietet umfassende Funktionen für Vertrieb, Marketing und Kundenservice.
    HubSpot ist ebenfalls eine beliebte Alternative für mittelständische Unternehmen.
    Microsoft Dynamics 365 integriert sich nahtlos in andere Microsoft-Produkte.
    Viele Kunden sind mit Salesforce sehr zufrieden und empfehlen es weiter.
    """

    # Target brands to detect
    target_brands = ["Salesforce", "HubSpot", "Microsoft Dynamics", "SAP"]

    print(f"📝 Analyzing text (length: {len(sample_text)} characters)")
    print(f"🎯 Looking for brands: {', '.join(target_brands)}")
    print("\n" + "="*50)

    try:
        # Perform brand detection
        result = await engine.detect_brands(
            text=sample_text,
            target_brands=target_brands
        )

        # Display results
        print(f"✅ Detection completed in {result.processing_time_ms:.1f}ms")
        print(f"🔍 Found {result.total_brands_found} brand mentions")
        print(f"🌍 Market: {result.market_adapter_used}")
        print(f"🗣️  Language: {result.language}")

        if result.mentions:
            print("\n📊 Brand Mentions Found:")
            print("-" * 30)

            for mention in result.mentions:
                print(f"🏢 Brand: {mention.brand}")
                print(f"   📄 Text: '{mention.original_text}'")
                print(f"   🎯 Confidence: {mention.confidence:.2f}")
                print(f"   💭 Sentiment: {mention.sentiment_score:.2f} ({mention.sentiment_polarity.value})")
                print(f"   🔧 Method: {mention.detection_method.value}")
                print(f"   📍 Mentions: {mention.mention_count}")

                if mention.contexts:
                    print(f"   📝 Context: '{mention.contexts[0].sentence[:100]}...'")
                print()
        else:
            print("❌ No brand mentions found")

        # Test competitive analysis
        print("\n" + "="*50)
        print("🏆 Running Competitive Analysis...")

        orchestrator = create_orchestrator(openai_api_key)

        competitive_analysis = await orchestrator.detect_competitive_analysis(
            text=sample_text,
            client_brand="Salesforce",
            competitors=["HubSpot", "Microsoft Dynamics"],
            market_code="DE"
        )

        print(f"📈 Client Brand: {competitive_analysis['client_brand']}")
        print(f"🏁 Total Mentions: {competitive_analysis['total_mentions']}")

        if competitive_analysis.get('brand_breakdown'):
            print("\n📊 Brand Breakdown:")
            for brand, data in competitive_analysis['brand_breakdown'].items():
                print(f"   {brand}: {data['mention_count']} mentions")

        if competitive_analysis.get('competitive_insights'):
            print("\n💡 Insights:")
            for insight in competitive_analysis['competitive_insights']:
                print(f"   • {insight}")

        # Display engine statistics
        print("\n" + "="*50)
        print("📈 Engine Statistics:")
        stats = engine.get_detection_statistics()
        print(f"   Success Rate: {stats['success_rate_percent']:.1f}%")
        print(f"   Cache Hit Rate: {stats['cache_hit_rate_percent']:.1f}%")
        print(f"   Total Detections: {stats['detection_stats']['total_detections']}")

    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        engine.cleanup()
        print("\n🧹 Cleanup completed")

if __name__ == "__main__":
    print("Brand Detection Engine - Example Usage")
    print("=" * 40)
    asyncio.run(main())
EOF

chmod +x example_usage.py
```

## Step 12: Final Setup and Testing

```bash
# Install the package in development mode
pip install -e .

# Set up environment
cp .env.template .env
echo "✅ Please edit .env file and add your OpenAI API key"

# Run basic tests
echo "🧪 Running basic tests..."
python -m pytest app/services/brand_detection/tests/test_basic.py -v

# Test the example
echo "🔬 Testing example usage..."
echo "Note: This requires a valid OpenAI API key in .env file"
echo "Run: python example_usage.py"

echo ""
echo "🎉 Brand Detection Engine Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Run: python example_usage.py"
echo "3. Integrate with your main application"
echo ""
echo "Documentation:"
echo "- All code is in app/services/brand_detection/"
echo "- Main classes: BrandDetectionEngine, BrandDetectionOrchestrator"
echo "- German market adapter is included and ready"
echo "- Tests are in app/services/brand_detection/tests/"
echo ""
echo "For production deployment:"
echo "- Set up Redis for caching (optional)"
echo "- Configure logging"
echo "- Set up monitoring"
echo "- Scale horizontally as needed"
```

---

## Summary

This single markdown file contains everything needed to build the complete Brand Detection Engine from scratch:

### ✅ **What's Included:**

1. **Complete file structure creation**
2. **All dependencies and setup**
3. **Core detection engine with multi-modal capabilities**
4. **German market adapter with compound word handling**
5. **Advanced sentiment analysis**
6. **Performance monitoring and caching**
7. **Comprehensive error handling**
8. **Test suite**
9. **Example usage**
10. **Environment configuration**

### 🚀 **To Use This:**

1. **Copy this entire markdown content**
2. **Execute each code block in order in your terminal/Cursor**
3. **Add your OpenAI API key to the .env file**
4. **Run `python example_usage.py` to test**

The system will be fully functional and ready for integration with your AEO audit tool. It's optimized for German market first with extensible architecture for additional markets.
