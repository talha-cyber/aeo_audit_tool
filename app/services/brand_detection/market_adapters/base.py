import re
from abc import ABC, abstractmethod
from typing import Dict, List, Set

from ..models.brand_mention import DetectionMethod


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
    def calculate_market_confidence(
        self, original_text: str, brand: str, context: str
    ) -> float:
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
        text = re.sub(r"\s+", " ", text).strip()

        # Market-specific preprocessing
        return self._market_specific_preprocessing(text)

    @abstractmethod
    def _market_specific_preprocessing(self, text: str) -> str:
        """Market-specific text preprocessing"""
        pass

    def validate_brand_mention(
        self, brand: str, context: str, detection_method: DetectionMethod
    ) -> bool:
        """Validate if brand mention is legitimate"""

        # Basic validation
        if not brand or len(brand.strip()) < 2:
            return False

        # Check for common false positives
        false_positive_patterns = [
            r"\b(the|and|or|but|in|on|at|to|for|of|with|by)\b",
            r"\b\d+\b",  # Pure numbers
            r"^[^\w\s]+$",  # Only special characters
        ]

        for pattern in false_positive_patterns:
            if re.match(pattern, brand.lower().strip()):
                return False

        return self._market_specific_validation(brand, context, detection_method)

    @abstractmethod
    def _market_specific_validation(
        self, brand: str, context: str, detection_method: DetectionMethod
    ) -> bool:
        """Market-specific validation logic"""
        pass

    def get_performance_stats(self) -> Dict[str, any]:
        """Get adapter performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = (
            (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        )

        return {
            "market_code": self.market_code,
            "language_code": self.language_code,
            "cache_hit_rate": cache_hit_rate,
            "total_requests": total_requests,
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
                "ADAPTER_NOT_FOUND",
            )

        try:
            return cls._adapters[market_code]()
        except Exception as e:
            raise MarketAdapterError(
                f"Failed to create adapter: {str(e)}",
                market_code,
                "ADAPTER_CREATION_FAILED",
            )

    @classmethod
    def get_available_markets(cls) -> List[str]:
        """Get list of available market codes"""
        return list(cls._adapters.keys())
