import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CacheKeyGenerator:
    """Generate consistent cache keys for different operations"""

    @staticmethod
    def brand_normalization_key(brand: str, market_code: str) -> str:
        """Generate key for brand normalization cache"""
        content = f"norm:{market_code}:{brand.lower()}"
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def detection_result_key(
        text: str, brands: List[str], market_code: str, confidence_threshold: float
    ) -> str:
        """Generate key for detection result cache"""
        # Create stable hash from input parameters
        brands_str = ",".join(sorted(b.lower() for b in brands))
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        content = (
            f"detect:{market_code}:{confidence_threshold}:{brands_str}:{text_hash}"
        )
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
                "created_at": time.time(),
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
            key
            for key, entry in self.cache.items()
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
            "cache_size": len(self.cache),
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
