# AI Platform Client Module - Purpose & Context

## 🎯 **Module Purpose**
This module provides a **unified interface for querying multiple AI platforms** (OpenAI, Anthropic, Perplexity, Google AI) as part of an **Answer Engine Optimization (AEO) audit tool**. It abstracts away platform-specific APIs into a consistent interface that the audit system can use reliably.

## 🔄 **How It Fits in the Larger System**
```
Question Engine → AI Platform Client → Brand Detector → Audit Processor
     ↓                    ↓                ↓              ↓
Generates        Queries AI APIs      Analyzes        Orchestrates
questions        concurrently        responses       entire audit
```

**The audit workflow:**
1. **Question Engine** generates ~200 questions about a brand/industry
2. **AI Platform Client** sends these questions to 4+ AI platforms simultaneously
3. **Brand Detector** analyzes AI responses for brand mentions and sentiment
4. **Audit Processor** coordinates everything and generates competitive intelligence reports

## 🎯 **Key Requirements**
- **Reliability**: Handle API failures gracefully (AI APIs fail frequently)
- **Performance**: Query multiple platforms concurrently with proper rate limiting
- **Consistency**: Normalize different response formats into unified output
- **Scalability**: Support hundreds of agencies running thousands of audits

## 🏢 **Business Context**
This is a **white-label SaaS tool for marketing agencies** to analyze how their clients appear in AI-generated responses compared to competitors. Agencies need reliable, fast competitive intelligence reports to advise clients on "Answer Engine Optimization" strategies.

## 🎯 **Success Criteria**
The module succeeds when agencies can run 200-question audits across 4 AI platforms in under 2 hours with <1% error rate, providing reliable competitive intelligence data for client reports.

# AI Platform Client Module — Build Plan

## 1) Purpose & Scope (Single Source of Truth)

The AI Platform Client module provides a unified interface for querying multiple AI platforms (OpenAI, Anthropic, Perplexity, Google AI) with:

* Abstract base class defining common interface
* Rate limiting per platform with configurable limits
* Standardized response format and text extraction
* Robust error handling and retry logic
* Async/await support for concurrent queries
* Platform-specific optimizations and configurations

**Out of scope:** Question generation, brand detection, audit orchestration, database operations.

---

## 2) Inputs, Outputs, Contracts

### Inputs
* `question: str` — user query to send to AI platform
* `platform_config: Dict` — API keys, rate limits, model preferences
* `query_params: Dict` — optional parameters (temperature, max_tokens, etc.)

### Outputs
* `safe_query() -> Dict[str, Any]` — standardized response wrapper:
  ```python
  {
    "success": bool,
    "response": Dict[str, Any],  # Raw platform response
    "platform": str,
    "error": Optional[str],
    "metadata": Dict[str, Any]   # Timing, retries, etc.
  }
  ```
* `extract_text_response(raw_response) -> str` — clean text from platform response

### Error surface
* Network failures: retry with exponential backoff
* Rate limit errors: queue and retry after delay
* API key issues: fail fast with clear error message
* Malformed responses: log warning, return error response
* Platform-specific errors: map to common error types

---

## 3) Dependencies (Must exist before coding here)

* **Settings**: `app.config.settings` with API keys and rate limits
* **Logger**: `app.utils.logger.logger` for structured logging
* **HTTP Client**: `aiohttp` or `httpx` for async requests
* **Rate Limiting**: Built-in `AIRateLimiter` class
* **Error Types**: Custom exception classes for different failure modes

---

## 4) File Layout (create / confirm)

```
app/services/ai_platforms/
  __init__.py
  base.py                  # Abstract base class + rate limiter
  openai_client.py        # OpenAI implementation
  anthropic_client.py     # Anthropic implementation
  perplexity_client.py    # Perplexity implementation
  google_ai_client.py     # Google AI implementation
  exceptions.py           # Platform-specific exceptions
  registry.py            # Platform registry and factory
```

---

## 5) Public API (what other code calls)

```python
# Base interface that all platforms implement
class BasePlatform(ABC):
    def __init__(self, api_key: str, rate_limit: int, **config):
        ...

    @abstractmethod
    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Platform-specific query implementation"""
        ...

    @abstractmethod
    def extract_text_response(self, raw_response: Dict[str, Any]) -> str:
        """Extract clean text from platform response"""
        ...

    async def safe_query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Public interface with error handling and rate limiting"""
        ...

# Factory for creating platform instances
class PlatformRegistry:
    @classmethod
    def create_platform(cls, platform_name: str, config: Dict) -> BasePlatform:
        ...

    @classmethod
    def get_available_platforms(cls) -> List[str]:
        ...
```

---

## 6) Detailed Behavior & Flow

### 6.1 Rate Limiting Strategy

* **Per-platform rate limiting**: Each platform has independent rate limiter
* **Token bucket algorithm**: Smooth rate limiting with burst capability
* **Configurable limits**: RPM (requests per minute) per platform
* **Graceful degradation**: Queue requests when rate limited
* **Sliding window**: Track requests in 60-second rolling window

### 6.2 Error Handling & Retries

* **Retry strategy**: Exponential backoff for transient failures
* **Circuit breaker**: Temporarily disable platform after consecutive failures
* **Error classification**:
  - `TRANSIENT`: Network timeouts, 5xx errors (retry)
  - `RATE_LIMIT`: 429 errors (queue and retry)
  - `AUTH_ERROR`: 401/403 errors (fail fast)
  - `QUOTA_EXCEEDED`: Monthly quota exceeded (fail fast)
  - `MALFORMED`: Invalid response format (log and fail)

### 6.3 Response Normalization

* **Standardized wrapper**: All platforms return same response format
* **Text extraction**: Platform-agnostic method to get response text
* **Metadata preservation**: Keep original response for debugging
* **Error details**: Include platform-specific error information

### 6.4 Configuration Management

```python
# Platform configurations
PLATFORM_CONFIGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4",
        "max_tokens": 500,
        "temperature": 0.1,
        "rate_limit": 50  # RPM
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-3-sonnet-20240229",
        "max_tokens": 500,
        "temperature": 0.1,
        "rate_limit": 100
    },
    # ... other platforms
}
```

---

## 7) Implementation Details

### 7.1 Rate Limiter Implementation

```python
# app/services/ai_platforms/base.py
import asyncio
import time
from collections import deque
from typing import Optional

class AIRateLimiter:
    def __init__(self, requests_per_minute: int, burst_limit: Optional[int] = None):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit or min(requests_per_minute // 4, 10)
        self.requests = deque()
        self.tokens = self.burst_limit
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens for request, blocking if necessary"""
        async with self._lock:
            await self._wait_for_tokens(tokens)
            self._consume_tokens(tokens)
            self.requests.append(time.time())

    async def _wait_for_tokens(self, needed_tokens: int) -> None:
        while self.tokens < needed_tokens:
            await self._refill_tokens()
            if self.tokens < needed_tokens:
                sleep_time = self._calculate_sleep_time()
                await asyncio.sleep(sleep_time)

    def _refill_tokens(self) -> None:
        now = time.time()
        time_passed = now - self.last_refill

        # Refill tokens based on time passed
        tokens_to_add = int(time_passed * (self.requests_per_minute / 60))
        self.tokens = min(self.burst_limit, self.tokens + tokens_to_add)
        self.last_refill = now

        # Clean old requests (older than 1 minute)
        cutoff = now - 60
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    def _calculate_sleep_time(self) -> float:
        """Calculate how long to sleep before next token is available"""
        return 60 / self.requests_per_minute

    def _consume_tokens(self, tokens: int) -> None:
        self.tokens -= tokens
```

### 7.2 Base Platform Class

```python
# app/services/ai_platforms/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import time
import aiohttp
from app.utils.logger import logger
from .exceptions import *

class BasePlatform(ABC):
    def __init__(self, api_key: str, rate_limit: int = 60, **config):
        self.api_key = api_key
        self.rate_limiter = AIRateLimiter(rate_limit)
        self.platform_name = self.__class__.__name__.lower().replace('platform', '')
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_open = False
        self.max_failures = 5
        self.circuit_timeout = 300  # 5 minutes

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=self._get_default_headers()
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @abstractmethod
    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Platform-specific implementation of query"""
        pass

    @abstractmethod
    def extract_text_response(self, raw_response: Dict[str, Any]) -> str:
        """Extract clean text from platform-specific response"""
        pass

    @abstractmethod
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for HTTP requests"""
        pass

    @abstractmethod
    def _prepare_request_payload(self, question: str, **kwargs) -> Dict[str, Any]:
        """Prepare platform-specific request payload"""
        pass

    async def safe_query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Public interface with error handling, rate limiting, and retries"""
        if self._is_circuit_open():
            return self._create_error_response("Circuit breaker open")

        await self.rate_limiter.acquire()

        start_time = time.time()
        retries = 0
        max_retries = 3

        while retries <= max_retries:
            try:
                response = await self._execute_query_with_timeout(question, **kwargs)

                # Reset circuit breaker on success
                self.failure_count = 0
                self.circuit_open = False

                return {
                    "success": True,
                    "response": response,
                    "platform": self.platform_name,
                    "error": None,
                    "metadata": {
                        "duration": time.time() - start_time,
                        "retries": retries,
                        "timestamp": time.time()
                    }
                }

            except RateLimitError as e:
                logger.warning(f"Rate limited by {self.platform_name}: {e}")
                await asyncio.sleep(e.retry_after or 60)
                retries += 1

            except TransientError as e:
                logger.warning(f"Transient error from {self.platform_name}: {e}")
                if retries < max_retries:
                    await asyncio.sleep(2 ** retries)  # Exponential backoff
                retries += 1

            except (AuthenticationError, QuotaExceededError) as e:
                logger.error(f"Non-retryable error from {self.platform_name}: {e}")
                self._record_failure()
                return self._create_error_response(str(e))

            except Exception as e:
                logger.error(f"Unexpected error from {self.platform_name}: {e}")
                self._record_failure()
                if retries >= max_retries:
                    return self._create_error_response(f"Max retries exceeded: {e}")
                retries += 1

        self._record_failure()
        return self._create_error_response("Max retries exceeded")

    async def _execute_query_with_timeout(self, question: str, **kwargs) -> Dict[str, Any]:
        """Execute query with timeout and error handling"""
        if not self.session:
            raise PlatformError("HTTP session not initialized")

        payload = self._prepare_request_payload(question, **kwargs)

        try:
            async with self.session.post(
                self._get_endpoint_url(),
                json=payload
            ) as response:
                response_data = await response.json()

                if response.status == 429:
                    retry_after = int(response.headers.get('retry-after', 60))
                    raise RateLimitError(f"Rate limited", retry_after=retry_after)
                elif response.status == 401:
                    raise AuthenticationError("Invalid API key")
                elif response.status == 403:
                    raise QuotaExceededError("Quota exceeded")
                elif response.status >= 500:
                    raise TransientError(f"Server error: {response.status}")
                elif response.status != 200:
                    raise PlatformError(f"HTTP {response.status}: {response_data}")

                return response_data

        except asyncio.TimeoutError:
            raise TransientError("Request timeout")
        except aiohttp.ClientError as e:
            raise TransientError(f"Network error: {e}")

    def _is_circuit_open(self) -> bool:
        if not self.circuit_open:
            return False

        # Check if circuit timeout has passed
        if time.time() - self.last_failure_time > self.circuit_timeout:
            self.circuit_open = False
            self.failure_count = 0
            logger.info(f"Circuit breaker reset for {self.platform_name}")
            return False

        return True

    def _record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.max_failures:
            self.circuit_open = True
            logger.warning(f"Circuit breaker opened for {self.platform_name}")

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "response": None,
            "platform": self.platform_name,
            "error": error_message,
            "metadata": {"timestamp": time.time()}
        }

    @abstractmethod
    def _get_endpoint_url(self) -> str:
        """Get the API endpoint URL for this platform"""
        pass
```

### 7.3 Platform-Specific Implementations

```python
# app/services/ai_platforms/openai_client.py
from .base import BasePlatform
from typing import Dict, Any

class OpenAIPlatform(BasePlatform):
    def __init__(self, api_key: str, rate_limit: int = 50, **config):
        super().__init__(api_key, rate_limit, **config)
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        self.default_model = config.get('default_model', 'gpt-4')
        self.max_tokens = config.get('max_tokens', 500)
        self.temperature = config.get('temperature', 0.1)

    def _get_default_headers(self) -> Dict[str, str]:
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'AEO-Audit-Tool/1.0'
        }

    def _get_endpoint_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _prepare_request_payload(self, question: str, **kwargs) -> Dict[str, Any]:
        return {
            'model': kwargs.get('model', self.default_model),
            'messages': [
                {'role': 'user', 'content': question}
            ],
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'stream': False
        }

    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """OpenAI-specific query implementation"""
        # This method is called by _execute_query_with_timeout
        # and should not be called directly
        raise NotImplementedError("Use safe_query() instead")

    def extract_text_response(self, raw_response: Dict[str, Any]) -> str:
        """Extract text from OpenAI response format"""
        try:
            return raw_response['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise ValueError(f"Invalid OpenAI response format: {e}")

# app/services/ai_platforms/anthropic_client.py
from .base import BasePlatform
from typing import Dict, Any

class AnthropicPlatform(BasePlatform):
    def __init__(self, api_key: str, rate_limit: int = 100, **config):
        super().__init__(api_key, rate_limit, **config)
        self.base_url = config.get('base_url', 'https://api.anthropic.com')
        self.default_model = config.get('default_model', 'claude-3-sonnet-20240229')
        self.max_tokens = config.get('max_tokens', 500)
        self.temperature = config.get('temperature', 0.1)

    def _get_default_headers(self) -> Dict[str, str]:
        return {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01',
            'User-Agent': 'AEO-Audit-Tool/1.0'
        }

    def _get_endpoint_url(self) -> str:
        return f"{self.base_url}/v1/messages"

    def _prepare_request_payload(self, question: str, **kwargs) -> Dict[str, Any]:
        return {
            'model': kwargs.get('model', self.default_model),
            'messages': [
                {'role': 'user', 'content': question}
            ],
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature)
        }

    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Anthropic-specific query implementation"""
        raise NotImplementedError("Use safe_query() instead")

    def extract_text_response(self, raw_response: Dict[str, Any]) -> str:
        """Extract text from Anthropic response format"""
        try:
            return raw_response['content'][0]['text'].strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise ValueError(f"Invalid Anthropic response format: {e}")

# app/services/ai_platforms/perplexity_client.py
from .base import BasePlatform
from typing import Dict, Any

class PerplexityPlatform(BasePlatform):
    def __init__(self, api_key: str, rate_limit: int = 20, **config):
        super().__init__(api_key, rate_limit, **config)
        self.base_url = config.get('base_url', 'https://api.perplexity.ai')
        self.default_model = config.get('default_model', 'llama-3.1-sonar-small-128k-online')
        self.max_tokens = config.get('max_tokens', 500)
        self.temperature = config.get('temperature', 0.1)

    def _get_default_headers(self) -> Dict[str, str]:
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'AEO-Audit-Tool/1.0'
        }

    def _get_endpoint_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _prepare_request_payload(self, question: str, **kwargs) -> Dict[str, Any]:
        return {
            'model': kwargs.get('model', self.default_model),
            'messages': [
                {'role': 'user', 'content': question}
            ],
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'stream': False
        }

    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Perplexity-specific query implementation"""
        raise NotImplementedError("Use safe_query() instead")

    def extract_text_response(self, raw_response: Dict[str, Any]) -> str:
        """Extract text from Perplexity response format"""
        try:
            return raw_response['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise ValueError(f"Invalid Perplexity response format: {e}")

# app/services/ai_platforms/google_ai_client.py
from .base import BasePlatform
from typing import Dict, Any

class GoogleAIPlatform(BasePlatform):
    def __init__(self, api_key: str, rate_limit: int = 60, **config):
        super().__init__(api_key, rate_limit, **config)
        self.base_url = config.get('base_url', 'https://generativelanguage.googleapis.com')
        self.default_model = config.get('default_model', 'gemini-pro')
        self.max_tokens = config.get('max_tokens', 500)
        self.temperature = config.get('temperature', 0.1)

    def _get_default_headers(self) -> Dict[str, str]:
        return {
            'Content-Type': 'application/json',
            'User-Agent': 'AEO-Audit-Tool/1.0'
        }

    def _get_endpoint_url(self) -> str:
        return f"{self.base_url}/v1/models/{self.default_model}:generateContent?key={self.api_key}"

    def _prepare_request_payload(self, question: str, **kwargs) -> Dict[str, Any]:
        return {
            'contents': [{
                'parts': [{'text': question}]
            }],
            'generationConfig': {
                'maxOutputTokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature)
            }
        }

    async def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Google AI-specific query implementation"""
        raise NotImplementedError("Use safe_query() instead")

    def extract_text_response(self, raw_response: Dict[str, Any]) -> str:
        """Extract text from Google AI response format"""
        try:
            return raw_response['candidates'][0]['content']['parts'][0]['text'].strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise ValueError(f"Invalid Google AI response format: {e}")
```

### 7.4 Exception Classes

```python
# app/services/ai_platforms/exceptions.py

class PlatformError(Exception):
    """Base exception for all platform errors"""
    pass

class TransientError(PlatformError):
    """Temporary error that should be retried"""
    pass

class RateLimitError(PlatformError):
    """Rate limit exceeded error"""
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after

class AuthenticationError(PlatformError):
    """Authentication/API key error"""
    pass

class QuotaExceededError(PlatformError):
    """Monthly quota exceeded"""
    pass

class MalformedResponseError(PlatformError):
    """Response format is invalid or unexpected"""
    pass
```

### 7.5 Platform Registry

```python
# app/services/ai_platforms/registry.py
from typing import Dict, Type, List
from .base import BasePlatform
from .openai_client import OpenAIPlatform
from .anthropic_client import AnthropicPlatform
from .perplexity_client import PerplexityPlatform
from .google_ai_client import GoogleAIPlatform

class PlatformRegistry:
    """Factory for creating and managing AI platform instances"""

    _platforms: Dict[str, Type[BasePlatform]] = {
        'openai': OpenAIPlatform,
        'anthropic': AnthropicPlatform,
        'perplexity': PerplexityPlatform,
        'google_ai': GoogleAIPlatform
    }

    @classmethod
    def create_platform(cls, platform_name: str, api_key: str, config: Dict = None) -> BasePlatform:
        """Create a platform instance by name"""
        if platform_name not in cls._platforms:
            raise ValueError(f"Unknown platform: {platform_name}")

        platform_class = cls._platforms[platform_name]
        platform_config = config or {}

        return platform_class(api_key=api_key, **platform_config)

    @classmethod
    def get_available_platforms(cls) -> List[str]:
        """Get list of available platform names"""
        return list(cls._platforms.keys())

    @classmethod
    def register_platform(cls, name: str, platform_class: Type[BasePlatform]) -> None:
        """Register a new platform implementation"""
        if not issubclass(platform_class, BasePlatform):
            raise TypeError(f"Platform class must inherit from BasePlatform")

        cls._platforms[name] = platform_class
```

---

## 8) Configuration Management

```python
# app/config/platform_settings.py
from typing import Dict, Any

PLATFORM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4",
        "max_tokens": 500,
        "temperature": 0.1,
        "rate_limit": 50,  # RPM
        "timeout": 30,
        "max_retries": 3
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-3-sonnet-20240229",
        "max_tokens": 500,
        "temperature": 0.1,
        "rate_limit": 100,
        "timeout": 30,
        "max_retries": 3
    },
    "perplexity": {
        "base_url": "https://api.perplexity.ai",
        "default_model": "llama-3.1-sonar-small-128k-online",
        "max_tokens": 500,
        "temperature": 0.1,
        "rate_limit": 20,
        "timeout": 45,  # Perplexity can be slower
        "max_retries": 2
    },
    "google_ai": {
        "base_url": "https://generativelanguage.googleapis.com",
        "default_model": "gemini-pro",
        "max_tokens": 500,
        "temperature": 0.1,
        "rate_limit": 60,
        "timeout": 30,
        "max_retries": 3
    }
}

# Environment variable mapping
REQUIRED_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "google_ai": "GOOGLE_AI_API_KEY"
}
```

---

## 9) Logging & Monitoring

```python
# Structured logging format for all platform operations
logger.info("Platform query started", extra={
    "platform": self.platform_name,
    "question_length": len(question),
    "operation": "query"
})

logger.warning("Rate limit hit", extra={
    "platform": self.platform_name,
    "retry_after": retry_after,
    "operation": "rate_limit"
})

logger.error("Platform query failed", extra={
    "platform": self.platform_name,
    "error_type": type(e).__name__,
    "error_message": str(e),
    "retries": retries,
    "operation": "query_error"
})
```

**Metrics to track:**
* Request latency per platform
* Success/failure rates per platform
* Rate limit hit frequency
* Circuit breaker activations
* Token usage per platform (if available)

---

## 10) Testing Strategy

### 10.1 Unit Tests

```python
# tests/test_rate_limiter.py
import pytest
import asyncio
import time
from app.services.ai_platforms.base import AIRateLimiter

@pytest.mark.asyncio
async def test_rate_limiter_basic():
    limiter = AIRateLimiter(requests_per_minute=60)  # 1 per second

    start_time = time.time()
    await limiter.acquire()
    await limiter.acquire()
    await limiter.acquire()
    end_time = time.time()

    # Should take at least 2 seconds for 3 requests
    assert end_time - start_time >= 2.0

@pytest.mark.asyncio
async def test_rate_limiter_burst():
    limiter = AIRateLimiter(requests_per_minute=60, burst_limit=5)

    start_time = time.time()
    # First 5 requests should be immediate (burst)
    for _ in range(5):
        await limiter.acquire()
    mid_time = time.time()

    # 6th request should be delayed
    await limiter.acquire()
    end_time = time.time()

    assert mid_time - start_time < 0.1  # Burst was immediate
    assert end_time - mid_time >= 0.9   # 6th request was delayed

# tests/test_platform_base.py
import pytest
from unittest.mock import AsyncMock, Mock, patch
from app.services.ai_platforms.base import BasePlatform
from app.services.ai_platforms.exceptions import *

class MockPlatform(BasePlatform):
    def _get_default_headers(self):
        return {"Authorization": "Bearer test"}

    def _get_endpoint_url(self):
        return "https://api.test.com/v1/chat"

    def _prepare_request_payload(self, question, **kwargs):
        return {"input": question}

    async def query(self, question, **kwargs):
        return {"output": f"Response to: {question}"}

    def extract_text_response(self, raw_response):
        return raw_response["output"]

@pytest.mark.asyncio
async def test_successful_query():
    platform = MockPlatform("test_key", 60)

    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"output": "Test response"}
        mock_post.return_value.__aenter__.return_value = mock_response

        async with platform:
            result = await platform.safe_query("Test question")

        assert result["success"] is True
        assert result["platform"] == "mock"
        assert "response" in result

@pytest.mark.asyncio
async def test_rate_limit_error():
    platform = MockPlatform("test_key", 60)

    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {"retry-after": "60"}
        mock_post.return_value.__aenter__.return_value = mock_response

        async with platform:
            result = await platform.safe_query("Test question")

        assert result["success"] is False
        assert "rate limit" in result["error"].lower()

@pytest.mark.asyncio
async def test_circuit_breaker():
    platform = MockPlatform("test_key", 60)
    platform.max_failures = 2

    with patch('aiohttp.ClientSession.post') as mock_post:
        # Simulate multiple failures
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        mock_post.return_value.__aenter__.return_value = mock_response

        async with platform:
            # First two failures
            await platform.safe_query("Test question")
            await platform.safe_query("Test question")

            # Third request should be circuit-broken
            result = await platform.safe_query("Test question")

        assert result["success"] is False
        assert "circuit breaker" in result["error"].lower()

# tests/test_platform_implementations.py
import pytest
from app.services.ai_platforms.openai_client import OpenAIPlatform
from app.services.ai_platforms.anthropic_client import AnthropicPlatform

def test_openai_text_extraction():
    platform = OpenAIPlatform("test_key")

    response = {
        "choices": [{
            "message": {
                "content": "This is a test response"
            }
        }]
    }

    text = platform.extract_text_response(response)
    assert text == "This is a test response"

def test_openai_malformed_response():
    platform = OpenAIPlatform("test_key")

    response = {"invalid": "format"}

    with pytest.raises(ValueError):
        platform.extract_text_response(response)

def test_anthropic_text_extraction():
    platform = AnthropicPlatform("test_key")

    response = {
        "content": [{
            "text": "This is a Claude response"
        }]
    }

    text = platform.extract_text_response(response)
    assert text == "This is a Claude response"

# tests/test_platform_registry.py
import pytest
from app.services.ai_platforms.registry import PlatformRegistry
from app.services.ai_platforms.openai_client import OpenAIPlatform

def test_create_platform():
    platform = PlatformRegistry.create_platform("openai", "test_key")
    assert isinstance(platform, OpenAIPlatform)
    assert platform.api_key == "test_key"

def test_unknown_platform():
    with pytest.raises(ValueError):
        PlatformRegistry.create_platform("unknown", "test_key")

def test_available_platforms():
    platforms = PlatformRegistry.get_available_platforms()
    assert "openai" in platforms
    assert "anthropic" in platforms
    assert len(platforms) >= 4
```

### 10.2 Integration Tests

```python
# tests/test_platform_integration.py
import pytest
import os
from app.services.ai_platforms.registry import PlatformRegistry

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key")
@pytest.mark.asyncio
async def test_openai_real_query():
    platform = PlatformRegistry.create_platform(
        "openai",
        os.getenv("OPENAI_API_KEY"),
        {"default_model": "gpt-3.5-turbo"}  # Use cheaper model for tests
    )

    async with platform:
        result = await platform.safe_query("What is 2+2?")

    assert result["success"] is True
    assert "response" in result
    text = platform.extract_text_response(result["response"])
    assert "4" in text

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No Anthropic API key")
@pytest.mark.asyncio
async def test_anthropic_real_query():
    platform = PlatformRegistry.create_platform(
        "anthropic",
        os.getenv("ANTHROPIC_API_KEY")
    )

    async with platform:
        result = await platform.safe_query("What is the capital of France?")

    assert result["success"] is True
    text = platform.extract_text_response(result["response"])
    assert "Paris" in text
```

### 10.3 Performance Tests

```python
# tests/test_platform_performance.py
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch
from app.services.ai_platforms.openai_client import OpenAIPlatform

@pytest.mark.asyncio
async def test_concurrent_queries():
    platform = OpenAIPlatform("test_key", rate_limit=120)  # 2 per second

    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        async with platform:
            start_time = time.time()

            # Execute 10 concurrent queries
            tasks = [
                platform.safe_query(f"Question {i}")
                for i in range(10)
            ]
            results = await asyncio.gather(*tasks)

            end_time = time.time()

        # All should succeed
        assert all(r["success"] for r in results)

        # Should take at least 4.5 seconds due to rate limiting (10 requests at 2/sec)
        assert end_time - start_time >= 4.0

@pytest.mark.asyncio
async def test_platform_timeout():
    platform = OpenAIPlatform("test_key")

    with patch('aiohttp.ClientSession.post') as mock_post:
        # Simulate timeout
        mock_post.side_effect = asyncio.TimeoutError()

        async with platform:
            start_time = time.time()
            result = await platform.safe_query("Test question")
            end_time = time.time()

        assert result["success"] is False
        assert "timeout" in result["error"].lower()
        # Should fail quickly and retry with backoff
        assert end_time - start_time < 10  # But not too long
```

---

## 11) Integration with Main Application

```python
# app/services/platform_manager.py
from typing import Dict, List
from app.config.settings import settings
from app.config.platform_settings import PLATFORM_CONFIGS, REQUIRED_ENV_VARS
from app.services.ai_platforms.registry import PlatformRegistry
from app.services.ai_platforms.base import BasePlatform
from app.utils.logger import logger

class PlatformManager:
    """Manages multiple AI platform instances for the application"""

    def __init__(self):
        self.platforms: Dict[str, BasePlatform] = {}
        self._initialize_platforms()

    def _initialize_platforms(self) -> None:
        """Initialize all configured platforms"""
        for platform_name, config in PLATFORM_CONFIGS.items():
            try:
                api_key_env = REQUIRED_ENV_VARS[platform_name]
                api_key = getattr(settings, api_key_env, None)

                if not api_key:
                    logger.warning(f"No API key found for {platform_name}, skipping")
                    continue

                platform = PlatformRegistry.create_platform(
                    platform_name,
                    api_key,
                    config
                )

                self.platforms[platform_name] = platform
                logger.info(f"Initialized platform: {platform_name}")

            except Exception as e:
                logger.error(f"Failed to initialize {platform_name}: {e}")

    def get_platform(self, name: str) -> BasePlatform:
        """Get a specific platform instance"""
        if name not in self.platforms:
            raise ValueError(f"Platform '{name}' not available")
        return self.platforms[name]

    def get_available_platforms(self) -> List[str]:
        """Get list of available platform names"""
        return list(self.platforms.keys())

    def is_platform_available(self, name: str) -> bool:
        """Check if a platform is available"""
        return name in self.platforms

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all platforms"""
        health_status = {}

        for name, platform in self.platforms.items():
            try:
                async with platform:
                    result = await platform.safe_query("Health check")
                health_status[name] = result["success"]
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                health_status[name] = False

        return health_status

# Usage in audit processor:
# app/services/audit_processor.py (updated integration)
from app.services.platform_manager import PlatformManager

class AuditProcessor:
    def __init__(self, db: Session):
        self.db = db
        self.question_engine = QuestionEngine()
        self.brand_detector = BrandDetector()
        self.platform_manager = PlatformManager()

    def register_platform(self, name: str, platform: BasePlatform) -> None:
        """Register additional platform (for testing or custom platforms)"""
        self.platform_manager.platforms[name] = platform

    async def run_audit(self, audit_config_id: uuid.UUID) -> uuid.UUID:
        # ... existing code ...

        # Use platform manager instead of direct platform access
        usable_platforms = [
            p for p in selected_platforms
            if self.platform_manager.is_platform_available(p)
        ]

        # ... rest of implementation ...

    async def _process_single_question(self, ...):
        platform = self.platform_manager.get_platform(platform_name)

        async with platform:
            resp_wrap = await platform.safe_query(question)
            # ... rest of processing ...
```

---

## 12) Acceptance Criteria

### Core Functionality
* ✅ All 4 platforms (OpenAI, Anthropic, Perplexity, Google AI) implement unified interface
* ✅ Rate limiting works correctly for each platform with different limits
* ✅ Circuit breaker prevents cascading failures
* ✅ Retry logic handles transient failures with exponential backoff
* ✅ Response text extraction works for all platform response formats
* ✅ Error handling distinguishes between retryable and non-retryable errors

### Performance Requirements
* ✅ Rate limiter allows burst requests up to configured limit
* ✅ Concurrent queries work without race conditions
* ✅ Platform initialization takes < 1 second
* ✅ Circuit breaker recovers automatically after timeout period
* ✅ Memory usage remains stable during long-running operations

### Error Handling
* ✅ Authentication errors fail fast with clear messages
* ✅ Rate limit errors automatically retry after delay
* ✅ Network timeouts retry with exponential backoff
* ✅ Malformed responses log warning and return error response
* ✅ Circuit breaker opens after consecutive failures

### Integration
* ✅ Platform manager correctly initializes all available platforms
* ✅ Health check endpoint works for all platforms
* ✅ Audit processor can use platforms through manager
* ✅ Configuration loading works from environment variables
* ✅ Logging provides structured output for monitoring

### Testing
* ✅ Unit tests cover rate limiting, circuit breaker, error handling
* ✅ Integration tests work with real API keys (when available)
* ✅ Performance tests validate concurrent query handling
* ✅ Mock tests cover all error scenarios
* ✅ Test coverage > 90% for all platform client code

---

## 13) AI Coding Agent Task List

1. **Create directory structure** in `app/services/ai_platforms/`
2. **Implement base.py** with `AIRateLimiter` and `BasePlatform` classes
3. **Create exceptions.py** with all error types
4. **Implement OpenAI client** in `openai_client.py`
5. **Implement Anthropic client** in `anthropic_client.py`
6. **Implement Perplexity client** in `perplexity_client.py`
7. **Implement Google AI client** in `google_ai_client.py`
8. **Create registry.py** with `PlatformRegistry` factory
9. **Add platform_settings.py** to config with all platform configurations
10. **Create platform_manager.py** for application integration
11. **Write comprehensive unit tests** for rate limiter and base platform
12. **Write platform-specific tests** for each implementation
13. **Write integration tests** for platform manager
14. **Add performance tests** for concurrent usage
15. **Update requirements.txt** with `aiohttp` dependency
16. **Verify all imports and type hints** are correct
17. **Run test suite** and ensure 90%+ coverage
18. **Test integration** with audit processor module

**Dependencies to install:**
```
aiohttp>=3.8.0
pytest-asyncio>=0.21.0
```

This build plan provides complete implementation details for a production-ready AI Platform Client module that can handle multiple AI platforms with robust error handling, rate limiting, and monitoring capabilities.
