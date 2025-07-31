from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePlatform(ABC):
    @abstractmethod
    async def query(self, question: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute query and return standardized response"""
        pass

    @abstractmethod
    def extract_text_response(self, raw_response: Dict[str, Any]) -> str:
        """Extract clean text from platform-specific response format"""
        pass
