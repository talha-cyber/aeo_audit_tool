import uuid
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field


class QuestionContext(BaseModel):
    """Data class holding the context for a question generation request."""

    client_brand: str
    competitors: List[str]
    industry: str
    product_type: str
    audit_run_id: uuid.UUID


class Question(BaseModel):
    """Defines the schema for a single question."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    question_text: str
    category: str
    provider: str
    priority_score: float = 0.0
    cost: Optional[float] = None
    tokens: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = {}


class ProviderResult(BaseModel):
    """Data class for the results returned by a question provider."""

    questions: List[Question]
    metadata: Dict[str, Any]


class QuestionProvider(Protocol):
    """A protocol that defines the interface for a question provider."""

    @property
    def name(self) -> str:
        """The unique name of the provider."""
        ...

    def can_handle(self, ctx: QuestionContext) -> bool:
        """
        Determines if the provider can handle the given context.

        This method allows for dynamically enabling or disabling providers based on the
        audit context, such as industry, product type, or other criteria. It provides a
        hook for more complex routing logic in the future without changing the core
        orchestration in the QuestionEngine. For instance, a specialized legal question
        provider might only activate if the `industry` is 'law'.

        Args:
            ctx: The context for the question generation request.

        Returns:
            True if the provider can handle the context, False otherwise.
        """
        ...

    async def generate(self, ctx: QuestionContext) -> ProviderResult:
        """
        Generates questions based on the provided context.

        This is the core method of the provider, responsible for the actual question
        generation logic. It must be implemented as an async method to allow for
        non-blocking I/O operations, such as making API calls to LLMs or querying
        a database.

        Args:
            ctx: The context for the question generation request.

        Returns:
            A ProviderResult containing the list of generated questions and any
            associated metadata (e.g., cost, latency, models used).
        """
        ...

    async def health_check(self) -> bool:
        """
        Performs a health check on the provider.

        This method is used to determine if the provider is healthy and able to
        service requests. It can be used to check dependencies, such as API
        connectivity or database access. The results of this health check can be
        exposed via a health API endpoint to monitor the status of the system.

        Returns:
            True if the provider is healthy, False otherwise.
        """
        ...
