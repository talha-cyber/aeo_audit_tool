import asyncio
import re
from typing import Any, Dict, List, Set

import structlog
import tenacity
from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletion

from app.core.config import settings
from app.services import metrics
from app.services.providers import (
    ProviderResult,
    Question,
    QuestionContext,
    QuestionProvider,
)
from app.utils.cache import CacheManager

logger = structlog.get_logger(__name__)


class TrendsAdapter:
    """
        Adapter for fetching trend-based seed keywords.

        In a real implementation, this would connect to an external service like
        Google Trends or a social media API. For this build, it provides a set of
        hardcoded fallback seeds to ensure the system is runnable without live
    e   xternal dependencies.
    """

    async def fetch_seeds(self, industry: str, product_type: str) -> List[str]:
        """
        Fetches seed keywords for a given industry and product type.

        Args:
            industry: The industry of the audit.
            product_type: The product type being audited.

        Returns:
            A list of seed keywords.
        """
        logger.info("Fetching seeds from TrendsAdapter (mock)", industry=industry)
        # Mock implementation with fallback seeds
        return [
            f"best {product_type} for {industry}",
            f"{industry} software trends",
            "top features to look for",
            "common problems in",
            "integration challenges",
            "future of",
        ]


class PromptBuilder:
    """
    Builds a deterministic, token-efficient prompt for the LLM.
    """

    def build(self, ctx: QuestionContext, seeds: List[str]) -> str:
        """
        Constructs the LLM prompt.

        Args:
            ctx: The question generation context.
            seeds: A list of seed keywords from the TrendsAdapter.

        Returns:
            A formatted prompt string.
        """
        prompt = f"""
        You are an expert in SEO and competitive analysis for the
        {ctx.industry} industry. Your task is to generate a list of 25
        insightful questions that a potential customer might ask when
        evaluating {ctx.product_type} solutions.

        Client Brand: {ctx.client_brand}
        Competitors: {', '.join(ctx.competitors)}
        Seed Topics: {', '.join(seeds)}

        Based on this context, generate a diverse list of questions
        covering comparisons, features, pricing, and use cases. Each
        question must be a single line. Do not number the questions.
        """
        logger.debug("Built LLM prompt", length=len(prompt))
        return prompt


class LLMClient:
    """
    An asynchronous client for interacting with the OpenAI API.

    This client includes a semaphore for concurrency limiting to manage costs
    and API rate limits, as well as exponential back-off retries for transient
    API errors.
    """

    def __init__(self, api_key: str, model: str, concurrency_limit: int):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._semaphore = asyncio.Semaphore(concurrency_limit)
        logger.info(
            "LLMClient initialized",
            model=model,
            concurrency_limit=concurrency_limit,
        )

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type((RateLimitError, APIError)),
        before_sleep=tenacity.before_sleep_log(logger, "info"),
    )
    async def generate_questions(self, prompt: str, max_tokens: int) -> ChatCompletion:
        """
        Generates questions using the LLM.

        Args:
            prompt: The prompt to send to the LLM.
            max_tokens: The maximum number of tokens for the response.

        Returns:
            The full ChatCompletion object from the OpenAI API.
        """
        async with self._semaphore:
            logger.info("Querying LLM", model=self._model)
            metrics.LLM_CALLS_TOTAL.labels(model=self._model).inc()
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7,
                    timeout=30.0,
                )
                logger.info("LLM query successful", model=self._model)
                return response
            except (RateLimitError, APIError, APITimeoutError) as e:
                logger.warning("LLM API error, will retry", error=str(e))
                metrics.LLM_FAILURES_TOTAL.labels(
                    model=self._model, error_type=type(e).__name__
                ).inc()
                raise
            except Exception as e:
                logger.error(
                    "An unexpected error occurred in LLMClient",
                    error=str(e),
                    exc_info=True,
                )
                metrics.LLM_FAILURES_TOTAL.labels(
                    model=self._model, error_type="unknown"
                ).inc()
                raise

    def _calculate_cost(self, usage: Dict[str, Any]) -> float:
        """Calculates the cost of an LLM call based on token usage."""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        input_cost = (prompt_tokens / 1000) * settings.LLM_INPUT_COST_PER_1K
        output_cost = (completion_tokens / 1000) * settings.LLM_OUTPUT_COST_PER_1K

        return input_cost + output_cost


class PostProcessor:
    """
    Cleans and formats the raw output from the LLM.
    """

    def process(self, raw_text: str, max_questions: int) -> List[Question]:
        """
        Processes the raw LLM output.

        This method performs several cleanup steps:
        1. Splits the text into individual questions.
        2. Strips leading/trailing whitespace and removes numbering.
        3. Converts questions to lowercase for deduplication.
        4. Filters out questions that are too long (more than 15 words).
        5. Tags each question with the "dynamic" category.
        6. Enforces the maximum question limit.

        Args:
            raw_text: The raw text response from the LLM.
            max_questions: The maximum number of questions to return.

        Returns:
            A list of cleaned and formatted Question objects.
        """
        questions: Set[str] = set()

        # Split by newline and filter empty lines
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

        for line in lines:
            # Remove potential list markers like "1. ", "- ", or "* "
            cleaned_line = re.sub(r"^\s*[\d\.\-\*]+\s*", "", line)

            # Normalize by lowercasing and removing trailing question marks
            # for deduplication
            normalized_question = cleaned_line.lower().rstrip("?").strip()

            # Filter out long questions and duplicates
            if (
                normalized_question
                and len(normalized_question.split()) <= 15
                and normalized_question not in questions
            ):
                questions.add(normalized_question)

        # Create a map from normalized to original casing
        original_casing_map = {
            q.lower().rstrip("?").strip(): q
            for q in lines
            if q.lower().rstrip("?").strip() in questions
        }

        final_questions = [
            Question(
                question_text=original_casing_map[q],
                category="dynamic",
                provider="dynamic_provider",
            )
            for q in sorted(list(questions))  # Sort for deterministic output
        ]

        logger.info(
            "Post-processed LLM output",
            initial_count=len(lines),
            final_count=len(final_questions),
        )
        return final_questions[:max_questions]


class DynamicProvider(QuestionProvider):
    """
    A question provider that dynamically generates questions using an LLM.

    This provider orchestrates the dynamic question generation pipeline:
    1. Checks the cache for existing questions.
    2. If cache miss, fetches seed keywords from the TrendsAdapter.
    3. Builds a prompt using the PromptBuilder.
    4. Queries the LLM via the LLMClient.
    5. Cleans the response with the PostProcessor.
    6. Caches the final result.
    """

    def __init__(self):
        self._trends_adapter = TrendsAdapter()
        self._prompt_builder = PromptBuilder()
        self._llm_client = LLMClient(
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL,
            concurrency_limit=settings.LLM_CONCURRENCY,
        )
        self._post_processor = PostProcessor()
        logger.info("DynamicProvider initialized")

    @property
    def name(self) -> str:
        return "dynamic_provider"

    def can_handle(self, ctx: QuestionContext) -> bool:
        """The dynamic provider is enabled or disabled via configuration."""
        return settings.DYNAMIC_Q_ENABLED

    async def generate(self, ctx: QuestionContext) -> ProviderResult:
        """Generates questions dynamically."""
        cache_key = CacheManager.generate_cache_key(
            ctx.industry, ctx.product_type, ctx.competitors
        )

        cached_result = await CacheManager.get(cache_key)
        if cached_result:
            logger.info("DynamicProvider cache hit", key=cache_key)
            # Re-create Pydantic models from cached data
            questions = [Question(**q) for q in cached_result]
            return ProviderResult(
                questions=questions,
                metadata={"source": self.name, "cache_hit": True},
            )

        logger.info(
            "DynamicProvider cache miss, generating new questions", key=cache_key
        )

        seeds = await self._trends_adapter.fetch_seeds(ctx.industry, ctx.product_type)
        prompt = self._prompt_builder.build(ctx, seeds)

        try:
            llm_response = await self._llm_client.generate_questions(
                prompt, max_tokens=1000
            )
            raw_response = llm_response.choices[0].message.content or ""
            usage = llm_response.usage
        except Exception as e:
            logger.error("Failed to get response from LLM", error=str(e), exc_info=True)
            return ProviderResult(
                questions=[],
                metadata={"source": self.name, "error": "LLM API call failed"},
            )

        if not raw_response:
            logger.warning("LLM returned empty response for DynamicProvider")
            return ProviderResult(
                questions=[],
                metadata={"source": self.name, "error": "Empty LLM response"},
            )

        questions = self._post_processor.process(raw_response, settings.DYNAMIC_Q_MAX)

        # Calculate cost and update questions
        cost = self._llm_client._calculate_cost(usage.model_dump())
        metrics.LLM_COST_TOTAL.labels(model=settings.LLM_MODEL).inc(cost)
        for q in questions:
            q.cost = cost / len(questions) if questions else 0
            q.tokens = usage.model_dump()

        # Convert Pydantic models to dicts for JSON serialization in cache
        await CacheManager.set(
            cache_key,
            [q.model_dump() for q in questions],
            ttl=settings.DYNAMIC_Q_CACHE_TTL,
        )

        return ProviderResult(
            questions=questions,
            metadata={
                "source": self.name,
                "cache_hit": False,
                "question_count": len(questions),
                "cost": cost,
            },
        )

    async def health_check(self) -> bool:
        """
        Checks the health of the provider by pinging the OpenAI API.
        This is a basic check. A more robust check might verify model access.
        """
        try:
            # A lightweight way to check API key and connectivity
            await self._llm_client._client.models.list(timeout=10)
            logger.info("DynamicProvider health check successful.")
            return True
        except Exception as e:
            logger.error("DynamicProvider health check failed", error=str(e))
            return False
