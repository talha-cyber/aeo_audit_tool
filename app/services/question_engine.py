"""
Question Engine Module for AEO Competitive Intelligence Tool

Orchestrates multiple question providers to generate a comprehensive and
prioritized list of questions for AI platform audits.
"""

import asyncio
import uuid
from typing import List, Optional

import structlog

from app.services import metrics
from app.services.providers import (
    ProviderResult,
    Question,
    QuestionContext,
    QuestionProvider,
)
from app.services.providers.dynamic_provider import DynamicProvider
from app.services.providers.template_provider import TemplateProvider

logger = structlog.get_logger(__name__)


class QuestionEngine:
    """
    Orchestrates question generation from multiple providers.

    This engine discovers and runs all registered question providers that can
    handle a given context. It uses asyncio.gather to run providers
    concurrently, merges their results, and then prioritizes the final list
    of questions.
    """

    def __init__(self, providers: Optional[List[QuestionProvider]] = None) -> None:
        """
        Initializes the QuestionEngine with a list of providers.

        If no providers are supplied, it defaults to loading the TemplateProvider
        and DynamicProvider.

        Args:
            providers: A list of question provider instances.
        """
        if providers is None:
            self.providers = [TemplateProvider(), DynamicProvider()]
        else:
            self.providers = providers

        logger.info(
            "QuestionEngine initialized",
            provider_count=len(self.providers),
            provider_names=[p.name for p in self.providers],
        )

    async def _safe_generate(
        self, provider: QuestionProvider, ctx: QuestionContext
    ) -> ProviderResult:
        """
        Safely executes a provider's generate method, handling exceptions.

        This wrapper ensures that an error in one provider does not disrupt the
        entire question generation process. Failed providers will return an
        empty result.

        Args:
            provider: The question provider to execute.
            ctx: The context for the question generation request.

        Returns:
            The result from the provider, or an empty ProviderResult on failure.
        """
        try:
            with metrics.PROVIDER_LATENCY_SECONDS.labels(provider.name).time():
                result = await provider.generate(ctx)
            metrics.PROVIDER_CALLS_TOTAL.labels(provider.name).inc()
            return result
        except Exception as e:
            logger.error(
                "Question provider failed during generation",
                provider_name=provider.name,
                error=str(e),
                exc_info=True,
            )
            metrics.PROVIDER_FAILURES_TOTAL.labels(provider.name).inc()
            return ProviderResult(questions=[], metadata={"error": str(e)})

    async def generate_questions(
        self,
        client_brand: str,
        competitors: List[str],
        industry: str,
        product_type: str,
        audit_run_id: uuid.UUID,
        max_questions: int = 100,
    ) -> List[Question]:
        """
        Generates and prioritizes a comprehensive question set from all providers.

        Args:
            client_brand: Name of the client's brand.
            competitors: List of competitor brand names.
            industry: Industry category for the audit.
            product_type: The type of product being audited.
            audit_run_id: The unique ID for the audit run.
            max_questions: The maximum number of questions to return after
                prioritization.

        Returns:
            A list of prioritized Question objects.
        """
        ctx = QuestionContext(
            client_brand=client_brand,
            competitors=competitors,
            industry=industry,
            product_type=product_type,
            audit_run_id=audit_run_id,
        )

        enabled_providers = [p for p in self.providers if p.can_handle(ctx)]
        logger.info(
            "Generating questions with enabled providers",
            enabled_providers=[p.name for p in enabled_providers],
            audit_run_id=audit_run_id,
        )

        # Concurrently run all enabled providers
        provider_results = await asyncio.gather(
            *[self._safe_generate(p, ctx) for p in enabled_providers],
        )

        # Merge questions from all successful provider results
        all_questions = [
            q
            for result in provider_results
            if isinstance(result, ProviderResult) and result.questions
            for q in result.questions
        ]

        logger.info(
            "Total questions generated before prioritization",
            count=len(all_questions),
            audit_run_id=audit_run_id,
        )

        return self.prioritize_questions(all_questions, max_questions)

    def prioritize_questions(
        self, questions: List[Question], max_questions: int
    ) -> List[Question]:
        """
        Prioritizes questions based on strategic value.

        Args:
            questions: List of Question objects from all providers.
            max_questions: Maximum number of questions to return.

        Returns:
            A sorted and truncated list of top-priority questions.
        """
        logger.info(
            "Prioritizing questions",
            input_count=len(questions),
            max_questions=max_questions,
        )

        priority_weights = {
            "comparison": 10,
            "recommendation": 9,
            "alternatives": 8,
            "reviews": 7,
            "industry_specific": 7,
            "features": 6,
            "pricing": 5,
            "dynamic": 8,  # Give dynamic questions a high priority
            "template": 5,  # Lower priority for generic templates
        }

        for question in questions:
            if question.priority_score == 0.0:
                category = question.category

                base_score = priority_weights.get(category, 5)

                question.priority_score = base_score

        sorted_questions = sorted(
            questions, key=lambda x: x.priority_score, reverse=True
        )

        # Deduplicate questions based on the question text, keeping the one
        # with the highest score
        unique_questions = {}
        for q in sorted_questions:
            question_text = q.question_text.lower().strip()
            if question_text not in unique_questions:
                unique_questions[question_text] = q

        final_questions = list(unique_questions.values())

        prioritized = final_questions[:max_questions]

        logger.info("Questions prioritized", output_count=len(prioritized))

        return prioritized
