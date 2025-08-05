import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import List

import structlog

from app.services.providers import (
    ProviderResult,
    Question,
    QuestionContext,
    QuestionProvider,
)

logger = structlog.get_logger(__name__)


class LegacyQuestionCategory(Enum):
    """Categories for audit questions."""

    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    FEATURES = "features"
    PRICING = "pricing"
    REVIEWS = "reviews"
    ALTERNATIVES = "alternatives"


@dataclass
class LegacyQuestionTemplate:
    """Template for generating audit questions."""

    category: LegacyQuestionCategory
    template: str
    variations: List[str]
    industry_specific: bool = False


class TemplateProvider(QuestionProvider):
    """
    A question provider that encapsulates the legacy template-based
    question engine logic.
    This provider contains the original, hardcoded question generation logic.
    """

    def __init__(self):
        """Initializes the TemplateProvider."""
        self._base_templates = [
            LegacyQuestionTemplate(
                category=LegacyQuestionCategory.COMPARISON,
                template="What is the best {industry} {product_type}?",
                variations=[
                    "Which {industry} {product_type} is the best?",
                    "What's the top {industry} {product_type}?",
                    "Best {industry} {product_type} in 2024?",
                ],
            ),
            LegacyQuestionTemplate(
                category=LegacyQuestionCategory.ALTERNATIVES,
                template="What are alternatives to {competitor}?",
                variations=["What are {competitor} competitors?"],
            ),
        ]
        self._industry_patterns = {
            "CRM": [
                "What CRM integrates with Salesforce?",
                "Best CRM for lead management?",
            ]
        }
        logger.info("TemplateProvider initialized")

    @property
    def name(self) -> str:
        """Returns the unique name of the provider."""
        return "template_provider"

    def can_handle(self, ctx: QuestionContext) -> bool:
        """
        Determines if the provider can handle the given context.
        The template provider is always considered capable of handling any context.
        """
        return True

    async def generate(self, ctx: QuestionContext) -> ProviderResult:
        """
        Generates questions using the legacy template engine logic.
        """
        logger.info(
            "Generating questions using TemplateProvider", audit_run_id=ctx.audit_run_id
        )

        loop = asyncio.get_running_loop()

        try:
            questions = await loop.run_in_executor(
                None,
                self._generate_questions_sync,
                ctx.client_brand,
                ctx.competitors,
                ctx.industry,
                ctx.product_type,
            )

            logger.info(
                "Successfully generated questions from TemplateProvider",
                question_count=len(questions),
            )
            return ProviderResult(
                questions=questions,
                metadata={"source": self.name, "question_count": len(questions)},
            )
        except Exception as e:
            logger.error(
                "Error generating questions from TemplateProvider",
                error=str(e),
                exc_info=True,
            )
            return ProviderResult(
                questions=[], metadata={"source": self.name, "error": str(e)}
            )

    def _generate_questions_sync(
        self,
        client_brand: str,
        competitors: List[str],
        industry: str,
        product_type: str,
    ) -> List[Question]:
        """The synchronous part of question generation, based on the old engine."""
        # TODO: Refactor priority weights to be shared with QuestionEngine
        priority_weights = {
            "comparison": 10,
            "recommendation": 9,
            "alternatives": 8,
            "reviews": 7,
            "industry_specific": 7,
            "features": 6,
            "pricing": 5,
        }

        questions = []
        for template in self._base_templates:
            for variation in template.variations:
                score = priority_weights.get(template.category.value, 5)
                if "{industry}" in variation or "{product_type}" in variation:
                    questions.append(
                        Question(
                            question_text=variation.format(
                                industry=industry, product_type=product_type
                            ),
                            category=template.category.value,
                            provider=self.name,
                            priority_score=score,
                        )
                    )
                if "{competitor}" in variation:
                    for competitor in competitors:
                        questions.append(
                            Question(
                                question_text=variation.format(competitor=competitor),
                                category=template.category.value,
                                provider=self.name,
                                priority_score=score,
                            )
                        )

        if industry in self._industry_patterns:
            score = priority_weights.get("industry_specific", 7)
            for question in self._industry_patterns[industry]:
                questions.append(
                    Question(
                        question_text=question,
                        category="industry_specific",
                        provider=self.name,
                        priority_score=score,
                    )
                )

        return questions

    async def health_check(self) -> bool:
        """Performs a health check on the provider."""
        logger.debug("Health check for TemplateProvider: OK")
        return True
