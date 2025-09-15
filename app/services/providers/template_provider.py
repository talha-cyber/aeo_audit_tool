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
        """Initializes the TemplateProvider with localized templates."""
        # Keep minimal legacy templates; expanded questions are generated below
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
        # Lightweight industry patterns remain as a fallback
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
                ctx.language,
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
        language: str = "en",
    ) -> List[Question]:
        """The synchronous part of question generation, based on the old engine."""
        from app.services.providers.industry_knowledge import IndustryKnowledge

        def t(en: str, de: str) -> str:
            return en if language == "en" else de

        # Seed integrations/compliance
        integrations = IndustryKnowledge.integrations(product_type)
        compliance = IndustryKnowledge.compliance(industry)

        questions: List[Question] = []

        def add(qtext: str, category: str, score: float, subcat: str | None = None):
            questions.append(
                Question(
                    question_text=qtext,
                    category=category,
                    provider=self.name,
                    priority_score=score,
                    metadata={
                        "language": language,
                        **({"sub_category": subcat} if subcat else {}),
                    },
                )
            )

        # Baseline legacy variations (localized slightly)
        legacy_score = 6
        for template in self._base_templates:
            for variation in template.variations:
                if "{industry}" in variation or "{product_type}" in variation:
                    q = variation.format(industry=industry, product_type=product_type)
                    add(q, template.category.value, legacy_score)
                if "{competitor}" in variation:
                    for c in competitors:
                        q = variation.format(competitor=c)
                        add(q, template.category.value, legacy_score)

        # Pairwise comparisons (client vs each competitor)
        for c in competitors:
            add(
                t(
                    f"{client_brand} vs {c}: which is better?",
                    f"{client_brand} vs. {c}: Was ist besser?",
                ),
                "comparison",
                10,
                subcat="comparison",
            )

        # Pricing per brand
        add(
            t(
                f"How much does {client_brand} cost?",
                f"Wie viel kostet {client_brand}?",
            ),
            "pricing",
            9,
            "pricing",
        )
        for c in competitors:
            add(
                t(f"{c} pricing and tiers?", f"{c} Preise und Tarife?"),
                "pricing",
                9,
                "pricing",
            )

        # Integrations
        for integ in integrations[:4]:
            add(
                t(
                    f"Does {client_brand} integrate with {integ}?",
                    f"Integriert sich {client_brand} mit {integ}?",
                ),
                "integrations",
                9,
                "integrations",
            )

        # Security / Compliance
        for std in compliance[:3]:
            add(
                t(
                    f"Is {client_brand} {std} compliant?",
                    f"Erfüllt {client_brand} {std}?",
                ),
                "security_compliance",
                9,
                "security_compliance",
            )

        # Implementation / Migration
        add(
            t(
                f"How long does {client_brand} take to implement?",
                f"Wie lange dauert die Implementierung von {client_brand}?",
            ),
            "implementation_migration",
            8,
            "implementation_migration",
        )
        for c in competitors[:2]:
            add(
                t(
                    f"Migrate from {c} to {client_brand}: steps?",
                    f"Migration von {c} zu {client_brand}: Schritte?",
                ),
                "implementation_migration",
                8,
                "implementation_migration",
            )

        # ROI / TCO
        add(
            t(
                f"{client_brand} total cost of ownership?",
                f"Gesamtkosten (TCO) von {client_brand}?",
            ),
            "roi_tco",
            8,
            "roi_tco",
        )

        # Support / Reliability
        add(
            t(
                f"{client_brand} SLA and uptime?",
                f"{client_brand} SLA und Verfügbarkeit?",
            ),
            "support_reliability",
            8,
            "support_reliability",
        )

        # Features
        add(
            t(
                f"Top features of {client_brand}?",
                f"Top-Funktionen von {client_brand}?",
            ),
            "features",
            6,
            "features",
        )

        # Reviews
        add(
            t(
                f"{client_brand} reviews and ratings?",
                f"{client_brand} Bewertungen und Rezensionen?",
            ),
            "reviews",
            7,
            "reviews",
        )

        # Geography
        add(
            t(
                f"Is {client_brand} available in the EU?",
                f"Ist {client_brand} in der EU verfügbar?",
            ),
            "geography",
            6,
            "geography",
        )

        # Industry-specific fallback
        if industry in self._industry_patterns:
            for q in self._industry_patterns[industry]:
                add(q, "industry_specific", 7, "industry_specific")

        return questions

    async def health_check(self) -> bool:
        """Performs a health check on the provider."""
        logger.debug("Health check for TemplateProvider: OK")
        return True
