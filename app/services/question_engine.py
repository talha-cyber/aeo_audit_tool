"""
Question Engine Module for AEO Competitive Intelligence Tool

Generates and prioritizes questions for AI platform audits based on client brand,
competitors, and industry context.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    # Fallback logger for testing without structlog
    import logging
    logger = logging.getLogger(__name__)


class QuestionCategory(Enum):
    """Categories for audit questions."""
    
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    FEATURES = "features"
    PRICING = "pricing"
    REVIEWS = "reviews"
    ALTERNATIVES = "alternatives"


@dataclass
class QuestionTemplate:
    """Template for generating audit questions."""
    
    category: QuestionCategory
    template: str
    variations: List[str]
    industry_specific: bool = False


class QuestionEngine:
    """
    Engine for generating and prioritizing audit questions.
    
    Generates questions by expanding templates with client brand, competitors,
    and industry context. Prioritizes questions based on strategic value.
    """
    
    def __init__(self) -> None:
        """Initialize question engine with base templates and industry patterns."""
        logger.info("Initializing QuestionEngine")
        
        self.base_templates = [
            QuestionTemplate(
                category=QuestionCategory.COMPARISON,
                template="What is the best {industry} software?",
                variations=[
                    "Which {industry} software is the best?",
                    "What's the top {industry} tool?",
                    "Best {industry} software in 2024?",
                    "Leading {industry} solutions?",
                    "Top-rated {industry} platforms?"
                ]
            ),
            QuestionTemplate(
                category=QuestionCategory.RECOMMENDATION,
                template="What {industry} software do you recommend?",
                variations=[
                    "Can you recommend a good {industry} tool?",
                    "What {industry} software should I use?",
                    "Which {industry} platform do you suggest?",
                    "Recommend {industry} software for small business?"
                ]
            ),
            QuestionTemplate(
                category=QuestionCategory.ALTERNATIVES,
                template="What are alternatives to {competitor}?",
                variations=[
                    "What are {competitor} competitors?",
                    "Software similar to {competitor}?",
                    "{competitor} alternatives?",
                    "Competitors of {competitor}?",
                    "Software like {competitor}?"
                ]
            ),
            QuestionTemplate(
                category=QuestionCategory.FEATURES,
                template="What features does {brand} have?",
                variations=[
                    "What can {brand} do?",
                    "{brand} capabilities?",
                    "Features of {brand}?",
                    "What does {brand} offer?",
                    "{brand} functionality?"
                ]
            ),
            QuestionTemplate(
                category=QuestionCategory.PRICING,
                template="How much does {brand} cost?",
                variations=[
                    "What is {brand} pricing?",
                    "{brand} price?",
                    "Cost of {brand}?",
                    "{brand} subscription cost?",
                    "How expensive is {brand}?"
                ]
            ),
            QuestionTemplate(
                category=QuestionCategory.REVIEWS,
                template="What are reviews of {brand}?",
                variations=[
                    "Is {brand} good?",
                    "{brand} reviews and ratings?",
                    "What do users think of {brand}?",
                    "Pros and cons of {brand}?",
                    "{brand} user experience?"
                ]
            )
        ]
        
        # Industry-specific question patterns
        self.industry_patterns = {
            "CRM": [
                "What CRM integrates with Salesforce?",
                "Best CRM for lead management?",
                "Which CRM has the best mobile app?",
                "What CRM works best for small teams?",
                "Best CRM for email marketing integration?"
            ],
            "Marketing Automation": [
                "What marketing automation tool has the best email features?",
                "Which platform is best for drip campaigns?",
                "Best marketing automation for ecommerce?",
                "What marketing tool integrates with CRM?",
                "Best platform for lead nurturing?"
            ],
            "Project Management": [
                "What project management tool is best for teams?",
                "Which PM software has Gantt charts?",
                "Best project management for remote teams?",
                "What PM tool integrates with Slack?",
                "Best project tracking software?"
            ],
            "Analytics": [
                "Best analytics platform for startups?",
                "What analytics tool tracks user behavior?",
                "Which platform has the best dashboards?",
                "Best analytics for ecommerce tracking?",
                "What tool provides real-time analytics?"
            ]
        }
        
        logger.info(
            "QuestionEngine initialized",
            template_count=len(self.base_templates),
            industry_pattern_count=len(self.industry_patterns)
        )
    
    def generate_questions(
        self, 
        client_brand: str, 
        competitors: List[str], 
        industry: str,
        categories: Optional[List[QuestionCategory]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate comprehensive question set for audit.
        
        Args:
            client_brand: Name of the client's brand
            competitors: List of competitor brand names
            industry: Industry category (e.g., "CRM", "Marketing Automation")
            categories: Optional list of question categories to include
            
        Returns:
            List of question dictionaries with metadata
        """
        if categories is None:
            categories = list(QuestionCategory)
        
        logger.info(
            "Generating questions",
            client_brand=client_brand,
            competitor_count=len(competitors),
            industry=industry,
            category_count=len(categories)
        )
        
        questions = []
        
        # Generate from base templates
        for template in self.base_templates:
            if template.category not in categories:
                continue
                
            # Generate variations for each template
            for variation in template.variations:
                question_data = {
                    "category": template.category.value,
                    "template": template.template,
                    "variation": variation,
                    "industry": industry,
                    "client_brand": client_brand,
                    "competitors": competitors
                }
                
                # Generate actual questions
                if "{industry}" in variation:
                    questions.append({
                        **question_data,
                        "question": variation.format(industry=industry),
                        "type": "industry_general"
                    })
                
                if "{brand}" in variation:
                    # Generate for client brand
                    questions.append({
                        **question_data,
                        "question": variation.format(brand=client_brand),
                        "type": "brand_specific",
                        "target_brand": client_brand
                    })
                    
                    # Generate for each competitor
                    for competitor in competitors:
                        questions.append({
                            **question_data,
                            "question": variation.format(brand=competitor),
                            "type": "competitor_specific",
                            "target_brand": competitor
                        })
                
                if "{competitor}" in variation:
                    for competitor in competitors:
                        questions.append({
                            **question_data,
                            "question": variation.format(competitor=competitor),
                            "type": "alternative_seeking",
                            "target_brand": competitor
                        })
        
        # Add industry-specific questions
        if industry in self.industry_patterns:
            for question in self.industry_patterns[industry]:
                questions.append({
                    "category": "industry_specific",
                    "question": question,
                    "type": "industry_specific",
                    "industry": industry,
                    "client_brand": client_brand,
                    "competitors": competitors
                })
        
        logger.info(
            "Questions generated",
            total_questions=len(questions),
            client_brand=client_brand
        )
        
        return questions
    
    def prioritize_questions(
        self, 
        questions: List[Dict[str, Any]], 
        max_questions: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Prioritize questions based on strategic value.
        
        Args:
            questions: List of question dictionaries
            max_questions: Maximum number of questions to return
            
        Returns:
            List of top-priority questions sorted by score
        """
        logger.info(
            "Prioritizing questions",
            input_count=len(questions),
            max_questions=max_questions
        )
        
        # Priority scoring weights
        priority_weights = {
            "comparison": 10,      # High value - direct competitive intelligence
            "recommendation": 9,   # High value - recommendation scenarios
            "alternatives": 8,     # High value - competitor displacement
            "features": 6,         # Medium value - feature positioning
            "pricing": 5,          # Medium value - pricing intelligence
            "reviews": 7,          # Medium-high value - sentiment analysis
            "industry_specific": 7 # Medium-high value - targeted insights
        }
        
        # Score each question
        for question in questions:
            base_score = priority_weights.get(question["category"], 5)
            
            # Boost score for certain question types
            if question["type"] == "industry_general":
                question["priority_score"] = base_score + 2
            elif question["type"] == "alternative_seeking":
                question["priority_score"] = base_score + 1
            else:
                question["priority_score"] = base_score
        
        # Sort by priority and return top questions
        sorted_questions = sorted(
            questions, 
            key=lambda x: x["priority_score"], 
            reverse=True
        )
        prioritized = sorted_questions[:max_questions]
        
        logger.info(
            "Questions prioritized",
            output_count=len(prioritized),
            avg_score=sum(q["priority_score"] for q in prioritized) / len(prioritized) if prioritized else 0
        )
        
        return prioritized 