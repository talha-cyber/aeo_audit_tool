"""
Central Nervous System for Organic Intelligence.

The brain of the organic intelligence system that coordinates learning,
decision making, and system-wide adaptations.
"""

from .adaptation_controller import AdaptationController
from .central_intelligence import CentralIntelligence, get_central_intelligence
from .decision_engine import DecisionEngine
from .memory_consolidation import MemoryConsolidator
from .pattern_recognition import PatternRecognizer

__all__ = [
    "CentralIntelligence",
    "get_central_intelligence",
    "PatternRecognizer",
    "DecisionEngine",
    "MemoryConsolidator",
    "AdaptationController",
]
