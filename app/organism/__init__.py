"""
Organic Intelligence System for AEO Audit Tool.

A biological-inspired self-healing, learning, and adaptive system that can be
completely disabled with a single command while preserving all normal operations.
"""

from .control.master_switch import OrganicMasterControl, get_organic_control
from .control.decorators import organic_enhancement, organic_wrapper, register_organic_feature

__all__ = [
    "OrganicMasterControl",
    "get_organic_control",
    "organic_enhancement",
    "organic_wrapper",
    "register_organic_feature"
]