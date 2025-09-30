"""
Master control system for organic intelligence features.
"""

from .master_switch import OrganicMasterControl, get_organic_control
from .feature_registry import OrganicFeatureRegistry
from .decorators import organic_enhancement, organic_wrapper, register_organic_feature

__all__ = [
    "OrganicMasterControl",
    "get_organic_control",
    "OrganicFeatureRegistry",
    "organic_enhancement",
    "organic_wrapper",
    "register_organic_feature"
]