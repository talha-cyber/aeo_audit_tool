"""
Master control system for organic intelligence features.
"""

from .decorators import organic_enhancement, organic_wrapper, register_organic_feature
from .feature_registry import OrganicFeatureRegistry
from .master_switch import OrganicMasterControl, get_organic_control

__all__ = [
    "OrganicMasterControl",
    "get_organic_control",
    "OrganicFeatureRegistry",
    "organic_enhancement",
    "organic_wrapper",
    "register_organic_feature",
]
