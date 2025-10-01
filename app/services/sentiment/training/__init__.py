"""
Training module for sentiment analysis domain adaptation.
"""

from .domain_adapter import DomainAdapter, augment_training_data, prepare_business_data

__all__ = ["DomainAdapter", "prepare_business_data", "augment_training_data"]
