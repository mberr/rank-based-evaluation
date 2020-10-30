# coding=utf-8
"""Entity Alignment Models."""
from .matching import GCNAlign, GraphBasedKGMatchingModel, KGMatchingModel, PureEmbeddingModel, get_matching_model_by_name

__all__ = [
    'GraphBasedKGMatchingModel',
    'GCNAlign',
    'KGMatchingModel',
    'PureEmbeddingModel',
    'get_matching_model_by_name',
]
