"""
Stage3 Classification Module with NMI/ARI Evaluation

Production-ready singleton-aware clustering system for LangGraph pipelines.
Uses NMI/ARI evaluation metrics for robust clustering quality assessment.
"""

from .stage3_node import stage3_classification_node
from .singleton_aware_stage3_node import singleton_aware_stage3_node
from .state_based_stage3_node import state_based_stage3_node
from .clustering_service import Stage3ClusteringService
from .singleton_aware_clustering_nmi import SingletonAwareClusteringNMI
from .nmi_ari_evaluation import NMIARIEvaluator

__all__ = [
    'stage3_classification_node',
    'singleton_aware_stage3_node', 
    'state_based_stage3_node',
    'Stage3ClusteringService',
    'SingletonAwareClusteringNMI',
    'NMIARIEvaluator'
]