"""
Stage3 Classification Module - Two-Phase Labeling System

New two-phase approach for robust text classification:
- Phase 1: Primary labeling using kNN → CSLS → MCL pipeline with singleton support
- Phase 2: Secondary labeling using graph-based community detection for semantic integration
- Quality assessment with SubsetScore, consistency metrics, and feedback loops

Legacy single-phase nodes are maintained for backward compatibility.
"""

# New Two-Phase System (Recommended)
from .two_phase_stage3_node import (
    stage3_main_node,
    two_phase_stage3_node,
    stage3_router_node
)
from .phase1_primary_labeling import (
    Phase1PrimaryLabeling,
    phase1_primary_labeling_node
)
from .phase2_secondary_labeling import (
    Phase2SecondaryLabeling,
    phase2_secondary_labeling_node
)
from .quality_assessment import (
    QualityAssessmentTools,
    quality_assessment_node
)

# Legacy System (Backward Compatibility)
from .stage3_node import stage3_classification_node
from .singleton_aware_stage3_node import singleton_aware_stage3_node
from .state_based_stage3_node import state_based_stage3_node
from .clustering_service import Stage3ClusteringService
from .singleton_aware_clustering_nmi import SingletonAwareClusteringNMI
from .nmi_ari_evaluation import NMIARIEvaluator

__all__ = [
    # New Two-Phase System - Primary Interface
    'stage3_main_node',           # Main entry point for new system
    'two_phase_stage3_node',      # Complete two-phase pipeline
    'stage3_router_node',         # Router for parameter validation
    
    # Phase-specific Nodes
    'phase1_primary_labeling_node',    # Phase 1: kNN → CSLS → MCL
    'Phase1PrimaryLabeling',           # Phase 1 processor class
    'phase2_secondary_labeling_node',  # Phase 2: Community detection
    'Phase2SecondaryLabeling',         # Phase 2 processor class
    
    # Quality Assessment
    'quality_assessment_node',     # Quality assessment node
    'QualityAssessmentTools',      # Quality assessment toolkit
    
    # Legacy System - Backward Compatibility
    'stage3_classification_node',     # Legacy main node
    'singleton_aware_stage3_node',    # Legacy singleton-aware node
    'state_based_stage3_node',        # Legacy state-based node
    'Stage3ClusteringService',        # Legacy clustering service
    'SingletonAwareClusteringNMI',    # Legacy NMI clustering
    'NMIARIEvaluator'                 # Legacy evaluation tools
]

# Version and metadata
__version__ = "2.0.0"
__description__ = "Two-phase text classification system with quality assessment"

# Default node recommendation
def get_recommended_stage3_node():
    """Get the recommended Stage 3 node for new projects."""
    return stage3_main_node

# Migration helper
def get_legacy_node(node_type: str = "default"):
    """
    Get legacy nodes for backward compatibility.
    
    Args:
        node_type: Type of legacy node ('default', 'singleton_aware', 'state_based')
        
    Returns:
        Legacy node function
    """
    if node_type == "singleton_aware":
        return singleton_aware_stage3_node
    elif node_type == "state_based":
        return state_based_stage3_node
    else:
        return stage3_classification_node