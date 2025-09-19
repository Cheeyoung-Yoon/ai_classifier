"""
Stage 3 Classification Node for LangGraph Pipeline.
Integrates trail3 MCL clustering with LangGraph state management.
"""
from typing import Dict, Any
import logging

# Import guards to prevent failures
try:
    from graph.state import GraphState
except ImportError:
    from ...graph.state import GraphState

try:
    from .classification import stage3_classify
except ImportError:
    def stage3_classify(state):
        return {**state, "stage3_status": "failed", "stage3_error": "Classification module not available"}

try:
    from .router import stage3_router
except ImportError:
    def stage3_router(state):
        return {**state, "stage3_status": "failed", "stage3_error": "Router module not available"}

logger = logging.getLogger(__name__)


def stage3_classification_node(state: GraphState) -> GraphState:
    """
    Stage 3 Classification Node for LangGraph.
    
    Performs MCL clustering on Stage 2 embeddings with evaluation capabilities.
    
    Args:
        state: LangGraph state containing matched_questions with stage2 data
        
    Returns:
        Updated state with stage3 classification results
    """
    logger.info("🎯 Starting Stage 3 Classification")
    
    try:
        # Check if we have stage2 data to process
        matched_questions = state.get("matched_questions", {})
        if not matched_questions:
            logger.warning("No matched_questions found in state")
            return {
                **state,
                "stage3_status": "skipped",
                "stage3_error": "No matched_questions found in state"
            }
        
        # Check if stage3 mode is set
        stage3_mode = state.get("stage3_mode", "estimate")
        logger.info(f"Stage 3 mode: {stage3_mode}")
        
        # Route and validate mode
        routed_state = stage3_router(state)
        
        if routed_state.get("stage3_status") == "failed":
            logger.error(f"Stage 3 routing failed: {routed_state.get('stage3_error')}")
            return routed_state
        
        # Perform classification
        result_state = stage3_classify(routed_state)
        
        if result_state.get("stage3_status") == "completed":
            logger.info("✅ Stage 3 Classification completed successfully")
            
            # Log summary information
            summary = result_state.get("stage3_cluster_summary", {})
            n_clusters = summary.get("n_clusters", 0)
            n_samples = summary.get("n_samples", 0)
            
            logger.info(f"Found {n_clusters} clusters from {n_samples} samples")
            
            # Log evaluation results if available
            if "stage3_best_evaluation" in result_state:
                eval_data = result_state["stage3_best_evaluation"]
                if "metrics" in eval_data:
                    metrics = eval_data["metrics"]
                    nmi = metrics.get("nmi", "N/A")
                    ari = metrics.get("ari", "N/A")
                    logger.info(f"Evaluation - NMI: {nmi}, ARI: {ari}")
            
        else:
            logger.error(f"❌ Stage 3 Classification failed: {result_state.get('stage3_error')}")
        
        return result_state
        
    except Exception as e:
        error_msg = f"Stage 3 classification node error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            **state,
            "stage3_status": "failed",
            "stage3_error": error_msg
        }


# Export for LangGraph integration
__all__ = ["stage3_classification_node"]