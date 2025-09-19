"""
State-based Stage3 Classification Node
Refactored node that reads embeddings directly from state and uses the clustering service.
"""

import logging
from typing import Dict, Any

from graph.state import GraphState
from .clustering_service import Stage3ClusteringService

logger = logging.getLogger(__name__)


def state_based_stage3_node(state: GraphState) -> GraphState:
    """
    State-based Stage3 classification node that processes embeddings from matched_questions.
    
    Args:
        state: LangGraph state containing matched_questions with embeddings data
        
    Returns:
        Updated GraphState with stage3_* fields populated
    """
    logger.info("Starting state-based Stage3 classification")
    
    try:
        # Extract required data from state
        matched_questions = state.get("matched_questions", {})
        if not matched_questions:
            logger.warning("No matched_questions found in state")
            return {
                **state,
                "stage3_status": "failed",
                "stage3_error": "No matched_questions found in state"
            }
        
        # Get clustering mode from state
        mode = state.get("stage3_mode", "estimate")
        logger.info(f"Processing Stage3 in {mode} mode")
        
        # Extract clustering configuration from state
        config = {
            "mcl_max_time": state.get("stage3_mcl_max_time", 60),
            "mcl_inflation": state.get("stage3_manual_inflation", 2.0),
            "mcl_k": state.get("stage3_manual_k", 15),
            "mcl_max_iters": state.get("stage3_manual_max_iters", 100),
            "auto_train_iterations": state.get("stage3_search_iterations", 10)
        }
        
        # Initialize clustering service
        clustering_service = Stage3ClusteringService(config)
        
        # Process the matched questions
        result = clustering_service.process_matched_questions(matched_questions, mode)
        
        # Update state with results
        updated_state = {**state}
        for key, value in result.items():
            updated_state[key] = value
        
        # Log completion
        status = result.get("stage3_status", "unknown")
        processing_time = result.get("processing_time_seconds", 0)
        logger.info(f"Stage3 node completed with status: {status} in {processing_time:.2f}s")
        
        if status == "failed":
            error = result.get("stage3_error", "Unknown error")
            logger.error(f"Stage3 failed: {error}")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"State-based Stage3 node failed: {str(e)}")
        return {
            **state,
            "stage3_status": "failed",
            "stage3_error": f"Node execution failed: {str(e)}"
        }