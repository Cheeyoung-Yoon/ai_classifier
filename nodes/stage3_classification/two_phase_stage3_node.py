"""
Stage 3 Two-Phase Classification Node
Main orchestrator for the new two-phase labeling approach.
"""

import logging
from typing import Dict, Any

from .phase1_primary_labeling import phase1_primary_labeling_node
from .phase2_secondary_labeling import phase2_secondary_labeling_node
from .quality_assessment import quality_assessment_node

logger = logging.getLogger(__name__)


def two_phase_stage3_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main Stage 3 node implementing two-phase labeling approach.
    
    Flow:
    1. Phase 1: Primary Labeling (kNN → CSLS → MCL with singletons)
    2. Phase 2: Secondary Labeling (Graph-based community detection)
    3. Quality Assessment: Comprehensive quality metrics and feedback
    
    Args:
        state: LangGraph state containing matched_questions with embeddings
        
    Returns:
        Updated state with complete Stage 3 results
    """
    logger.info("🚀 Starting Stage 3: Two-Phase Classification")
    
    try:
        # Check prerequisites
        matched_questions = state.get("matched_questions", {})
        if not matched_questions:
            return {
                **state,
                "stage3_status": "failed",
                "stage3_error": "No matched_questions found in state"
            }
        
        # Initialize Stage 3 status
        state["stage3_status"] = "in_progress"
        state["stage3_current_phase"] = "starting"
        
        # Phase 1: Primary Labeling
        logger.info("Executing Phase 1: Primary Labeling...")
        state = phase1_primary_labeling_node(state)
        
        if state.get("stage3_phase1_status") != "completed":
            error_msg = state.get("stage3_error", "Phase 1 failed")
            return {
                **state,
                "stage3_status": "failed",
                "stage3_error": f"Phase 1 failed: {error_msg}"
            }
        
        # Phase 2: Secondary Labeling
        logger.info("Executing Phase 2: Secondary Labeling...")
        state = phase2_secondary_labeling_node(state)
        
        if state.get("stage3_phase2_status") != "completed":
            error_msg = state.get("stage3_error", "Phase 2 failed")
            return {
                **state,
                "stage3_status": "failed", 
                "stage3_error": f"Phase 2 failed: {error_msg}"
            }
        
        # Quality Assessment
        logger.info("Executing Quality Assessment...")
        state = quality_assessment_node(state)
        
        # Finalize Stage 3
        state["stage3_status"] = "completed"
        state["stage3_current_phase"] = "completed"
        
        # Log completion summary
        phase1_groups = state.get("stage3_phase1_groups", {})
        phase2_labels = state.get("stage3_phase2_labels", {})
        
        total_phase1_clusters = sum(
            result.get("n_clusters", 0) 
            for result in phase1_groups.values() 
            if result.get("status") == "completed"
        )
        
        total_phase2_labels = len(phase2_labels)
        
        logger.info(f"✅ Stage 3 Two-Phase Classification completed:")
        logger.info(f"   Phase 1: {total_phase1_clusters} primary clusters")
        logger.info(f"   Phase 2: {total_phase2_labels} semantic labels")
        
        return state
        
    except Exception as e:
        error_msg = f"Stage 3 two-phase classification failed: {str(e)}"
        logger.error(error_msg)
        
        return {
            **state,
            "stage3_status": "failed",
            "stage3_error": error_msg
        }


def stage3_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Router node to determine Stage 3 processing mode and validate inputs.
    
    Args:
        state: LangGraph state
        
    Returns:
        Updated state with routing decisions and parameter defaults
    """
    logger.info("🔀 Stage 3 Router: Determining processing mode")
    
    try:
        # Set default parameters if not specified
        if "stage3_mode" not in state or not state["stage3_mode"]:
            state["stage3_mode"] = "two_phase"  # Default to new two-phase approach
        
        # Set Phase 1 defaults
        if not state.get("stage3_phase1_knn_k"):
            state["stage3_phase1_knn_k"] = 50
        if not state.get("stage3_phase1_csls_threshold"):
            state["stage3_phase1_csls_threshold"] = 0.1
        if not state.get("stage3_phase1_mcl_inflation"):
            state["stage3_phase1_mcl_inflation"] = 2.0
        if not state.get("stage3_phase1_top_m"):
            state["stage3_phase1_top_m"] = 20
        
        # Set Phase 2 defaults
        if not state.get("stage3_phase2_algorithm"):
            state["stage3_phase2_algorithm"] = "louvain"
        if not state.get("stage3_phase2_resolution"):
            state["stage3_phase2_resolution"] = 1.0
        if not state.get("stage3_phase2_mode"):
            state["stage3_phase2_mode"] = "llm_assisted"
        if not state.get("stage3_phase2_edge_weights"):
            state["stage3_phase2_edge_weights"] = {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
        
        # Initialize constraint lists if not present
        if not state.get("stage3_phase2_must_links"):
            state["stage3_phase2_must_links"] = []
        if not state.get("stage3_phase2_cannot_links"):
            state["stage3_phase2_cannot_links"] = []
        
        # Log routing decision
        mode = state["stage3_mode"]
        logger.info(f"Stage 3 mode: {mode}")
        
        if mode == "two_phase":
            logger.info("Using new two-phase labeling approach")
            logger.info(f"Phase 1: kNN(k={state['stage3_phase1_knn_k']}) → CSLS(τ={state['stage3_phase1_csls_threshold']}) → MCL(r={state['stage3_phase1_mcl_inflation']})")
            logger.info(f"Phase 2: {state['stage3_phase2_algorithm']} community detection (resolution={state['stage3_phase2_resolution']})")
        else:
            logger.warning(f"Unknown mode '{mode}', defaulting to two_phase")
            state["stage3_mode"] = "two_phase"
        
        return state
        
    except Exception as e:
        error_msg = f"Stage 3 routing failed: {str(e)}"
        logger.error(error_msg)
        
        return {
            **state,
            "stage3_status": "failed",
            "stage3_error": error_msg
        }


def stage3_main_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main Stage 3 entry point that handles routing and execution.
    
    Args:
        state: LangGraph state
        
    Returns:
        Updated state with Stage 3 results
    """
    logger.info("🎯 Starting Stage 3 Classification")
    
    try:
        # Route and validate
        state = stage3_router_node(state)
        
        if state.get("stage3_status") == "failed":
            return state
        
        # Execute based on mode
        mode = state.get("stage3_mode", "two_phase")
        
        if mode == "two_phase":
            return two_phase_stage3_node(state)
        else:
            # Fallback to two-phase (only supported mode)
            logger.warning(f"Unsupported mode '{mode}', using two_phase")
            state["stage3_mode"] = "two_phase"
            return two_phase_stage3_node(state)
        
    except Exception as e:
        error_msg = f"Stage 3 main execution failed: {str(e)}"
        logger.error(error_msg)
        
        return {
            **state,
            "stage3_status": "failed",
            "stage3_error": error_msg
        }