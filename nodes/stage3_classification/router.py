"""
Simple router for Stage 3 MCL clustering modes.
Clean and minimal validation.
"""
from typing import Any, Dict


def stage3_router(state: Dict[str, Any]) -> Dict[str, Any]:
    """Route Stage 3 classification request to appropriate mode.
    
    Args:
        state: LangGraph state containing classification request
        
    Returns:
        Updated state with routing decision
    """
    # Get mode from state
    mode = state.get("stage3_mode", "estimate")
    
    # Validate mode
    valid_modes = ["estimate", "auto_train", "manual_train"]
    if mode not in valid_modes:
        return {
            **state,
            "stage3_status": "failed",
            "stage3_error": f"Invalid mode '{mode}'. Must be one of: {valid_modes}"
        }
    
    # Check if we have data
    matched_questions = state.get("matched_questions", {})
    if not matched_questions:
        return {
            **state,
            "stage3_status": "failed", 
            "stage3_error": "No matched_questions found in state"
        }
    
    # Basic mode-specific validation
    if mode == "manual_train":
        # Check for required manual parameters
        required_params = ["stage3_manual_inflation", "stage3_manual_k"]
        missing_params = [p for p in required_params if p not in state]
        
        if missing_params:
            return {
                **state,
                "stage3_status": "failed",
                "stage3_error": f"Manual mode missing parameters: {missing_params}"
            }
    
    elif mode == "auto_train":
        # Set default search iterations if not provided
        if "stage3_search_iterations" not in state:
            state = {**state, "stage3_search_iterations": 10}
    
    # All validation passed
    print(f"🧭 Stage 3 Router: Mode '{mode}' validated successfully")
    
    return {
        **state,
        "stage3_status": "ready",
        "stage3_validated_mode": mode
    }