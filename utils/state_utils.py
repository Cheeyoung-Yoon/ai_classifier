# utils/state_utils.py
import os
from typing import Dict, Any
from graph.improved_state import ImprovedGraphState

def initialize_project_state(
    project_name: str,
    survey_filename: str,
    data_filename: str,
    base_dir: str = "/home/cyyoon/test_area/ai_text_classification/2.langgraph"
) -> ImprovedGraphState:
    """
    Initialize state with project-based file paths
    
    Args:
        project_name: Name of the project (used for directory structure)
        survey_filename: Name of the survey file (e.g., "test.txt")
        data_filename: Name of the data file (e.g., "-SUV_776부.xlsx")
        base_dir: Base directory path
        
    Returns:
        ImprovedGraphState with properly set paths
    """
    # Create project-based paths
    project_data_dir = os.path.join(base_dir, "data", project_name)
    survey_file_path = os.path.join(project_data_dir, survey_filename)
    data_file_path = os.path.join(project_data_dir, data_filename)
    
    # Initialize state
    state = ImprovedGraphState()
    state["project_name"] = project_name
    state["survey_file_path"] = survey_file_path
    state["data_file_path"] = data_file_path
    
    # Initialize other fields to None
    state["raw_survey_info"] = None
    state["raw_data_info"] = None
    state["parsed_survey"] = None
    state["question_data_match"] = None
    state["integrated_map"] = None
    state["current_question_index"] = None
    state["focus_qid"] = None
    state["current_question_info"] = None
    state["router_decision"] = None
    state["classification_results"] = None
    state["processing_complete"] = None
    state["total_questions"] = None
    state["error"] = None
    
    return state

def get_project_file_path(project_name: str, filename: str, base_dir: str = "/home/cyyoon/test_area/ai_text_classification/2.langgraph") -> str:
    """Generate project-based file path"""
    return os.path.join(base_dir, "data", project_name, filename)

def cleanup_state_memory(state: ImprovedGraphState, cleanup_stage: str) -> ImprovedGraphState:
    """
    Clean up state memory at different stages
    
    Args:
        state: Current state
        cleanup_stage: Stage of cleanup ("after_survey_parse", "after_column_detection", "after_question_match")
    """
    if cleanup_stage == "after_survey_parse":
        state["raw_survey_info"] = None
        
    elif cleanup_stage == "after_column_detection":
        if state["raw_data_info"]:
            # Keep only essential info, remove meta
            state["raw_data_info"] = {
                "path": state["raw_data_info"]["path"],
                "dataframe_path": state["raw_data_info"]["dataframe_path"]
            }
            
    elif cleanup_stage == "after_question_match":
        state["parsed_survey"] = None
        # Remove redundant matched_questions since we have question_data_match
        if "matched_questions" in state:
            del state["matched_questions"]
    
    return state
