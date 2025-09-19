# graph/state.py - Memory Optimized State
from typing import TypedDict, List, Dict, Any, Optional
import os

class GraphState(TypedDict):
    """Memory optimized LangGraph state for text classification workflow"""
    
    # Core project information - maintained throughout
    project_name: Optional[str]  # Initial setup, maintained throughout for path references
    
    # Dynamic file paths based on project_name
    survey_file_path: Optional[str]  # ./data/{project_name}/{survey_file}
    data_file_path: Optional[str]    # ./data/{project_name}/{data_file}
    
    # Project directory structure
    project_directories: Optional[Dict[str, str]]  # Project directory paths
    
    # Temporary raw data (kept for compatibility, nulled after processing)
    raw_survey_info: Optional[Dict[str, Any]]  # Nulled after parse_survey
    raw_data_info: Optional[Dict[str, Any]]    # Nulled after get_open_columns
    
    # Data processing fields (essential for data access)
    raw_dataframe_path: Optional[str]          # CSV path for data access
    data_sample: Optional[Dict[str, Any]]      # Nulled after LLM processing
    
    # Temporary parsed data (kept for compatibility, nulled after matching)
    parsed_survey: Optional[Dict[str, Any]]    # Nulled after match_questions
    survey_context: Optional[str]              # Survey overall context/purpose summary
    open_columns: Optional[List[str]]          # Nulled after matching (available in raw_data_info)
    
    # Final matching results (only final format preserved)
    question_data_match: Optional[str]         # JSON string format (nulled after integration)
    matched_questions: Optional[Dict[str, Any]]  # Final structured format - ONLY THIS PRESERVED
    matched_questions_meta: Optional[Dict[str, Any]]  # Nulled after integration
    
    # LLM tracking - 필수 최소 정보만
    total_llm_cost_usd: Optional[float]        # Total LLM usage cost in USD (누적 비용 추적)
    
    # Stage tracking - 필수 최소 정보만 (나머지는 history 파일로)
    current_stage: Optional[str]               # Current processing stage
    
    # Pipeline management - 기본 식별 정보만
    pipeline_id: Optional[str]                 # Unique pipeline identifier
    stage_history_file: Optional[str]          # Path to separate stage history file
    # Error handling - essential
    error: Optional[str]
    
    # Stage 2 data preprocessing fields
    question_type: Optional[str]               # WORD/SENTENCE/ETC classification result
    current_question_id: Optional[str]         # Currently processing question ID
    current_question_type: Optional[str]       # Current question type for processing
    current_question_index: Optional[int]      # Current question index (0-based)
    stage2_processing_complete: Optional[bool] # Whether Stage 2 processing is complete
    total_questions_stage2: Optional[int]      # Total number of questions for Stage 2
    stage2_processing_results: Optional[Dict[str, Any]]  # Stage 2 processing results
    stage2_csv_output_path: Optional[str]      # Path to Stage 2 CSV output file
    grammar_corrected_text: Optional[str]      # Grammar-corrected version for SENTENCE type
    sentence_analysis_result: Optional[Dict[str, Any]]  # Sentence analysis result
    
    # Stage 3 classification fields
    stage3_mode: Optional[str]                 # MCL mode: estimate/auto_train/manual_train
    stage3_status: Optional[str]               # Status: completed/failed/skipped
    stage3_error: Optional[str]                # Error message if failed
    stage3_search_iterations: Optional[int]    # Auto-train search iterations
    stage3_manual_inflation: Optional[float]   # Manual mode inflation parameter
    stage3_manual_k: Optional[int]             # Manual mode k neighbors parameter
    stage3_manual_max_iters: Optional[int]     # Manual mode max iterations
    stage3_best_parameters: Optional[Dict[str, Any]]     # Auto-train best parameters
    stage3_best_score: Optional[float]         # Auto-train best score
    stage3_best_evaluation: Optional[Dict[str, Any]]     # Evaluation metrics (NMI/ARI)
    stage3_cluster_labels: Optional[List[int]] # Cluster labels for each embedding
    stage3_cluster_mapping: Optional[Dict[str, Any]]     # Detailed cluster mapping
    stage3_cluster_summary: Optional[Dict[str, Any]]     # Cluster summary statistics
    stage3_estimated_clusters: Optional[int]   # Estimate mode result
    stage3_recommended_k: Optional[int]        # Estimate mode recommended k
    stage3_data_summary: Optional[Dict[str, Any]]        # Data processing summary
    processing_time_seconds: Optional[float]   # Processing time for stage3

# Utility functions for state management
def get_project_file_path(project_name: str, filename: str, base_dir: str = "/home/cyyoon/test_area/ai_text_classification/2.langgraph", subdir: str = "") -> str:
    """Generate project-based file path"""
    if subdir:
        return os.path.join(base_dir, "data", project_name, subdir, filename)
    return os.path.join(base_dir, "data", project_name, filename)

def get_raw_file_path(project_name: str, filename: str, base_dir: str = "/home/cyyoon/test_area/ai_text_classification/2.langgraph") -> str:
    """Generate raw data file path"""
    return get_project_file_path(project_name, filename, base_dir, "raw")

def initialize_project_state(
    project_name: str,
    survey_filename: str,
    data_filename: str,
    base_dir: str = "/home/cyyoon/test_area/ai_text_classification/2.langgraph",
    use_raw_dir: bool = True
) -> GraphState:
    """
    Initialize state with project-based file paths
    
    Args:
        project_name: Name of the project (used for directory structure)
        survey_filename: Name of the survey file (e.g., "test.txt")
        data_filename: Name of the data file (e.g., "-SUV_776부.xlsx")
        base_dir: Base directory path
        use_raw_dir: Whether to use raw subdirectory for file paths
        
    Returns:
        GraphState with properly set paths
    """
    # Create project-based paths
    if use_raw_dir:
        survey_file_path = get_raw_file_path(project_name, survey_filename, base_dir)
        data_file_path = get_raw_file_path(project_name, data_filename, base_dir)
    else:
        survey_file_path = get_project_file_path(project_name, survey_filename, base_dir)
        data_file_path = get_project_file_path(project_name, data_filename, base_dir)
    
    # Initialize state
    state = GraphState()
    state["project_name"] = project_name
    state["survey_file_path"] = survey_file_path
    state["data_file_path"] = data_file_path
    
    # Initialize other fields to None
    state["raw_survey_info"] = None
    state["raw_data_info"] = None
    state["raw_dataframe_path"] = None
    state["data_sample"] = None
    state["parsed_survey"] = None
    state["survey_context"] = None
    state["open_columns"] = None
    state["question_data_match"] = None
    state["matched_questions"] = None
    state["matched_questions_meta"] = None
    
    # Initialize minimal tracking fields only
    state["total_llm_cost_usd"] = 0.0
    state["current_stage"] = "PIPELINE_INITIALIZED"
    
    # Initialize separate stage history management
    from utils.stage_history_manager import get_or_create_history_manager
    history_manager = get_or_create_history_manager(project_name)
    state["pipeline_id"] = history_manager.pipeline_id
    state["stage_history_file"] = history_manager.get_history_file_path()
    
    # Stage 2 전용 필드들은 Stage 1에서 초기화하지 않음
    # (integrated_map, current_question_index, focus_qid, current_question_info,
    #  router_decision, classification_results, processing_complete, total_questions)
    # 이들은 Stage 2 시작 시점에서만 필요시 초기화됨
    
    # Error handling - 필수 필드만 초기화
    state["error"] = None
    
    # Stage 2 fields - 초기에는 None으로 설정
    state["question_type"] = None
    state["current_question_id"] = None
    state["current_question_type"] = None
    state["current_question_index"] = None
    state["stage2_processing_complete"] = None
    state["total_questions_stage2"] = None
    state["stage2_processing_results"] = None
    state["stage2_csv_output_path"] = None
    state["grammar_corrected_text"] = None
    state["sentence_analysis_result"] = None
    
    # Stage 3 fields - 초기에는 None으로 설정
    state["stage3_mode"] = None
    state["stage3_status"] = None
    state["stage3_error"] = None
    state["stage3_search_iterations"] = None
    state["stage3_manual_inflation"] = None
    state["stage3_manual_k"] = None
    state["stage3_manual_max_iters"] = None
    state["stage3_best_parameters"] = None
    state["stage3_best_score"] = None
    state["stage3_best_evaluation"] = None
    state["stage3_cluster_labels"] = None
    state["stage3_cluster_mapping"] = None
    state["stage3_cluster_summary"] = None
    state["stage3_estimated_clusters"] = None
    state["stage3_recommended_k"] = None
    state["stage3_data_summary"] = None
    state["processing_time_seconds"] = None
    
    return state
