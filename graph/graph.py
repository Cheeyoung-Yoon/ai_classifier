# graph/graph.py - Memory Optimized Graph
from typing import Dict, Any
import sys
import os
from pathlib import Path

# Import settings to get project root path
from config.config import settings

# Add project root to path using settings
project_root = Path(settings.PROJECT_DATA_BASE_DIR).resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph.graph import StateGraph, START, END
import json

# Import the updated state with memory optimization
from graph.state import GraphState, initialize_project_state

# Import project directory manager
from utils.project_manager import get_project_manager, initialize_project_directories
from config.config import settings

# Import existing nodes
from nodes.stage1_data_preparation.survey_loader import load_survey_node
from nodes.stage1_data_preparation.data_loader import load_data_node
from nodes.stage1_data_preparation.survey_parser import parse_survey_node
from nodes.stage1_data_preparation.survey_context import survey_context_node
from nodes.stage1_data_preparation.column_extractor import get_open_column_node
from nodes.stage1_data_preparation.question_matcher import question_data_matcher_node
from nodes.survey_data_integrate import survey_data_integrate_node
from nodes.stage1_data_preparation.memory_optimizer import stage1_memory_flush_node
from nodes.state_flush_node import memory_status_check_node

# Import Stage 2 router and nodes
from router.stage2_router import stage2_type_router
from nodes.stage2_data_preprocessing.stage2_main import stage2_data_preprocessing_node
from nodes.stage2_data_preprocessing.stage2_word_node import stage2_word_node
from nodes.stage2_data_preprocessing.stage2_sentence_node import stage2_sentence_node
from nodes.stage2_data_preprocessing.stage2_etc_node import stage2_etc_node
from nodes.stage2_next_question import stage2_next_question_node, stage2_completion_router

# Import Stage 3 classification node
from nodes.stage3_classification.stage3_node import stage3_classification_node

# Import stage tracking nodes
from nodes.shared.stage_tracker import (
    stage1_data_preparation_completion,
    stage1_memory_flush_completion,
    stage2_classification_start,
    final_completion,
    print_pipeline_status
)

def pipeline_initialization_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize pipeline with proper tracking and separated history"""
    print("🚀 PIPELINE INITIALIZATION")
    print("─" * 40)
    
    # 프로젝트 디렉토리 구조 생성
    project_name = state.get('project_name', 'unknown')
    project_dirs = initialize_project_directories(project_name)
    
    print(f"Project: {project_name}")
    print(f"Project Directory: {project_dirs['project_dir']}")
    print(f"Temp Data Directory: {project_dirs['temp_data_dir']}")
    print(f"State File: {project_dirs['state_file']}")
    print(f"Survey: {state.get('survey_file_path', 'Unknown')}")
    print(f"Data: {state.get('data_file_path', 'Unknown')}")
    
    # 상태에 프로젝트 디렉토리 정보 추가
    state['project_directories'] = project_dirs
    print("─" * 40)
    
    # Initialize history manager and update stage tracking
    from utils.stage_history_manager import get_or_create_history_manager
    from nodes.shared.stage_tracker import update_stage_tracking
    
    # Get or create history manager for this pipeline
    history_manager = get_or_create_history_manager(state.get('pipeline_id'))
    
    # Update stage tracking for pipeline initialization
    updated_state = update_stage_tracking(state, "PIPELINE_INITIALIZATION")
    
    # Print initial status
    print_pipeline_status(updated_state)
    
    return updated_state

def create_workflow() -> StateGraph:
    """Create the memory optimized LangGraph workflow"""
    
    workflow = StateGraph(GraphState)
    
    # Add pipeline initialization
    workflow.add_node("pipeline_init", pipeline_initialization_node)
    
    # Add Stage 1 nodes (data preparation)
    workflow.add_node("load_survey", load_survey_node)
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("parse_survey", parse_survey_node)
    workflow.add_node("extract_survey_context", survey_context_node)
    workflow.add_node("get_open_columns", get_open_column_node)
    workflow.add_node("match_questions", question_data_matcher_node)
    workflow.add_node("survey_data_integrate", survey_data_integrate_node)
    
    # Add memory management nodes
    workflow.add_node("stage1_memory_flush", stage1_memory_flush_node)
    workflow.add_node("memory_status_check", memory_status_check_node)
    
    # Add Stage 2 nodes (data preprocessing with router)
    workflow.add_node("stage2_main", stage2_data_preprocessing_node)
    workflow.add_node("stage2_word_node", stage2_word_node)
    workflow.add_node("stage2_sentence_node", stage2_sentence_node)
    workflow.add_node("stage2_etc_node", stage2_etc_node)
    workflow.add_node("stage2_next_question", stage2_next_question_node)
    
    # Add Stage 3 node (MCL classification)
    workflow.add_node("stage3_classification", stage3_classification_node)
    
    # Add stage tracking nodes
    workflow.add_node("stage1_completion", stage1_data_preparation_completion)
    workflow.add_node("stage1_flush_completion", stage1_memory_flush_completion)
    workflow.add_node("stage2_start", stage2_classification_start)
    
    # Define Stage 1 workflow edges
    workflow.add_edge(START, "pipeline_init")
    workflow.add_edge("pipeline_init", "load_survey")
    workflow.add_edge("load_survey", "load_data")
    workflow.add_edge("load_data", "parse_survey")
    workflow.add_edge("parse_survey", "extract_survey_context")
    workflow.add_edge("extract_survey_context", "get_open_columns")
    workflow.add_edge("get_open_columns", "match_questions")
    workflow.add_edge("match_questions", "survey_data_integrate")
    
    # Add stage completion tracking
    workflow.add_edge("survey_data_integrate", "stage1_completion")
    workflow.add_edge("stage1_completion", "stage1_memory_flush")
    workflow.add_edge("stage1_memory_flush", "stage1_flush_completion")
    workflow.add_edge("stage1_flush_completion", "memory_status_check")
    
    # Stage 1 to Stage 2 transition
    workflow.add_edge("memory_status_check", "stage2_start")
    workflow.add_edge("stage2_start", "stage2_main")
    
    # Stage 2 router-based conditional edges
    workflow.add_conditional_edges(
        "stage2_main",
        stage2_type_router,
        {
            "WORD": "stage2_word_node",
            "SENTENCE": "stage2_sentence_node", 
            "ETC": "stage2_etc_node",
            "__END__": END
        }
    )
    
    # All Stage 2 processing nodes go to next question iterator
    workflow.add_edge("stage2_word_node", "stage2_next_question")
    workflow.add_edge("stage2_sentence_node", "stage2_next_question")
    workflow.add_edge("stage2_etc_node", "stage2_next_question")
    
    # From next question, decide whether to continue or end
    workflow.add_conditional_edges(
        "stage2_next_question",
        stage2_completion_router,
        {
            "CONTINUE": "stage2_main",  # Loop back to main for next question
            "COMPLETE": "stage3_classification"  # Move to Stage 3 when all questions done
        }
    )
    
    # Stage 3 to end
    workflow.add_edge("stage3_classification", END)
    
    return workflow

def run_pipeline(project_name: str, survey_filename: str, data_filename: str):
    """
    Run the complete memory optimized pipeline
    
    Args:
        project_name: Project name for path organization
        survey_filename: Survey file name (e.g., "test.txt")
        data_filename: Data file name (e.g., "-SUV_776부.xlsx")
    """
    print(f"🚀 Starting Memory Optimized Pipeline")
    print(f"📁 Project: {project_name}")
    print(f"📄 Survey: {survey_filename}")
    print(f"📊 Data: {data_filename}")
    print("-" * 50)
    
    # Initialize optimized state
    initial_state = initialize_project_state(project_name, survey_filename, data_filename)
    
    # Create and compile workflow
    workflow = create_workflow()
    app = workflow.compile()
    
    print("🔄 Executing pipeline...")
    
    try:
        # Run the workflow with increased recursion limit
        result = app.invoke(initial_state, config={"recursion_limit": 100})
        
        print("✅ Pipeline completed successfully!")
        print(f"📋 Final state summary:")
        print(f"  Project: {result.get('project_name')}")
        print(f"  Current Stage: {result.get('current_stage', 'Unknown')}")
        print(f"  Pipeline ID: {result.get('pipeline_id', 'Unknown')}")
        print(f"  Total LLM Cost: ${result.get('total_llm_cost_usd', 0):.4f}")
        print(f"  Pipeline Runtime: {result.get('pipeline_runtime', 'Unknown')}")
        print(f"  Question matches: {bool(result.get('question_data_match'))}")
        print(f"  Error: {result.get('error', 'None')}")
        
        # Show memory optimization results
        memory_optimized_fields = [
            "raw_survey_info", "parsed_survey", "matched_questions", 
            "matched_questions_meta", "open_columns", "data_sample"
        ]
        
        cleaned_fields = [field for field in memory_optimized_fields if not result.get(field)]
        print(f"🧹 Cleaned fields: {len(cleaned_fields)}/{len(memory_optimized_fields)}")
        
        # Show stage history file information
        pipeline_id = result.get('pipeline_id')
        if pipeline_id:
            from utils.stage_history_manager import get_or_create_history_manager
            history_manager = get_or_create_history_manager(pipeline_id)
            history_file_path = history_manager.get_history_file_path()
            print(f"📄 Stage history saved to: {history_file_path}")
        
        return result
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        return {"error": str(e)}

# Legacy function for backward compatibility
def create_memory_optimized_workflow():
    """Legacy function name - redirects to create_workflow"""
    return create_workflow()

def run_memory_optimized_pipeline(project_name: str, survey_filename: str, data_filename: str):
    """Legacy function name - redirects to run_pipeline"""
    return run_pipeline(project_name, survey_filename, data_filename)

if __name__ == "__main__":
    # Example usage
    result = run_pipeline(
        project_name="test",
        survey_filename="test.txt", 
        data_filename="-SUV_776부.xlsx"
    )
    
    print(f"\n📋 Result Keys: {list(result.keys())}")
    
    if result.get("question_data_match"):
        print(f"🔗 Question Matches Preview:")
        try:
            matches = json.loads(result["question_data_match"]) if isinstance(result["question_data_match"], str) else result["question_data_match"]
            for question, columns in list(matches.items())[:3]:  # Show first 3
                print(f"  {question}: {columns}")
            if len(matches) > 3:
                print(f"  ... and {len(matches) - 3} more")
        except:
            print(f"  Raw data: {str(result['question_data_match'])[:100]}...")
