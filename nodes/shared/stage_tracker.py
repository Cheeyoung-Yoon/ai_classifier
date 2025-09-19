# nodes/shared/stage_tracker.py - Stage Tracking and Management Nodes
from typing import Dict, Any
from datetime import datetime
import logging

# Use relative imports instead of sys.path manipulation
from ...utils.cost_tracker import (
    calculate_total_llm_cost, 
    create_stage_info, 
    print_stage_summary,
    calculate_pipeline_runtime,
    print_pipeline_status
)
from ...utils.stage_history_manager import get_or_create_history_manager
from ...utils.project_manager import get_project_manager
from ...config.config import settings

logger = logging.getLogger(__name__)

def update_stage_tracking(state: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    """
    Update stage tracking with separate history file management
    
    Args:
        state: Current graph state
        stage_name: Name of the current stage
        
    Returns:
        Updated state with stage tracking information
    """
    current_time = datetime.now().isoformat()
    
    # 프로젝트 매니저를 통해 state 저장
    project_name = state.get('project_name')
    if project_name and settings.SAVE_STATE_LOG:
        project_manager = get_project_manager(project_name)
        config = {'save_state_log': settings.SAVE_STATE_LOG}
        
        # current_stage 업데이트 후 저장
        updated_state = {**state, 'current_stage': stage_name}
        project_manager.save_state(dict(updated_state), config)
    
    # Calculate total LLM cost - 기존 값 유지 또는 새로 계산
    existing_cost = state.get('total_llm_cost_usd', 0.0)
    if existing_cost > 0:
        # 이미 계산된 비용이 있으면 유지
        total_cost = existing_cost
    else:
        # 없으면 llm_logs에서 새로 계산
        total_cost = calculate_total_llm_cost(state.get('llm_logs', []))
    
    # Get pipeline timing information
    pipeline_start_time = state.get('pipeline_start_time')
    last_stage_time = state.get('last_stage_time', pipeline_start_time)
    pipeline_id = state.get('pipeline_id')
    
    # Calculate runtimes
    stage_runtime = None
    if last_stage_time:
        stage_runtime = calculate_pipeline_runtime(last_stage_time, current_time)
    
    # Create comprehensive stage info
    stage_info = create_stage_info(
        stage_name=stage_name,
        llm_logs=state.get('llm_logs', []),
        start_time=last_stage_time,
        completion_time=current_time,
        pipeline_start_time=pipeline_start_time
    )
    
    # Calculate total pipeline runtime
    pipeline_runtime = 0.0
    if pipeline_start_time:
        pipeline_runtime = calculate_pipeline_runtime(pipeline_start_time, current_time)
    
    # Update state - 최소한의 정보만 유지 (나머지는 history로)
    updated_state = state.copy()
    updated_state['total_llm_cost_usd'] = total_cost  # LLM 비용 추적용
    updated_state['current_stage'] = stage_name       # 현재 진행 상태만
    
    # Update separate stage history file
    if pipeline_id:
        try:
            project_name = state.get('project_name', 'Unknown')
            history_manager = get_or_create_history_manager(project_name, pipeline_id)
            
            # Add stage to separate history file
            metadata = history_manager.add_stage(stage_info)
            
            # Print comprehensive summary
            print_stage_summary(
                stage_name, 
                state.get('llm_logs', []),
                pipeline_start_time,
                stage_runtime
            )
            
            # Print current pipeline status with history manager info
            history_manager.print_current_status()
            
        except Exception as e:
            print(f"⚠️ Failed to update stage history: {e}")
            # Fall back to basic stage summary
            print_stage_summary(
                stage_name, 
                state.get('llm_logs', []),
                pipeline_start_time,
                stage_runtime
            )
    
    return updated_state

def create_stage_completion_node(stage_name: str):
    """
    Create a node function for stage completion tracking
    
    Args:
        stage_name: Name of the stage to track
        
    Returns:
        Node function that updates stage tracking
    """
    def stage_completion_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Node function for stage completion"""
        return update_stage_tracking(state, stage_name)
    
    return stage_completion_node

# Pre-defined stage completion nodes
def stage1_data_preparation_completion(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 1: Data Preparation Completion Node"""
    return update_stage_tracking(state, "STAGE_1_DATA_PREPARATION")

def stage1_memory_flush_completion(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 1: Memory Flush Completion Node"""
    return update_stage_tracking(state, "STAGE_1_MEMORY_FLUSH")

def stage2_classification_start(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 2: Classification Start Node"""
    return update_stage_tracking(state, "STAGE_2_CLASSIFICATION_START")

def stage2_classification_completion(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 2: Classification Completion Node"""
    return update_stage_tracking(state, "STAGE_2_CLASSIFICATION_COMPLETE")

def final_completion(state: Dict[str, Any]) -> Dict[str, Any]:
    """Final Pipeline Completion Node"""
    return update_stage_tracking(state, "PIPELINE_COMPLETE")

# Memory status tracking helper
def add_memory_status_tracking(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add memory status information to state with improved tracking
    
    Args:
        state: Current graph state
        
    Returns:
        State with memory status information
    """
    if 'memory_status' not in state:
        # Count non-null fields
        non_null_fields = sum(1 for value in state.values() if value is not None)
        total_fields = len(state)
        
        # Identify missing essential fields (you can customize this list)
        essential_fields = ['project_name', 'survey_file_path', 'data_file_path']
        missing_essential = [field for field in essential_fields if not state.get(field)]
        
        memory_status = {
            "non_null_fields": non_null_fields,
            "total_fields": total_fields,
            "missing_essential": missing_essential,
            "status": "checked",
            "last_updated": datetime.now().isoformat(),
            "pipeline_runtime": state.get('pipeline_runtime_seconds', 0.0)
        }
        
        updated_state = state.copy()
        updated_state['memory_status'] = memory_status
        return updated_state
    
    return state

def print_final_pipeline_summary(state: Dict[str, Any]):
    """
    Print comprehensive final pipeline summary using separate history file
    
    Args:
        state: Final graph state
    """
    from utils.cost_tracker import get_pipeline_summary, format_runtime_display, format_cost_summary
    
    summary = get_pipeline_summary(state)
    pipeline_id = state.get('pipeline_id')
    
    print(f"\n🎉 PIPELINE EXECUTION COMPLETE")
    print(f"{'='*80}")
    
    # Overall stats
    print(f"Final Stage: {summary['current_stage']}")
    print(f"Pipeline ID: {pipeline_id}")
    if 'total_runtime_display' in summary:
        print(f"Total Runtime: {summary['total_runtime_display']}")
    print(f"Total Cost: {format_cost_summary(summary['total_cost_usd'])}")
    print(f"Total LLM Calls: {summary['llm_usage']['total_calls']}")
    print(f"Total Tokens: {summary['llm_usage']['total_tokens']:,}")
    
    # Get stage breakdown from separate history file
    if pipeline_id:
        try:
            project_name = state.get('project_name', 'Unknown')
            history_manager = get_or_create_history_manager(project_name, pipeline_id)
            stage_summary = history_manager.get_stage_summary()
            
            if stage_summary:
                print(f"\n📊 STAGE BREAKDOWN (from {history_manager.get_history_file_path()}):")
                print(f"{'─'*80}")
                for stage in stage_summary:
                    stage_num = stage['stage_number']
                    stage_name = stage['stage_name']
                    runtime = format_runtime_display(stage['runtime_seconds'])
                    cost = format_cost_summary(stage['cost_usd'])
                    
                    print(f"{stage_num:2d}. {stage_name:<30} Runtime: {runtime:<10} Cost: {cost}")
                    
            print(f"\n📁 Stage History File: {history_manager.get_history_file_path()}")
            
        except Exception as e:
            print(f"⚠️ Could not load stage history: {e}")
    
    print(f"{'='*80}\n")