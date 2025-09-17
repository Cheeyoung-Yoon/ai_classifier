# nodes/state_flush_node.py
# Memory management utilities - contains non-stage1 specific functions

from graph.state import GraphState
from nodes.stage1_data_preparation.memory_optimizer import stage1_memory_flush_node, force_memory_cleanup
from nodes.shared.stage_tracker import add_memory_status_tracking, update_stage_tracking
from utils.project_manager import get_project_manager
from config.config import settings

def memory_status_check_node(state: GraphState) -> GraphState:
    """
    Memory status check node - checks memory status after cleanup operations
    
    This function validates the state after memory cleanup and provides
    status information for debugging and monitoring purposes.
    """
    try:
        print("🔍 Memory Status Check...")
        
        # 🔍 DEBUG: Check what fields exist at this point
        print(f"🧪 DEBUG - State at memory_status_check:")
        print(f"   Total keys in state: {len(state)}")
        removed_fields = ['raw_survey_info', 'raw_data_info', 'parsed_survey', 
                         'data_sample', 'open_columns', 'question_data_match', 'matched_questions_meta']
        still_present = [f for f in removed_fields if f in state]
        print(f"   Previously removed fields now present: {still_present}")
        
        # Count non-null fields
        non_null_fields = 0
        total_fields = 0
        for key, value in state.items():
            total_fields += 1
            if value is not None:
                non_null_fields += 1
        
        # Check for essential fields that should exist - UPDATED LIST
        essential_fields = ["matched_questions", "survey_context", "current_stage"]
        missing_essential = []
        for field in essential_fields:
            if not state.get(field):
                missing_essential.append(field)
        
        print(f"   State fields: {non_null_fields}/{total_fields} non-null")
        
        if missing_essential:
            print(f"   ⚠️  Missing essential fields: {', '.join(missing_essential)}")
        else:
            print(f"   ✅ Essential fields present")
        
        # 상태 저장 (config 설정에 따라)
        project_name = state.get('project_name')
        if project_name and settings.SAVE_STATE_LOG:
            project_manager = get_project_manager(project_name)
            config = {'save_state_log': settings.SAVE_STATE_LOG}
            project_manager.save_state(dict(state), config)
        
        # Update stage tracking (memory_status는 state에 추가하지 않음)
        state = update_stage_tracking(state, "MEMORY_STATUS_CHECK")
        
        return state
        
    except Exception as e:
        print(f"❌ Memory status check failed: {str(e)}")
        state["error"] = f"Memory status check error: {str(e)}"
        return state