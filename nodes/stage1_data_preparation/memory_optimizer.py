# nodes/stage1_data_preparation/memory_optimizer.py
# Memory optimizer for Stage 1 - Data Preparation

import gc
import sys
from typing import Dict, Any
from graph.state import GraphState
from nodes.shared.stage_tracker import update_stage_tracking

def stage1_memory_flush_node(state: GraphState) -> GraphState:
    """
    Stage 1 (첫 번째 파이프라인) 완료 후 메모리 정리
    
    정리 대상:
    - raw_survey_info: 대용량 텍스트 데이터
    - raw_data_info: 대용량 메타데이터 (dataframe_path는 raw_dataframe_path로 이동)
    - parsed_survey: 구조화된 질문 데이터 (이미 매칭 완료)
    - data_sample: 임시 샘플 데이터
    - open_columns: matched_questions에 정보 포함으로 중복 제거
    - question_data_match: matched_questions로 통합되어 중복 제거
    - llm_logs, llm_meta: history에 중복 저장되어 있음
    - stage_info 간소화: LLM 사용량 제거, 기본 정보만 유지
    
    보존 대상:
    - survey_context: 설문 전체 맥락 정보 (다음 단계에서 필요)
    - matched_questions: 최종 통합된 질문-데이터 매칭 정보 (유일 보존)
    - raw_dataframe_path: 데이터 접근 경로 (명시적 이름)
    - current_stage: 현재 진행 상태 파악용
    - total_llm_cost_usd: 누적 비용 추적용
    """
    try:
        print("🧹 Starting Stage 1 Memory Flush...")
        
        # 메모리 사용량 확인 (정리 전)
        memory_before = _calculate_state_memory_usage(state)
        
        # LLM 비용을 미리 계산하여 보존 (llm_logs 제거 전에)
        current_cost = state.get("total_llm_cost_usd", 0.0)
        if current_cost <= 0 and state.get("llm_logs"):
            # 기존 비용이 없고 llm_logs가 있으면 새로 계산
            from utils.cost_tracker import calculate_total_llm_cost
            total_cost = calculate_total_llm_cost(state["llm_logs"])
            state["total_llm_cost_usd"] = total_cost
            print(f"  • Calculated total LLM cost: ${total_cost:.6f}")
        elif current_cost > 0:
            # 기존 비용이 있으면 그대로 유지
            print(f"  • Preserved existing LLM cost: ${current_cost:.6f}")
        
        # raw_dataframe_path 미리 보존 (raw_data_info에서)
        if state.get("raw_data_info") and isinstance(state["raw_data_info"], dict):
            if "dataframe_path" in state["raw_data_info"] and not state.get("raw_dataframe_path"):
                state["raw_dataframe_path"] = state["raw_data_info"]["dataframe_path"]
                print(f"  • Preserved dataframe_path to raw_dataframe_path")
        
        # 강력한 메모리 정리 실행
        cleanup_report = force_memory_cleanup(state)
        
        # stage_info 간소화 (메모리 플러시 후에만)
        if state.get("stage_info") and isinstance(state["stage_info"], dict):
            simplified_stage_info = {
                "stage_name": state["stage_info"].get("stage_name"),
                "completion_time": state["stage_info"].get("completion_time"),
                "status": "completed"
            }
            state["stage_info"] = simplified_stage_info
            print(f"  • Simplified stage_info (removed LLM usage details)")
        
        # 메모리 사용량 확인 (정리 후)
        memory_after = _calculate_state_memory_usage(state)
        
        print(f"✅ Stage 1 Memory Flush completed!")
        print(f"   Memory usage: {memory_before} → {memory_after} fields")
        print(f"   Fields cleaned: {', '.join(cleanup_report['fields_cleaned'])}")
        print(f"   Garbage collected: {cleanup_report['gc_collected']} objects")
        if cleanup_report['memory_saved'] > 0:
            print(f"   Memory saved: {cleanup_report['memory_saved']:.1f} MB")
        
        # 🔍 DEBUG: Check state immediately after cleanup
        print(f"🧪 DEBUG - State immediately after memory cleanup:")
        print(f"   Total keys in state: {len(state)}")
        print(f"   Removed fields still present: {[f for f in cleanup_report['fields_cleaned'] if f in state]}")
        
        # Add stage tracking with comprehensive information
        state = update_stage_tracking(state, "STAGE_1_MEMORY_FLUSH")
        
        return state
        
    except Exception as e:
        print(f"❌ Stage 1 Memory Flush failed: {str(e)}")
        state["error"] = f"Memory flush error: {str(e)}"
        return state

def _calculate_state_memory_usage(state: GraphState) -> int:
    """State의 대략적인 메모리 사용량 계산 (필드 수 기반)"""
    non_null_fields = 0
    for key, value in state.items():
        if value is not None:
            non_null_fields += 1
    return non_null_fields

def force_memory_cleanup(state: GraphState) -> Dict[str, Any]:
    """
    호환성을 유지하는 메모리 정리 실행 - 필드는 유지하되 null로 설정
    
    LangGraph TypedDict 호환성을 위해 필드는 제거하지 않고 null로 설정:
    - raw_survey_info, raw_data_info, parsed_survey: 파싱 완료 후 불필요
    - data_sample, open_columns, question_data_match: 매칭 완료 후 불필요  
    - matched_questions_meta: 불필요한 메타데이터
    
    보존 대상:
    - survey_context: 다음 단계에서 필요한 설문 맥락
    - matched_questions: 최종 통합된 질문-데이터 매칭 정보
    - raw_dataframe_path: 데이터 접근 경로
    - current_stage, total_llm_cost_usd: 진행상태 추적용
    """
    fields_cleaned = []
    memory_saved = 0
    
    # 1. null로 설정할 필드 목록 정의 (pop 대신 null 할당)
    fields_to_nullify = [
        # Stage 1 완료 후 불필요한 원본 데이터
        "raw_survey_info",
        "raw_data_info", 
        "parsed_survey",
        "data_sample",
        "open_columns",
        "question_data_match",
        
        # 불필요한 메타데이터
        "matched_questions_meta",
    ]
    
    # 2. 필드들을 null로 설정 (호환성 유지)
    for field in fields_to_nullify:
        if field in state and state[field] is not None:
            # 메모리 사이즈 추정
            try:
                field_size = len(str(state[field])) / (1024 * 1024)  # MB
                memory_saved += field_size
                print(f"  • Nullified {field} (~{field_size:.1f} MB)")
            except:
                print(f"  • Nullified {field}")
            
            # null로 설정 (pop 대신)
            state[field] = None
            fields_cleaned.append(field)
        elif field in state:
            print(f"  • Nullified {field} (was already null)")
            fields_cleaned.append(field)
    
    # 3. raw_dataframe_path 특별 처리 (raw_data_info에서 이동)
    if "raw_data_info" in fields_cleaned and state.get("raw_dataframe_path") is None:
        # raw_data_info가 null로 설정되기 전에 dataframe_path를 보존했어야 함
        print(f"  • Warning: raw_dataframe_path not preserved from raw_data_info")
    
    # 4. Python 가비지 컬렉션 강제 실행
    collected = gc.collect()
    
    return {
        "fields_cleaned": fields_cleaned,
        "memory_saved": memory_saved,
        "gc_collected": collected
    }