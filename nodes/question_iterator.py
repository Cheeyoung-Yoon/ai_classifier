# nodes/question_iterator.py
"""
질문 반복 처리 노드 - Simple Wrapper Node
"""

from typing import Dict, Any, List
from graph.state import GraphState
from utils.stage_converter import (
    needs_stage_conversion, 
    convert_stage1_to_stage2, 
    initialize_first_question,
    validate_stage2_state
)

def question_iterator_node(state: GraphState) -> Dict[str, Any]:
    """
    질문 순회 처리를 위한 단순 래퍼 노드
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        처리할 질문 목록과 현재 상태
    """
    try:
        # Stage 변환이 필요한지 확인 및 처리
        if needs_stage_conversion(state):
            print("🔄 Stage1 → Stage2 자동 변환 중...")
            stage2_state = convert_stage1_to_stage2(state)
            state.update(stage2_state)
        
        # Stage2 상태 검증
        if not validate_stage2_state(state):
            return {
                "error": "Invalid stage2 state - missing required fields",
                "processing_complete": True
            }
        
        # 질문 처리 로직 위임
        return _process_question_iteration(state)
        
    except Exception as e:
        return {
            "error": f"Question iterator failed: {str(e)}",
            "processing_complete": True
        }


def _process_question_iteration(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    실제 질문 순회 처리 로직 (내부 함수)
    
    Args:
        state: 유효한 Stage2 상태
        
    Returns:
        업데이트된 상태
    """
    integrated_map = state.get("integrated_map", {})
    all_qids = list(integrated_map.keys())
    current_index = state.get("current_question_index", 0)
    
    # 처리 완료 확인
    if current_index >= len(all_qids):
        return {
            "questions_to_process": all_qids,
            "total_questions": len(all_qids),
            "current_index": current_index,
            "processing_complete": True,
            "message": f"All {len(all_qids)} questions processed"
        }
    
    # 현재 처리할 질문 설정
    current_qid = all_qids[current_index]
    current_question_info = integrated_map[current_qid]
    
    return {
        "questions_to_process": all_qids,
        "total_questions": len(all_qids),
        "current_index": current_index,
        "focus_qid": current_qid,
        "current_question_info": current_question_info,
        "processing_complete": False,
        "message": f"Processing question {current_index + 1}/{len(all_qids)}: {current_qid}"
    }

def increment_question_index(state: GraphState) -> Dict[str, Any]:
    """
    다음 질문으로 인덱스를 증가시킴
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태
    """
    current_index = state.get("current_question_index", 0)
    new_index = current_index + 1
    
    return {
        "current_question_index": new_index,
        "message": f"Moving to question index {new_index}"
    }

def should_continue_processing(state: GraphState) -> bool:
    """
    처리를 계속할지 결정
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        계속 처리할지 여부
    """
    total_questions = state.get("total_questions", 0)
    current_index = state.get("current_question_index", 0)
    
    return current_index < total_questions
