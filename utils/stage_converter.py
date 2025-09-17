# utils/stage_converter.py
"""
Stage 간 변환을 담당하는 유틸리티
"""

from typing import Dict, Any, List
import copy


def convert_stage1_to_stage2(stage1_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage1 결과를 Stage2 처리용 상태로 변환
    
    Args:
        stage1_state: Stage1 완료 후의 상태
        
    Returns:
        Stage2 처리용으로 변환된 상태
    """
    # 원본 state 보존을 위해 복사
    stage2_state = copy.deepcopy(stage1_state)
    
    # matched_questions를 integrated_map으로 변환
    matched_questions = stage1_state.get('matched_questions', {})
    if not matched_questions:
        raise ValueError("No matched_questions found in stage1_state")
    
    # Stage2 필수 필드 추가
    stage2_state.update({
        "integrated_map": matched_questions,
        "questions_to_process": list(matched_questions.keys()),
        "total_questions": len(matched_questions),
        "current_question_index": 0,
        "focus_qid": None,
        "current_question_info": None,
        "processing_complete": False,
        "classification_results": {},
        "stage": "stage2_preprocessing"
    })
    
    return stage2_state


def needs_stage_conversion(state: Dict[str, Any]) -> bool:
    """
    Stage 변환이 필요한지 확인
    
    Args:
        state: 현재 상태
        
    Returns:
        변환 필요 여부
    """
    # integrated_map이 없고 matched_questions가 있으면 변환 필요
    has_matched_questions = 'matched_questions' in state and state['matched_questions']
    has_integrated_map = 'integrated_map' in state and state['integrated_map']
    
    return has_matched_questions and not has_integrated_map


def initialize_first_question(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    첫 번째 질문을 현재 처리 대상으로 설정
    
    Args:
        state: Stage2 변환된 상태
        
    Returns:
        첫 번째 질문이 설정된 상태
    """
    questions_to_process = state.get("questions_to_process", [])
    if not questions_to_process:
        return {
            "processing_complete": True,
            "message": "No questions to process"
        }
    
    # 첫 번째 질문 설정
    first_qid = questions_to_process[0]
    integrated_map = state.get("integrated_map", {})
    first_question_info = integrated_map.get(first_qid, {})
    
    return {
        "focus_qid": first_qid,
        "current_question_info": first_question_info,
        "current_question_index": 0,
        "processing_complete": False,
        "message": f"Set first question: {first_qid}"
    }


def get_question_type(state: Dict[str, Any]) -> str:
    """
    현재 질문의 타입 추출
    
    Args:
        state: 현재 상태
        
    Returns:
        질문 타입 문자열
    """
    current_question_info = state.get("current_question_info", {})
    question_info = current_question_info.get("question_info", {})
    return question_info.get("question_type", "unknown")


def validate_stage2_state(state: Dict[str, Any]) -> bool:
    """
    Stage2 상태가 유효한지 검증
    
    Args:
        state: 검증할 상태
        
    Returns:
        유효성 여부
    """
    required_fields = [
        "integrated_map",
        "questions_to_process", 
        "total_questions",
        "current_question_index"
    ]
    
    return all(field in state for field in required_fields)