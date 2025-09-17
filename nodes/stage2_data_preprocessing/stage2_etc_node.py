"""
Stage2 ETC Node - Handles ETC type questions
"""
from typing import Dict, Any, Optional


def stage2_etc_node(state: Dict[str, Any], deps: Optional[Any] = None) -> Dict[str, Any]:
    """
    ETC 타입 질문 처리 노드
    
    ETC 타입은 기타 분류되지 않은 질문들로 기본적인 처리만 수행
    
    Args:
        state: Current graph state
        deps: Dependencies
        
    Returns:
        Updated state indicating ETC processing completion
    """
    current_question_id = state.get('current_question_id')
    current_question_type = state.get('current_question_type')
    
    print(f"stage2_etc_node: Processing ETC type question {current_question_id} ({current_question_type})")
    
    # ETC 타입은 기본적인 처리만 수행
    updated_state = {
        **state,
        'stage2_etc_processed': True,
        'stage2_processing_type': 'ETC',
        'stage2_status': 'completed',
        'stage2_result': {
            'question_id': current_question_id,
            'question_type': current_question_type,
            'processing_type': 'ETC',
            'note': 'ETC type questions processed with basic handling'
        }
    }
    
    print(f"ETC processing completed for {current_question_id}")
    return updated_state