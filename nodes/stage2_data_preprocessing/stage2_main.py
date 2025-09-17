"""
Stage2 Data Preprocessing Node - Simple wrapper for question routing
"""
from typing import Dict, Any, Optional


def stage2_data_preprocessing_node(state: Dict[str, Any], deps: Optional[Any] = None) -> Dict[str, Any]:
    """
    Stage2 질문 설정 및 라우팅 준비 노드
    
    Args:
        state: 현재 그래프 상태 
        deps: Dependencies
        
    Returns:
        Updated state ready for router
    """
    # 디버깅: 현재 상태 출력
    current_question_id = state.get('current_question_id')
    question_type = state.get('question_type')
    current_index = state.get('current_question_index')
    
    print(f"🔍 DEBUG stage2_main:")
    print(f"    current_question_id: {current_question_id}")
    print(f"    question_type: {question_type}")
    print(f"    current_question_index: {current_index}")
    
    # 이미 질문이 설정되어 있으면 그대로 진행
    if current_question_id and question_type and current_index is not None:
        print(f"🔄 CONTINUING: {current_question_id} (type: {question_type})")
        return state
    
    # 첫 실행시에만 초기화
    print("🚀 FIRST INITIALIZATION")
    
    matched_questions = state.get('matched_questions', {})
    if not matched_questions:
        return {
            **state,
            'error': "No matched_questions found",
            'stage2_processing_complete': True
        }
    
    question_ids = list(matched_questions.keys())
    if not question_ids:
        return {
            **state,
            'error': "No questions found",
            'stage2_processing_complete': True
        }
    
    # 첫 번째 질문으로 초기화
    first_question_id = question_ids[0]
    question_info = matched_questions[first_question_id].get('question_info', {})
    first_question_type = question_info.get('question_type', 'ETC')
    
    print(f"🎯 Initializing: {first_question_id} (type: {first_question_type})")
    
    return {
        **state,
        'current_question_id': first_question_id,
        'question_type': first_question_type,
        'current_question_type': first_question_type,  # SENTENCE 노드에서 사용
        'current_question_index': 0,
        'total_questions_stage2': len(question_ids),
        'stage2_processing_complete': False
    }