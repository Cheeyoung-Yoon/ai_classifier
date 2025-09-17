"""
Stage2 Question Iterator for continuous processing
"""
from typing import Dict, Any

def stage2_next_question_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 2에서 다음 질문으로 이동하는 노드
    
    Args:
        state: 현재 상태
        
    Returns:
        다음 질문으로 업데이트된 상태
    """
    print("🔄 Moving to next question...")
    
    matched_questions = state.get('matched_questions', {})
    all_question_ids = list(matched_questions.keys())
    current_index = state.get('current_question_index', 0)
    
    # 다음 질문으로 이동
    next_index = current_index + 1
    
    if next_index >= len(all_question_ids):
        print(f"✅ All {len(all_question_ids)} questions completed!")
        return {
            **state,
            'stage2_processing_complete': True,
            'current_stage': 'STAGE_2_COMPLETED',
            'message': f"Stage 2 processing completed for all {len(all_question_ids)} questions"
        }
    
    # 다음 질문 설정
    next_question_id = all_question_ids[next_index]
    question_info = matched_questions[next_question_id].get('question_info', {})
    question_type = question_info.get('question_type', 'ETC')
    
    print(f"📋 Moving to question {next_index + 1}/{len(all_question_ids)}")
    print(f"🎯 Next question: {next_question_id}")
    print(f"📝 Question type: {question_type}")
    
    return {
        **state,
        'current_question_id': next_question_id,
        'question_type': question_type,
        'current_question_type': question_type,  # SENTENCE 노드에서 사용
        'current_question_index': next_index,
        'stage2_processing_complete': False,
        'message': f"Processing question {next_index + 1}/{len(all_question_ids)}: {next_question_id}"
    }

def stage2_completion_router(state: Dict[str, Any]) -> str:
    """
    Stage 2 완료 여부에 따른 라우팅
    
    Returns:
        "CONTINUE" or "__END__"
    """
    is_complete = state.get('stage2_processing_complete', False)
    
    if is_complete:
        print("Stage2 Router: All questions processed -> __END__")
        return "__END__"
    else:
        print("Stage2 Router: More questions to process -> CONTINUE")
        return "CONTINUE"