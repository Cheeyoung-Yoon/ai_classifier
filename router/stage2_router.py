"""
Stage2 Router - Question type based routing for stage2 preprocessing
"""
from typing import Dict, Any, TypedDict, Literal

# 기존 qytpe_router.py와 동일한 패턴 사용
Branch = Literal["WORD", "SENTENCE", "ETC", "__END__"]

# 매핑 테이블 - 기존과 동일
QTYPE_MAP = {
    "concept": "WORD",
    "img": "WORD", 
    "depend": "SENTENCE",
    "depend_pos_neg": "SENTENCE",
    "pos_neg": "SENTENCE",
    "etc": "ETC",
}

def stage2_type_router(state: Dict[str, Any]) -> Branch:
    """
    Stage2에서 현재 질문의 타입에 따라 분기 결정
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        처리할 노드 타입 ("WORD", "SENTENCE", "ETC", "__END__")
    """
    # 처리 완료 확인
    if state.get("stage2_processing_complete", False):
        return "__END__"
    
    # 현재 질문 정보 가져오기
    current_question_id = state.get('current_question_id')
    if not current_question_id:
        print("Router: No current_question_id -> __END__")
        return "__END__"
    
    matched_questions = state.get('matched_questions', {})
    if current_question_id not in matched_questions:
        print(f"Router: Question {current_question_id} not found -> __END__")
        return "__END__"
    
    # 질문 타입 추출
    question_info = matched_questions[current_question_id]['question_info']
    qtype = question_info.get('question_type')
    
    if not qtype:
        print(f"Router: No question_type for {current_question_id} -> __END__")
        return "__END__"
    
    # 타입에 따른 분기 결정
    branch = QTYPE_MAP.get(qtype, "ETC")
    
    print(f"Stage2 Router: QID {current_question_id}, Type: {qtype} -> Branch: {branch}")
    
    return branch


def stage2_continue_router(state: Dict[str, Any]) -> Branch:
    """
    Stage2에서 다음 질문 처리를 위한 계속 여부 결정
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        계속 처리할지 여부 ("CONTINUE", "__END__")
    """
    # 질문 리스트 확인
    questions_to_process = state.get("questions_to_process", [])
    current_index = state.get("current_question_index", 0)
    
    if current_index < len(questions_to_process):
        next_qid = questions_to_process[current_index]
        print(f"Stage2 Continue Router: Processing next question {next_qid} (index {current_index})")
        return "CONTINUE"
    else:
        print("Stage2 Continue Router: All questions processed -> __END__")
        return "__END__"