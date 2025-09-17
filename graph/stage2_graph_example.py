# 실제 LangGraph Stage2 워크플로우 구현 예시

from langgraph.graph import StateGraph, START, END
from typing import Dict, Any

def create_stage2_workflow() -> StateGraph:
    """Stage2 실제 LangGraph 워크플로우 구현"""
    
    workflow = StateGraph(GraphState)
    
    # Stage2 노드들 추가
    workflow.add_node("question_iterator", question_iterator_node)
    workflow.add_node("increment_question", increment_question_index)
    workflow.add_node("word_processor", word_processing_node)
    workflow.add_node("sentence_processor", sentence_processing_node) 
    workflow.add_node("etc_processor", etc_processing_node)
    
    # 시작점: question_iterator로 시작
    workflow.add_edge(START, "question_iterator")
    
    # question_iterator → 조건부 라우팅
    workflow.add_conditional_edges(
        "question_iterator",
        question_type_router,  # 라우터 함수
        {
            "WORD": "word_processor",
            "SENTENCE": "sentence_processor", 
            "ETC": "etc_processor",
            "__END__": END
        }
    )
    
    # 각 처리 노드 → increment_question
    workflow.add_edge("word_processor", "increment_question")
    workflow.add_edge("sentence_processor", "increment_question")
    workflow.add_edge("etc_processor", "increment_question")
    
    # increment_question → 계속 처리할지 판단
    workflow.add_conditional_edges(
        "increment_question",
        should_continue_router,  # 계속 처리할지 판단
        {
            "CONTINUE": "question_iterator",  # 다음 질문으로 루프
            "__END__": END  # 모든 질문 처리 완료
        }
    )
    
    return workflow

def run_stage2_pipeline(stage1_result: Dict[str, Any]):
    """Stage2 파이프라인 실행"""
    
    print("🚀 Stage2 파이프라인 시작")
    
    # Stage2 워크플로우 생성 및 컴파일
    workflow = create_stage2_workflow()
    app = workflow.compile()
    
    # Stage1 결과를 Stage2 상태로 변환
    stage2_state = convert_stage1_to_stage2(stage1_result)
    
    try:
        # 워크플로우 실행 - LangGraph가 자동으로 루프 처리
        result = app.invoke(stage2_state)
        
        print("✅ Stage2 파이프라인 완료!")
        return result
        
    except Exception as e:
        print(f"❌ Stage2 파이프라인 실패: {str(e)}")
        return {"error": str(e)}

# 핵심 포인트 설명:
"""
🔄 LangGraph에서 루프 처리 방식:

1. **조건부 엣지 (Conditional Edges)**:
   - `add_conditional_edges()`로 라우터 함수 지정
   - 라우터 함수가 반환하는 값에 따라 다음 노드 결정

2. **자동 루프**:
   - increment_question → should_continue_router → question_iterator
   - should_continue_router가 "CONTINUE" 반환하면 자동으로 다시 question_iterator로
   - "__END__" 반환하면 자연스럽게 종료

3. **수동 iteration 불필요**:
   - while 루프나 for 루프 없음
   - LangGraph 엔진이 조건부 엣지를 따라 자동 실행
   - 각 노드는 단순히 상태를 업데이트하고 다음 노드로 전달

4. **상태 관리**:
   - 모든 상태는 GraphState 객체로 관리
   - 각 노드가 상태를 수정하면 자동으로 다음 노드로 전달
   - 메모리 효율적이고 추적 가능

5. **실제 실행 흐름**:
   START → question_iterator → [WORD/SENTENCE/ETC 처리] 
   → increment_question → [CONTINUE/END 판단] → 루프 또는 종료
"""