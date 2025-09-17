# graph/stage2_graph.py - 실제 Stage2 LangGraph 워크플로우

import sys
import os

# Add project root to path
project_root = "/home/cyyoon/test_area/ai_text_classification/2.langgraph"
sys.path.insert(0, project_root)

from langgraph.graph import StateGraph, START, END
from typing import Dict, Any

# Import state and utilities
from graph.state import GraphState
from utils.stage_converter import convert_stage1_to_stage2

# Import routers
from router.qytpe_router import question_type_router, should_continue_router

# Import nodes
from nodes.question_iterator import question_iterator_node, increment_question_index

# 임시 플레이스홀더 노드들 (실제로는 별도 파일에 구현)
def word_processing_node(state):
    """WORD 타입 질문 처리 노드 (플레이스홀더)"""
    print(f"🔤 WORD 처리: {state.get('focus_qid')}")
    return state

def sentence_processing_node(state):
    """SENTENCE 타입 질문 처리 노드 (플레이스홀더)"""  
    print(f"📝 SENTENCE 처리: {state.get('focus_qid')}")
    return state

def etc_processing_node(state):
    """ETC 타입 질문 처리 노드 (플레이스홀더)"""
    print(f"❓ ETC 처리: {state.get('focus_qid')}")
    return state

def create_stage2_workflow() -> StateGraph:
    """Stage2 실제 LangGraph 워크플로우 구현
    
    🔄 워크플로우 흐름:
    START → question_iterator → [조건부 라우팅] → 처리 노드 → increment → [루프/종료]
    """
    
    workflow = StateGraph(GraphState)
    
    # Stage2 노드들 추가
    workflow.add_node("question_iterator", question_iterator_node)
    workflow.add_node("increment_question", increment_question_index)
    workflow.add_node("word_processor", word_processing_node)
    workflow.add_node("sentence_processor", sentence_processing_node) 
    workflow.add_node("etc_processor", etc_processing_node)
    
    # 🚀 시작점: question_iterator로 시작
    workflow.add_edge(START, "question_iterator")
    
    # 🎯 question_iterator → 조건부 라우팅 (핵심!)
    workflow.add_conditional_edges(
        "question_iterator",  # 현재 노드
        question_type_router,  # 라우터 함수 (WORD/SENTENCE/ETC/__END__ 반환)
        {
            # 라우터 반환값 → 다음 노드 매핑
            "WORD": "word_processor",
            "SENTENCE": "sentence_processor", 
            "ETC": "etc_processor",
            "__END__": END  # 모든 질문 처리 완료시 종료
        }
    )
    
    # 📝 각 처리 노드 → increment_question (고정 엣지)
    workflow.add_edge("word_processor", "increment_question")
    workflow.add_edge("sentence_processor", "increment_question")
    workflow.add_edge("etc_processor", "increment_question")
    
    # 🔄 increment_question → 루프 제어 (핵심!)
    workflow.add_conditional_edges(
        "increment_question",  # 현재 노드
        should_continue_router,  # 계속 처리할지 판단 (CONTINUE/__END__ 반환)
        {
            # 라우터 반환값 → 다음 노드 매핑
            "CONTINUE": "question_iterator",  # 🔄 다음 질문으로 루프!
            "__END__": END  # 🏁 모든 질문 처리 완료
        }
    )
    
    return workflow

def run_stage2_pipeline(stage1_result: Dict[str, Any]) -> Dict[str, Any]:
    """Stage2 파이프라인 실행
    
    Args:
        stage1_result: Stage1에서 나온 결과 (matched_questions 등 포함)
        
    Returns:
        Stage2 처리 완료된 결과
    """
    
    print("\n" + "="*60)
    print("🚀 Stage2 LangGraph 파이프라인 시작")
    print("="*60)
    
    # Stage2 워크플로우 생성 및 컴파일
    workflow = create_stage2_workflow()
    app = workflow.compile()
    
    # Stage1 결과를 Stage2 상태로 변환
    print("🔄 Stage1 → Stage2 상태 변환...")
    stage2_state = convert_stage1_to_stage2(stage1_result)
    
    print(f"📊 처리할 질문 수: {stage2_state.get('total_questions', 0)}")
    
    try:
        # 🚀 워크플로우 실행 - LangGraph가 자동으로 루프 처리!
        print("🔄 LangGraph 워크플로우 실행 중...")
        result = app.invoke(stage2_state)
        
        print("✅ Stage2 파이프라인 완료!")
        
        # 결과 요약
        processed_count = len(result.get('classification_results', {}))
        total_count = result.get('total_questions', 0)
        
        print(f"📊 처리 결과:")
        print(f"  - 처리된 질문: {processed_count}/{total_count}")
        print(f"  - 처리 완료: {result.get('processing_complete', False)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Stage2 파이프라인 실패: {str(e)}")
        return {"error": str(e)}

def run_complete_pipeline(project_name: str, survey_filename: str, data_filename: str):
    """Stage1 + Stage2 완전한 파이프라인 실행"""
    
    print("🚀 완전한 파이프라인 실행 (Stage1 → Stage2)")
    
    # Stage1 실행 (기존 코드)
    from graph.graph import run_pipeline as run_stage1
    stage1_result = run_stage1(project_name, survey_filename, data_filename)
    
    if stage1_result.get('error'):
        print(f"❌ Stage1 실패: {stage1_result['error']}")
        return stage1_result
    
    # Stage2 실행
    stage2_result = run_stage2_pipeline(stage1_result)
    
    return stage2_result

# 핵심 차이점 정리:
"""
🔄 수동 루프 vs LangGraph 자동 루프:

📋 현재 테스트 코드 (수동):
```python
while iteration < max_iterations:  # 수동 루프
    # 1. question_iterator_node(state)
    # 2. question_type_router(state) 
    # 3. [처리 노드 시뮬레이션]
    # 4. increment_question_index(state)
    # 5. should_continue_router(state)
    iteration += 1
```

🚀 실제 LangGraph (자동):
```python
# 노드와 엣지만 정의, 루프는 LangGraph가 자동 처리
workflow.add_conditional_edges(
    "increment_question",
    should_continue_router,
    {
        "CONTINUE": "question_iterator",  # 자동 루프!
        "__END__": END
    }
)
```

✨ LangGraph의 장점:
1. **자동 루프**: 조건부 엣지로 자연스러운 루프
2. **상태 관리**: GraphState 자동 전달
3. **추적 가능**: 각 실행 단계 추적
4. **확장 가능**: 노드 추가/수정 용이
5. **에러 처리**: 내장된 예외 처리
"""

if __name__ == "__main__":
    # 테스트 실행
    result = run_complete_pipeline(
        project_name="test",
        survey_filename="test.txt", 
        data_filename="-SUV_776부.xlsx"
    )
    print(f"\n📋 최종 결과: {list(result.keys())}")