# nodes/classifications/depend.py
"""
DEPEND 타입 질문 분류 처리 노드
종속형 질문 분류
"""

import sys
import os
sys.path.append("/home/cyyoon/test_area/ai_text_classification/2.langgraph")

from typing import Dict, Any

def depend_classification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    DEPEND 타입 질문 분류 처리
    """
    focus_qid = state.get("focus_qid")
    current_question_info = state.get("current_question_info", {})
    
    print(f"Processing DEPEND classification for QID: {focus_qid}")
    
    # 현재는 Mock 처리
    classification_result = {
        "classification_type": "DEPEND",
        "qid": focus_qid,
        "status": "processed",
        "method": "dependency_analysis",
        "timestamp": "mock_timestamp"
    }
    
    # 결과를 상태에 저장
    if "classification_results" not in state:
        state["classification_results"] = {}
    
    state["classification_results"][focus_qid] = classification_result
    
    return {
        "classification_results": state["classification_results"],
        "message": f"DEPEND classification completed for {focus_qid}"
    }
