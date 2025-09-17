# nodes/result_collector.py
"""
분류 결과 수집 및 통합 노드
- 각 질문별 분류 결과를 수집
- 전체 처리 결과를 통합하여 최종 결과 생성
"""

from typing import Dict, Any
from graph.state import GraphState

def collect_classification_result(state: GraphState) -> Dict[str, Any]:
    """
    현재 질문의 분류 결과를 수집하고 전체 결과에 추가
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태
    """
    try:
        # 현재 질문 정보
        focus_qid = state.get("focus_qid")
        if not focus_qid:
            return {"error": "No focus_qid found"}
        
        # 기존 결과 가져오기
        classification_results = state.get("classification_results", {})
        
        # 현재 질문의 결과 찾기 (분류 노드들이 반환한 결과)
        current_result = None
        
        # 각 분류 타입별로 결과 확인
        for result_key in ["word_result", "depend_result", "sentence_result", "pos_neg_result", "etc_result"]:
            if result_key in state:
                current_result = state[result_key]
                break
        
        if current_result and current_result.get("status") == "success":
            # 성공적인 결과만 저장
            classification_results[focus_qid] = {
                "qid": focus_qid,
                "question_type": current_result.get("question_type"),
                "matched_columns": current_result.get("matched_columns", []),
                "classification_result": current_result.get("classification_result", {}),
                "clusters": current_result.get("clusters", []),
                "total_items": current_result.get("total_words", current_result.get("total_sentences", 0)),
                "total_clusters": current_result.get("total_clusters", 0),
                "processing_time": current_result.get("processing_time"),
                "status": "success"
            }
            
            message = f"Collected result for QID {focus_qid}: {current_result.get('total_clusters', 0)} clusters"
        else:
            # 실패한 경우도 기록
            classification_results[focus_qid] = {
                "qid": focus_qid,
                "status": "failed",
                "error": current_result.get("error", "Unknown error") if current_result else "No result found"
            }
            
            message = f"Failed to process QID {focus_qid}"
        
        return {
            "classification_results": classification_results,
            "message": message,
            "processed_questions": len(classification_results)
        }
        
    except Exception as e:
        return {
            "error": f"Result collection failed: {str(e)}",
            "message": f"Failed to collect result for QID {focus_qid}"
        }

def finalize_results(state: GraphState) -> Dict[str, Any]:
    """
    모든 처리가 완료된 후 최종 결과를 정리
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        최종 처리 결과
    """
    try:
        classification_results = state.get("classification_results", {})
        total_questions = state.get("total_questions", 0)
        
        # 성공/실패 통계
        successful_results = {qid: result for qid, result in classification_results.items() 
                            if result.get("status") == "success"}
        failed_results = {qid: result for qid, result in classification_results.items() 
                         if result.get("status") == "failed"}
        
        # 타입별 통계
        type_stats = {}
        for qid, result in successful_results.items():
            qtype = result.get("question_type", "unknown")
            if qtype not in type_stats:
                type_stats[qtype] = {"count": 0, "total_clusters": 0}
            type_stats[qtype]["count"] += 1
            type_stats[qtype]["total_clusters"] += result.get("total_clusters", 0)
        
        final_result = {
            "total_questions": total_questions,
            "processed_questions": len(classification_results),
            "successful_questions": len(successful_results),
            "failed_questions": len(failed_results),
            "success_rate": len(successful_results) / max(total_questions, 1) * 100,
            "type_statistics": type_stats,
            "classification_results": classification_results,
            "successful_results": successful_results,
            "failed_results": failed_results,
            "status": "completed"
        }
        
        return {
            "final_classification_results": final_result,
            "processing_complete": True,
            "message": f"Processing completed: {len(successful_results)}/{total_questions} questions successfully classified"
        }
        
    except Exception as e:
        return {
            "error": f"Result finalization failed: {str(e)}",
            "processing_complete": True,
            "status": "failed"
        }
