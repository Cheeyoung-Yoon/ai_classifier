# nodes/classifications/word.py
"""
WORD 타입 질문 분류 처리 노드
개념/이미지 등의 단어 분류
"""

import sys
import os
sys.path.append("/home/cyyoon/test_area/ai_text_classification/2.langgraph")

from typing import Dict, Any

def word_classification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    WORD 타입 질문 분류 처리
    """
    focus_qid = state.get("focus_qid")
    current_question_info = state.get("current_question_info", {})
    
    print(f"Processing WORD classification for QID: {focus_qid}")
    
    # 현재는 Mock 처리
    classification_result = {
        "classification_type": "WORD",
        "qid": focus_qid,
        "status": "processed",
        "method": "word_embedding_mcl",
        "timestamp": "mock_timestamp"
    }
    
    # 결과를 상태에 저장
    if "classification_results" not in state:
        state["classification_results"] = {}
    
    state["classification_results"][focus_qid] = classification_result
    
    return {
        "classification_results": state["classification_results"],
        "message": f"WORD classification completed for {focus_qid}"
    }

from typing import Dict, Any, List
import pandas as pd
from graph.state import GraphState
from core.classification.embed import EmbeddingProcessor
from core.classification.word_classification import WordClassifier
from utils.data_utils import DataHelper

def word_classification_node(state: GraphState) -> Dict[str, Any]:
    """
    WORD 타입 질문 처리
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        처리 결과가 포함된 딕셔너리
    """
    try:
        # 현재 처리 중인 질문 정보 가져오기
        focus_qid = state.get("focus_qid")
        integrated_map = state.get("integrated_map", {})
        data_path = state.get("data_path")
        
        if not focus_qid or focus_qid not in integrated_map:
            return {
                "error": f"Focus QID {focus_qid} not found in integrated_map",
                "classification_result": {}
            }
            
        question_info = integrated_map[focus_qid]
        matched_columns = question_info.get("matched_columns", [])
        
        if not matched_columns:
            return {
                "warning": f"No matched columns for QID {focus_qid}",
                "classification_result": {}
            }
        
        # 1. 데이터 로드
        df = pd.read_csv(data_path)
        
        # 2. 매칭된 컬럼에서 데이터 추출
        word_data = []
        for col in matched_columns:
            if col in df.columns:
                # 빈 값이 아닌 데이터만 추출
                col_data = df[col].dropna().astype(str).tolist()
                word_data.extend([(word.strip(), col) for word in col_data if word.strip()])
        
        if not word_data:
            return {
                "warning": f"No valid word data found for QID {focus_qid}",
                "classification_result": {}
            }
        
        # 3. 단어 분류 처리
        classifier = WordClassifier()
        
        # 단어만 추출 (중복 제거)
        unique_words = list(set([word for word, _ in word_data]))
        
        # 4. Embedding 처리
        embedder = EmbeddingProcessor()
        embeddings = embedder.embed_words(unique_words)
        
        # 5. MCL 기반 클러스터링
        clusters = classifier.cluster_words(embeddings, unique_words)
        
        # 6. LLM을 통한 클러스터 검증 및 정제
        refined_clusters = classifier.refine_clusters_with_llm(clusters)
        
        # 7. 최종 매핑 테이블 생성
        mapping_table = classifier.create_mapping_table(refined_clusters)
        
        return {
            "qid": focus_qid,
            "question_type": "WORD",
            "matched_columns": matched_columns,
            "total_words": len(unique_words),
            "total_clusters": len(refined_clusters),
            "classification_result": mapping_table,
            "clusters": refined_clusters,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": f"Word classification failed for QID {focus_qid}: {str(e)}",
            "classification_result": {},
            "status": "failed"
        }
