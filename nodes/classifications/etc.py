# nodes/classifications/etc.py
"""
ETC 타입 질문 처리 노드
- 기타 선택지나 예외적인 경우 처리
- 기본적인 텍스트 분류 적용
"""

from typing import Dict, Any, List
import pandas as pd
from graph.state import GraphState
from core.classification.word_classification import WordClassifier
from core.classification.embed import EmbeddingProcessor

def etc_classification_node(state: GraphState) -> Dict[str, Any]:
    """
    ETC 타입 질문 처리
    
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
        text_data = []
        for col in matched_columns:
            if col in df.columns:
                col_data = df[col].dropna().astype(str).tolist()
                text_data.extend([(text.strip(), col) for text in col_data if text.strip()])
        
        if not text_data:
            return {
                "warning": f"No valid text data found for QID {focus_qid}",
                "classification_result": {}
            }
        
        # 3. 기타 분류 처리 (단순한 텍스트 분류 적용)
        classifier = WordClassifier()  # 단순 분류 사용
        
        # 텍스트만 추출 (중복 제거)
        unique_texts = list(set([text for text, _ in text_data]))
        
        # 4. Embedding 처리
        embedder = EmbeddingProcessor()
        embeddings = embedder.embed_words(unique_texts)  # 단어 수준 임베딩
        
        # 5. 기본 클러스터링
        clusters = classifier.cluster_words(embeddings, unique_texts)
        
        # 6. LLM을 통한 클러스터 검증
        refined_clusters = classifier.refine_clusters_with_llm(clusters)
        
        # 7. 최종 매핑 테이블 생성
        mapping_table = classifier.create_mapping_table(refined_clusters)
        
        return {
            "qid": focus_qid,
            "question_type": "ETC",
            "matched_columns": matched_columns,
            "total_texts": len(unique_texts),
            "total_clusters": len(refined_clusters),
            "classification_result": mapping_table,
            "clusters": refined_clusters,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": f"ETC classification failed for QID {focus_qid}: {str(e)}",
            "classification_result": {},
            "status": "failed"
        }
