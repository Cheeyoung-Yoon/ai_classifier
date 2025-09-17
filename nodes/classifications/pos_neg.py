# nodes/classifications/pos_neg.py
"""
POS_NEG 타입 질문 처리 노드
- 긍정/부정 이유를 묻는 질문 처리
- 감정 분석과 함께 문장 분류 진행
"""

from typing import Dict, Any, List
import pandas as pd
from graph.state import GraphState
from core.classification.depend_classfication import DependentClassifier
from core.classification.embed import EmbeddingProcessor

def pos_neg_classification_node(state: GraphState) -> Dict[str, Any]:
    """
    POS_NEG 타입 질문 처리
    
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
        
        # 2. 매칭된 컬럼에서 문장 데이터 추출
        sentence_data = []
        for col in matched_columns:
            if col in df.columns:
                col_data = df[col].dropna().astype(str).tolist()
                sentence_data.extend([(sentence.strip(), col) for sentence in col_data if sentence.strip()])
        
        if not sentence_data:
            return {
                "warning": f"No valid sentence data found for QID {focus_qid}",
                "classification_result": {}
            }
        
        # 3. 긍정/부정 분류 처리
        classifier = DependentClassifier()
        unique_sentences = list(set([sentence for sentence, _ in sentence_data]))
        
        # 4. 감정 분석 추가 (긍정/부정 판별)
        sentiment_analysis = classifier.analyze_sentiment(unique_sentences)
        
        # 5. S,V,C 분해 및 키워드 추출 (감정 정보 포함)
        parsed_sentences = classifier.parse_sentences_to_svc_with_sentiment(unique_sentences, sentiment_analysis)
        
        # 6. 원자적 의미의 문장 생성
        atomic_sentences = classifier.generate_atomic_sentences_with_sentiment(parsed_sentences)
        
        # 7. Embedding 처리
        embedder = EmbeddingProcessor()
        embeddings = embedder.embed_sentences(atomic_sentences)
        
        # 8. 감정별 클러스터링
        pos_neg_clusters = classifier.cluster_by_sentiment(embeddings, atomic_sentences, sentiment_analysis)
        
        # 9. LLM을 통한 클러스터 검증
        refined_clusters = classifier.refine_pos_neg_clusters_with_llm(pos_neg_clusters)
        
        # 10. 최종 매핑 테이블 생성
        mapping_table = classifier.create_pos_neg_mapping_table(refined_clusters)
        
        return {
            "qid": focus_qid,
            "question_type": "POS_NEG",
            "matched_columns": matched_columns,
            "total_sentences": len(unique_sentences),
            "total_clusters": len(refined_clusters),
            "positive_clusters": len([c for c in refined_clusters if c.get("sentiment") == "positive"]),
            "negative_clusters": len([c for c in refined_clusters if c.get("sentiment") == "negative"]),
            "classification_result": mapping_table,
            "clusters": refined_clusters,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": f"Pos/Neg classification failed for QID {focus_qid}: {str(e)}",
            "classification_result": {},
            "status": "failed"
        }
