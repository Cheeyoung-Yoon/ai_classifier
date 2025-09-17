# nodes/classifications/sentence.py
"""
SENTENCE 타입 질문 처리 노드 (일반적인 문장 처리)
- 각 문장은 S,V,C로 분해하여 키워드 추출
- 각 문장은 추출된 S,V,C 기반, 원자적 의미를 가지도록 최대 3개의 문장을 생성
- 각 문장을 embedding 처리
- embedding 한 결과 KNN -> CSLS -> MCL 기반 mapping 진행
- mapping 한 결과를 key로 묶음
- key 기반하여 유의미한 매핑이 있는지 다시 확인
- LLM 확인하여 동일 의미 또는 오탈자로 인한 group 있는지 확인
- LLM 활용하여 S,V 기반 요약된 key에 대한 요약 문장 생성
- 매핑 테이블 생성
"""

from typing import Dict, Any, List
import pandas as pd
from graph.state import GraphState
from core.classification.depend_classfication import DependentClassifier  # 문장 처리 로직은 동일
from core.classification.embed import EmbeddingProcessor

def sentence_classification_node(state: GraphState) -> Dict[str, Any]:
    """
    SENTENCE 타입 질문 처리 (의존성 없는 일반 문장)
    
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
                # 빈 값이 아닌 문장 데이터만 추출
                col_data = df[col].dropna().astype(str).tolist()
                sentence_data.extend([(sentence.strip(), col) for sentence in col_data if sentence.strip()])
        
        if not sentence_data:
            return {
                "warning": f"No valid sentence data found for QID {focus_qid}",
                "classification_result": {}
            }
        
        # 3. 문장 분류 처리 (의존성 없는 처리)
        classifier = DependentClassifier()  # 동일한 문장 처리 로직 사용
        
        # 문장만 추출 (중복 제거)
        unique_sentences = list(set([sentence for sentence, _ in sentence_data]))
        
        # 4. S,V,C 분해 및 키워드 추출
        parsed_sentences = classifier.parse_sentences_to_svc(unique_sentences)
        
        # 5. 원자적 의미의 문장 생성 (최대 3개씩) - 의존성 정보 없이
        atomic_sentences = classifier.generate_atomic_sentences(parsed_sentences, context_info=None)
        
        # 6. Embedding 처리
        embedder = EmbeddingProcessor()
        embeddings = embedder.embed_sentences(atomic_sentences)
        
        # 7. KNN -> CSLS -> MCL 기반 매핑
        clusters = classifier.cluster_sentences_advanced(embeddings, atomic_sentences)
        
        # 8. 유의미한 매핑 확인
        validated_clusters = classifier.validate_clusters(clusters)
        
        # 9. LLM을 통한 클러스터 검증 및 정제
        refined_clusters = classifier.refine_clusters_with_llm(validated_clusters, context_info=None)
        
        # 10. S,V 기반 요약 문장 생성
        summary_clusters = classifier.generate_cluster_summaries(refined_clusters)
        
        # 11. 최종 매핑 테이블 생성
        mapping_table = classifier.create_mapping_table(summary_clusters, context_info=None)
        
        return {
            "qid": focus_qid,
            "question_type": "SENTENCE", 
            "matched_columns": matched_columns,
            "total_sentences": len(unique_sentences),
            "total_atomic_sentences": len(atomic_sentences),
            "total_clusters": len(summary_clusters),
            "classification_result": mapping_table,
            "clusters": summary_clusters,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": f"Sentence classification failed for QID {focus_qid}: {str(e)}",
            "classification_result": {},
            "status": "failed"
        }
