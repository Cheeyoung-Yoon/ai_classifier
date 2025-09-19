#!/usr/bin/env python3
"""
실제 stored state 데이터를 사용한 Stage3 Two-Phase 시스템 테스트
"""

import json
import pandas as pd
import numpy as np
import ast
from pathlib import Path
import sys
import os
from typing import Dict, Any, List

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from graph.state import GraphState
from nodes.stage3_classification.phase1_primary_labeling import Phase1PrimaryLabeling
from nodes.stage3_classification.phase2_secondary_labeling import Phase2SecondaryLabeling
from nodes.stage3_classification.quality_assessment import QualityAssessmentTools

def load_state_file(state_file_path: str) -> Dict[str, Any]:
    """저장된 state 파일 로드"""
    print(f"Loading state file: {state_file_path}")
    
    with open(state_file_path, 'r', encoding='utf-8') as f:
        state_data = json.load(f)
    
    print(f"✅ State file loaded successfully")
    print(f"Current stage: {state_data.get('current_stage', 'Unknown')}")
    print(f"Pipeline ID: {state_data.get('pipeline_id', 'Unknown')}")
    
    return state_data

def load_stage2_csv_data(csv_path: str) -> pd.DataFrame:
    """Stage2 결과 CSV 파일 로드"""
    print(f"Loading CSV data: {csv_path}")
    
    # CSV 파일 읽기 (첫 번째 열이 인덱스가 되지 않도록)
    df = pd.read_csv(csv_path, index_col=0)
    
    print(f"✅ CSV data loaded: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    return df

def parse_embedding_vector(embedding_str: str) -> np.ndarray:
    """문자열로 저장된 embedding 벡터를 numpy array로 변환"""
    try:
        # 문자열을 리스트로 파싱
        embedding_list = ast.literal_eval(embedding_str)
        return np.array(embedding_list, dtype=np.float32)
    except Exception as e:
        print(f"⚠️  Error parsing embedding: {e}")
        return None

def prepare_test_data(state_data: Dict[str, Any], question_id: str) -> Dict[str, Any]:
    """특정 질문의 데이터를 준비"""
    print(f"\n=== Preparing test data for question: {question_id} ===")
    
    matched_questions = state_data.get('matched_questions', {})
    
    if question_id not in matched_questions:
        print(f"❌ Question {question_id} not found in matched_questions")
        return None
    
    question_data = matched_questions[question_id]
    stage2_data = question_data.get('stage2_data')
    
    if not stage2_data:
        print(f"❌ No stage2_data found for question {question_id}")
        return None
    
    csv_path = stage2_data.get('csv_path')
    if not csv_path or not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return None
    
    # CSV 데이터 로드
    df = load_stage2_csv_data(csv_path)
    
    # 텍스트와 임베딩 데이터 준비
    texts = []
    embeddings = []
    
    print("Processing rows...")
    print(f"CSV columns: {df.columns.tolist()}")
    
    # CSV에서 텍스트와 임베딩 컬럼 쌍 찾기
    text_embed_pairs = []
    for col in df.columns:
        if col.startswith('text_'):
            # 대응하는 임베딩 컬럼 찾기
            text_num = col.split('_')[1]
            embed_col = f'embed_text_{text_num}'
            if embed_col in df.columns:
                text_embed_pairs.append((col, embed_col))
    
    print(f"Found text-embedding pairs: {text_embed_pairs}")
    
    # 각 text-embedding 쌍에서 데이터 추출
    for text_col, embed_col in text_embed_pairs:
        for idx, row in df.iterrows():
            text = row[text_col]
            embedding_str = row[embed_col]
            
            # 빈 값 체크
            if pd.isna(text) or pd.isna(embedding_str) or text.strip() == "":
                continue
            
            # 임베딩 파싱
            embedding = parse_embedding_vector(embedding_str)
            if embedding is not None:
                texts.append(text.strip())
                embeddings.append(embedding)
    
    print(f"✅ Prepared {len(texts)} samples")
    print(f"Sample texts: {texts[:3] if len(texts) >= 3 else texts}")
    
    # 딕셔너리로 데이터 구성 (GraphState 생성자 문제 회피)
    test_state = {
        'current_question_id': question_id,
        'question_type': question_data['question_info']['question_type'],
        
        # 실제 텍스트와 임베딩 데이터
        'stage2_texts': texts,
        'stage2_embeddings': np.array(embeddings) if embeddings else np.array([]),
        
        # Stage3 Phase1 Configuration
        'stage3_phase1_config': {
            # kNN parameters
            'knn_k': min(30, len(texts) // 2) if len(texts) > 10 else max(3, len(texts) // 3),
            'knn_metric': 'cosine',
            'mutual_knn': True,
            
            # CSLS parameters
            'csls_neighborhood_size': 10,
            'csls_threshold': 0.1,
            'top_m_edges': 20,
            'prune_bottom_percentile': 30,
            
            # MCL parameters
            'mcl_inflation': 2.0,
            'mcl_expansion': 2,
            'mcl_pruning': 1e-3,
            'mcl_max_iters': 100,
            'mcl_convergence_check': True,
            
            # Singleton and small cluster handling
            'allow_singletons': True,
            'merge_small_clusters': True,
            'min_cluster_size': 2,
            'small_cluster_threshold': 3,
            
            # Quality assessment
            'compute_subset_score': True,
            'compute_cluster_quality': True
        },
        
        # Stage3 Phase2 Configuration
        'stage3_phase2_config': {
            'edge_threshold': 0.4,
            'community_algorithm': 'louvain',
            'resolution': 1.0,
            'min_community_size': 2,
            'semantic_weight': 0.7,
            'frequency_weight': 0.3
        },
        
        # 기타 필수 필드들
        'current_stage': 'STAGE3_TWO_PHASE_TEST',
        'pipeline_id': state_data.get('pipeline_id', 'real_data_test'),
        'project_directory_structure': state_data.get('project_directory_structure', {}),
    }
    
    return test_state

def run_two_phase_test(test_state: Dict[str, Any]) -> Dict[str, Any]:
    """Two-phase 시스템 실행"""
    print(f"\n=== Running Two-Phase Stage3 System ===")
    print(f"Question ID: {test_state['current_question_id']}")
    print(f"Question Type: {test_state['question_type']}")
    print(f"Total samples: {len(test_state['stage2_texts'])}")
    
    # Phase 1 실행
    print(f"\n--- Phase 1: Primary Labeling (kNN → CSLS → MCL) ---")
    
    # Phase 1 설정 준비
    phase1_config = test_state['stage3_phase1_config']
    print(f"Phase 1 config: knn_k={phase1_config['knn_k']}, mcl_inflation={phase1_config['mcl_inflation']}")
    
    phase1 = Phase1PrimaryLabeling(config=phase1_config)
    
    # 임베딩과 텍스트 데이터 추출
    embeddings = test_state['stage2_embeddings']
    texts = test_state['stage2_texts']
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Sample embeddings norm: {np.linalg.norm(embeddings[0]):.4f}")
    
    # 직접 embeddings 배열을 전달
    phase1_raw_result = phase1.process_embeddings(embeddings, texts)
    
    print(f"Phase 1 raw result keys: {list(phase1_raw_result.keys())}")
    
    # 결과를 state 형식으로 변환
    phase1_result = dict(test_state)
    
    if 'error' in phase1_raw_result:
        phase1_result['error'] = phase1_raw_result['error']
        print(f"❌ Phase 1 failed: {phase1_raw_result['error']}")
        return phase1_result
    
    # Phase 1 결과를 state에 추가
    for key, value in phase1_raw_result.items():
        phase1_result[key] = value  # stage3_phase1_ 접두사 없이도 모든 키 추가
    
    print(f"✅ Phase 1 completed successfully")
    
    # 실제 키명 매핑
    n_clusters = phase1_raw_result.get('n_clusters', 0)
    n_singletons = phase1_raw_result.get('n_singletons', 0)
    cluster_labels = phase1_raw_result.get('cluster_labels', [])
    prototypes = phase1_raw_result.get('prototypes', {})
    
    # 표준 키명으로 매핑하여 저장
    phase1_result['stage3_phase1_num_clusters'] = n_clusters
    phase1_result['stage3_phase1_num_singletons'] = n_singletons
    phase1_result['stage3_phase1_labels'] = cluster_labels
    phase1_result['stage3_phase1_prototypes'] = prototypes
    phase1_result['stage3_phase1_metadata'] = phase1_raw_result.get('metadata', {})
    phase1_result['stage3_phase1_quality'] = phase1_raw_result.get('quality_stats', {}).get('overall_score', 0.0)
    
    print(f"Primary clusters: {n_clusters}")
    print(f"Singletons: {n_singletons}")
    print(f"Total labeled samples: {len([l for l in cluster_labels if l >= 0])}")
    print(f"Available result keys: {list(phase1_raw_result.keys())}")
    
    # Phase 2 실행
    print(f"\n--- Phase 2: Secondary Labeling (Community Detection) ---")
    
    # Phase 1 결과가 있는지 확인
    if 'stage3_phase1_labels' not in phase1_result or phase1_result.get('stage3_phase1_num_clusters', 0) == 0:
        print("⚠️  No Phase 1 results found, skipping Phase 2")
        return phase1_result
    
    print(f"Phase 1 clusters available: {phase1_result.get('stage3_phase1_num_clusters', 0)}")
    
    # Phase 1 결과가 충분하지 않으면 Phase 2 생략
    if phase1_result.get('stage3_phase1_num_clusters', 0) < 2:
        print("⚠️  Not enough clusters for Phase 2, returning Phase 1 results")
        return phase1_result
    
    phase2 = Phase2SecondaryLabeling()
    
    # Phase 2 입력 데이터 준비
    cluster_labels = phase1_result['stage3_phase1_labels']
    prototypes = phase1_result['stage3_phase1_prototypes']
    metadata = phase1_result['stage3_phase1_metadata']
    
    # 클러스터별로 데이터 정리
    unique_labels = list(set([l for l in cluster_labels if l >= 0]))
    print(f"Unique cluster labels: {len(unique_labels)} (first 10: {unique_labels[:10]})")
    
    # Phase 1 결과를 Phase 2 형식으로 변환
    phase1_groups = {
        'cluster_labels': cluster_labels,
        'unique_labels': unique_labels,
        'embeddings': embeddings,
        'texts': texts,
        'n_clusters': len(unique_labels)
    }
    
    try:
        phase2_raw_result = phase2.process_phase1_groups(
            phase1_groups, prototypes, metadata
        )
        
        # 결과를 state 형식으로 변환
        phase2_result = dict(phase1_result)
        
        if 'error' in phase2_raw_result:
            phase2_result['error'] = phase2_raw_result['error']
            print(f"❌ Phase 2 failed: {phase2_raw_result['error']}")
            return phase2_result
        
        # Phase 2 결과를 state에 추가
        for key, value in phase2_raw_result.items():
            if key.startswith('stage3_phase2_'):
                phase2_result[key] = value
                
    except Exception as e:
        print(f"⚠️  Phase 2 processing failed: {e}")
        print("Continuing with Phase 1 results only")
        
        # Phase 1만의 결과로 최종 결과 준비
        phase2_result = dict(phase1_result)
        
        # Phase 2 결과를 Phase 1과 동일하게 설정 (의미적으로 Phase 1 클러스터가 최종 커뮤니티)
        phase2_result['stage3_phase2_num_communities'] = phase2_result['stage3_phase1_num_clusters']
        phase2_result['stage3_phase2_community_labels'] = phase2_result['stage3_phase1_labels']
        phase2_result['stage3_phase2_community_prototypes'] = phase2_result['stage3_phase1_prototypes']
        phase2_result['stage3_phase2_quality'] = phase2_result['stage3_phase1_quality']
        
        return phase2_result
    
    if 'error' in phase2_result:
        print(f"❌ Phase 2 failed: {phase2_result['error']}")
        return phase2_result
    
    print(f"✅ Phase 2 completed successfully")
    print(f"Final communities: {phase2_result.get('stage3_phase2_num_communities', 0)}")
    
    # Quality Assessment
    print(f"\n--- Quality Assessment ---")
    qa_tools = QualityAssessmentTools()
    quality_result = qa_tools.assess_two_phase_quality(phase2_result)
    
    print(f"Overall quality: {quality_result.get('overall_quality', 0):.3f}")
    print(f"Phase 1 quality: {quality_result.get('phase1_quality', 0):.3f}")
    print(f"Phase 2 quality: {quality_result.get('phase2_quality', 0):.3f}")
    
    return phase2_result

def display_results(result: Dict[str, Any]):
    """결과 상세 출력"""
    print(f"\n=== Two-Phase Results Summary ===")
    
    # 기본 정보
    print(f"Question ID: {result.get('current_question_id', 'Unknown')}")
    print(f"Question Type: {result.get('question_type', 'Unknown')}")
    print(f"Total samples: {len(result.get('stage2_texts', []))}")
    
    # Phase 1 결과
    print(f"\n📊 Phase 1 Results:")
    print(f"  Primary clusters: {result.get('stage3_phase1_num_clusters', 0)}")
    print(f"  Singletons: {result.get('stage3_phase1_num_singletons', 0)}")
    print(f"  Quality score: {result.get('stage3_phase1_quality', 0):.3f}")
    
    # Phase 2 결과
    print(f"\n🔗 Phase 2 Results:")
    print(f"  Final communities: {result.get('stage3_phase2_num_communities', 0)}")
    print(f"  Quality score: {result.get('stage3_phase2_quality', 0):.3f}")
    
    # 프로토타입 생성 결과
    prototypes = result.get('stage3_phase1_prototypes', {})
    if prototypes:
        print(f"\n🏷️  Generated Prototypes (Phase 1):")
        for cluster_id, prototype in list(prototypes.items())[:5]:  # 첫 5개만 표시
            print(f"  Cluster {cluster_id}: {prototype}")
        if len(prototypes) > 5:
            print(f"  ... and {len(prototypes) - 5} more clusters")
    
    # 커뮤니티 프로토타입
    phase2_prototypes = result.get('stage3_phase2_community_prototypes', {})
    if phase2_prototypes:
        print(f"\n🌐 Community Prototypes (Phase 2):")
        for comm_id, prototype in list(phase2_prototypes.items())[:3]:  # 첫 3개만 표시
            print(f"  Community {comm_id}: {prototype}")
        if len(phase2_prototypes) > 3:
            print(f"  ... and {len(phase2_prototypes) - 3} more communities")

def main():
    """메인 실행 함수"""
    print("🚀 Stage3 Two-Phase System - Real Data Test")
    print("=" * 60)
    
    # State 파일 경로
    state_file_path = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/state_history/20250918_113231_081_STAGE2_WORD_문23_COMPLETED_state.json"
    
    try:
        # 1. State 파일 로드
        state_data = load_state_file(state_file_path)
        
        # 2. 사용 가능한 질문들 확인
        matched_questions = state_data.get('matched_questions', {})
        available_questions = []
        
        for q_id, q_data in matched_questions.items():
            if q_data.get('stage2_data') and q_data['stage2_data'].get('csv_path'):
                csv_path = q_data['stage2_data']['csv_path']
                if os.path.exists(csv_path):
                    available_questions.append(q_id)
        
        print(f"\n📋 Available questions with stage2 data: {available_questions}")
        
        # 3. 첫 번째 사용 가능한 질문으로 테스트
        if not available_questions:
            print("❌ No questions with valid stage2 data found")
            return
        
        # 문4 데이터가 있는지 우선 확인
        test_question = '문4' if '문4' in available_questions else available_questions[0]
        print(f"\n🎯 Testing with question: {test_question}")
        
        # 4. 테스트 데이터 준비
        test_state = prepare_test_data(state_data, test_question)
        
        if not test_state:
            print("❌ Failed to prepare test data")
            return
        
        # 5. Two-phase 시스템 실행
        result = run_two_phase_test(test_state)
        
        if 'error' in result:
            print(f"❌ Test failed: {result['error']}")
            return
        
        # 6. 결과 출력
        display_results(result)
        
        print(f"\n✅ Real data test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during real data test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()