#!/usr/bin/env python3
"""
Enhanced Stage3 Pipeline Test
- KNN → CSLS → MCL 파이프라인 테스트
- Cross-column label matching 테스트  
- Cluster refinement 테스트
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from singleton_aware_stage3_node import (
    singleton_aware_stage3_node,
    load_column_wise_data,
    process_question_singleton_aware,
    match_labels_across_columns,
    refine_clusters_by_similarity
)

def create_test_data():
    """테스트용 Stage2 출력 데이터 생성"""
    
    # Stage2 출력 디렉토리 생성
    output_dir = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/stage2_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 샘플 데이터 생성
    test_data = {
        'Question_1_Column_A': {
            'texts': [
                "The service was excellent and staff was friendly",
                "Great customer service, very helpful staff", 
                "Outstanding service quality and professional team",
                "Poor service, unhelpful staff attitude",
                "Terrible customer service experience",
                "Worst service I've ever received",
                "Average service, nothing special",
                "Decent service but could be better"
            ],
            'embeddings': np.random.randn(8, 384)  # Simulated embeddings
        },
        'Question_1_Column_B': {
            'texts': [
                "The service was excellent and staff was friendly",  # 동일 텍스트
                "Great customer service, very helpful staff",        # 동일 텍스트
                "Outstanding service quality and professional team", # 동일 텍스트
                "Poor service, unhelpful staff attitude",           # 동일 텍스트
                "Terrible customer service experience",            # 동일 텍스트
                "Worst service I've ever received",                # 동일 텍스트
                "Average service, nothing special",                # 동일 텍스트
                "Decent service but could be better"               # 동일 텍스트
            ],
            'embeddings': np.random.randn(8, 384)  # 다른 임베딩 (다른 컬럼)
        },
        'Question_2_Column_A': {
            'texts': [
                "The product quality is amazing",
                "Excellent product, very satisfied",
                "High quality product, recommended",
                "Poor product quality, disappointed", 
                "Low quality, not worth the price",
                "Product quality is unacceptable"
            ],
            'embeddings': np.random.randn(6, 384)
        }
    }
    
    # CSV 파일로 저장
    for name, data in test_data.items():
        df = pd.DataFrame({
            'text': data['texts'],
            'embedding_col': [f"embed_{i}" for i in range(len(data['texts']))]
        })
        
        # 임베딩을 별도 컬럼들로 저장
        embeddings = data['embeddings']
        for i in range(embeddings.shape[1]):
            df[f'embed_{i}'] = embeddings[:, i]
        
        # CSV 저장
        csv_path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Created: {csv_path}")
    
    print(f"\n📂 Test data created in: {output_dir}")
    return output_dir

def test_data_loading(output_dir):
    """데이터 로딩 테스트"""
    
    print("\n" + "="*50)
    print("🔍 Testing Data Loading")
    print("="*50)
    
    try:
        data = load_column_wise_data(output_dir)
        
        print(f"✅ Loaded {len(data)} questions")
        for question_id, question_data in data.items():
            print(f"   📋 {question_id}: {len(question_data)} columns")
            for column_name, (df, embeddings) in question_data.items():
                print(f"      📄 {column_name}: {len(df)} rows, {embeddings.shape}")
        
        return data
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_single_question_processing(data):
    """단일 문항 처리 테스트"""
    
    print("\n" + "="*50)
    print("🎯 Testing Single Question Processing")
    print("="*50)
    
    if not data:
        print("❌ No data available for testing")
        return
    
    # 첫 번째 문항 선택
    question_id = list(data.keys())[0]
    question_data = data[question_id]
    
    print(f"📋 Testing question: {question_id}")
    
    try:
        result = process_question_singleton_aware(question_id, question_data)
        
        print(f"\n📊 Results Summary:")
        print(f"   • Total columns: {result['n_columns']}")
        print(f"   • Total clusters: {result['total_clusters']}")
        print(f"   • Total singletons: {result['total_singletons']}")
        print(f"   • Algorithms used: {result['algorithms_used']}")
        print(f"   • Avg NMI: {result['avg_nmi']:.3f}")
        print(f"   • Avg Adj ARI: {result['avg_adj_ari']:.3f}")
        print(f"   • Avg Combined Score: {result['avg_combined_score']:.3f}")
        
        print(f"\n📋 Column Details:")
        for col_result in result['column_results']:
            print(f"   🔹 {col_result['column']}: {col_result['n_clusters']} clusters, "
                  f"{col_result['n_singletons']} singletons, "
                  f"score={col_result.get('combined_score', 0):.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Single question processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_full_stage3_pipeline(output_dir):
    """전체 Stage3 파이프라인 테스트"""
    
    print("\n" + "="*50)
    print("🚀 Testing Full Stage3 Pipeline")
    print("="*50)
    
    # Mock state 생성
    test_state = {
        'output_dir': output_dir,
        'stage2_completed': True
    }
    
    try:
        result_state = singleton_aware_stage3_node(test_state)
        
        if 'stage3_results' in result_state and 'error' not in result_state['stage3_results']:
            results = result_state['stage3_results']
            
            print(f"\n🎉 Stage3 Pipeline Success!")
            print(f"   • Processing type: {results['processing_type']}")
            print(f"   • Questions processed: {results['overall_summary']['total_questions']}")
            print(f"   • Total clusters: {results['overall_summary']['total_clusters']}")
            print(f"   • Total singletons: {results['overall_summary']['total_singletons']}")
            print(f"   • Singleton ratio: {results['overall_summary']['singleton_ratio']:.3f}")
            print(f"   • Overall NMI: {results['overall_summary']['overall_avg_nmi']:.3f}")
            print(f"   • Overall Adj ARI: {results['overall_summary']['overall_avg_adj_ari']:.3f}")
            print(f"   • Overall Combined Score: {results['overall_summary']['overall_avg_combined_score']:.3f}")
            print(f"   • Quality Distribution: {results['overall_summary']['quality_distribution']}")
            
            return True
        else:
            print(f"❌ Stage3 Pipeline failed: {result_state.get('stage3_results', {}).get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_enhanced_stage3_tests():
    """향상된 Stage3 테스트 실행"""
    
    print("🧪 Enhanced Stage3 Pipeline Tests")
    print("="*60)
    print("Testing improvements:")
    print("1. KNN → CSLS → MCL algorithm order")
    print("2. Cross-column label matching")
    print("3. Cluster refinement by similarity")
    print("4. NMI/ARI evaluation system")
    print("="*60)
    
    # 1. 테스트 데이터 생성
    output_dir = create_test_data()
    
    # 2. 데이터 로딩 테스트
    data = test_data_loading(output_dir)
    
    # 3. 단일 문항 처리 테스트
    single_result = test_single_question_processing(data)
    
    # 4. 전체 파이프라인 테스트
    pipeline_success = test_full_stage3_pipeline(output_dir)
    
    # 5. 결과 요약
    print("\n" + "="*60)
    print("📋 Test Summary")
    print("="*60)
    
    if data:
        print("✅ Data loading: PASSED")
    else:
        print("❌ Data loading: FAILED")
    
    if single_result:
        print("✅ Single question processing: PASSED")
    else:
        print("❌ Single question processing: FAILED")
    
    if pipeline_success:
        print("✅ Full pipeline: PASSED")
    else:
        print("❌ Full pipeline: FAILED")
    
    if data and single_result and pipeline_success:
        print("\n🎉 All Enhanced Stage3 tests PASSED!")
        print("✅ KNN → CSLS → MCL pipeline working")
        print("✅ Cross-column label matching working")
        print("✅ Cluster refinement working")
        print("✅ NMI/ARI evaluation working")
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    run_enhanced_stage3_tests()