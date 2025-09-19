#!/usr/bin/env python3
"""
Real Data Stage3 Pipeline Test
실제 stage2 출력 데이터를 사용한 Stage3 파이프라인 테스트
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from singleton_aware_stage3_node import singleton_aware_stage3_node

def test_with_real_data():
    """실제 Stage2 출력 데이터로 테스트"""
    
    print("🔍 Real Data Stage3 Pipeline Test")
    print("=" * 60)
    
    # 실제 Stage2 데이터 디렉토리
    real_data_dir = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/temp_data/stage2_results copy"
    
    if not os.path.exists(real_data_dir):
        print(f"❌ Real data directory not found: {real_data_dir}")
        return False
    
    # CSV 파일들 확인
    import glob
    csv_files = glob.glob(os.path.join(real_data_dir, "*.csv"))
    print(f"📂 Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        print(f"   📄 {file_name}")
    
    # Mock state 생성
    test_state = {
        'output_dir': real_data_dir,
        'stage2_completed': True
    }
    
    try:
        print(f"\n🚀 Running Stage3 Pipeline on Real Data...")
        print("=" * 60)
        
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
            
            # 문항별 상세 결과
            print(f"\n📋 Question Details:")
            for question_id, question_result in results['questions'].items():
                print(f"\n   📌 {question_id}:")
                print(f"      • Columns: {question_result['n_columns']}")
                print(f"      • Clusters: {question_result['total_clusters']}")
                print(f"      • Singletons: {question_result['total_singletons']}")
                print(f"      • Avg Combined Score: {question_result['avg_combined_score']:.3f}")
                print(f"      • Quality Distribution: {question_result['quality_distribution']}")
                
                # 컬럼별 상세
                for col_result in question_result['column_results']:
                    quality = col_result.get('quality_assessment', 'UNKNOWN')
                    score = col_result.get('combined_score', 0)
                    algorithm = col_result.get('algorithm', 'unknown')
                    print(f"         🔸 {col_result['column']}: {col_result['n_clusters']} clusters, "
                          f"{col_result['n_singletons']} singletons, {algorithm}, "
                          f"score={score:.3f} ({quality})")
            
            # 품질 기준별 분석
            print(f"\n📊 Quality Analysis (Updated Criteria):")
            print(f"   • EXCELLENT (≥0.96): Count = {results['overall_summary']['quality_distribution'].get('EXCELLENT', 0)}")
            print(f"   • GOOD (≥0.88): Count = {results['overall_summary']['quality_distribution'].get('GOOD', 0)}")
            print(f"   • FAIR (≥0.83): Count = {results['overall_summary']['quality_distribution'].get('FAIR', 0)}")
            print(f"   • IMPROVE (≥0.72): Count = {results['overall_summary']['quality_distribution'].get('IMPROVE', 0)}")
            print(f"   • FAIL (<0.72): Count = {results['overall_summary']['quality_distribution'].get('FAIL', 0)}")
            
            return True
        else:
            error_msg = result_state.get('stage3_results', {}).get('error', 'Unknown error')
            print(f"❌ Stage3 Pipeline failed: {error_msg}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_data_structure(data_dir):
    """데이터 구조 분석"""
    
    print(f"\n🔍 Data Structure Analysis")
    print("=" * 40)
    
    import glob
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    for csv_file in csv_files[:3]:  # 처음 3개 파일만 분석
        file_name = os.path.basename(csv_file)
        print(f"\n📄 {file_name}:")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"   • Rows: {len(df)}")
            print(f"   • Columns: {list(df.columns)}")
            
            # 문항 정보
            if 'question_id' in df.columns:
                unique_questions = df['question_id'].unique()
                print(f"   • Questions: {unique_questions}")
            
            # 텍스트 컬럼 확인
            text_cols = [col for col in df.columns if col.startswith('text_')]
            print(f"   • Text columns: {text_cols}")
            
            # 임베딩 컬럼 확인
            embed_cols = [col for col in df.columns if col.startswith('embed_')]
            print(f"   • Embedding columns: {len(embed_cols)} columns")
            
            # 샘플 데이터
            if len(df) > 0:
                print(f"   • Sample text: {df.iloc[0].get('text_1', 'N/A')}")
                
        except Exception as e:
            print(f"   ❌ Error reading {file_name}: {e}")

if __name__ == "__main__":
    # 실제 데이터 디렉토리
    real_data_dir = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/temp_data/stage2_results copy"
    
    # 1. 데이터 구조 분석
    analyze_data_structure(real_data_dir)
    
    # 2. 실제 데이터로 테스트
    success = test_with_real_data()
    
    print(f"\n" + "=" * 60)
    if success:
        print("🎉 Real Data Test PASSED!")
        print("✅ Updated quality criteria applied")
        print("✅ Real stage2 data processed successfully")
    else:
        print("❌ Real Data Test FAILED!")
        print("Please check the logs above for details")
    print("=" * 60)