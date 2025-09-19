#!/usr/bin/env python3
"""실제 Stage2 데이터로 향상된 Stage3 파이프라인 테스트"""

import sys
import os
sys.path.append('/home/cyyoon/test_area/ai_text_classification/2.langgraph')

from nodes.stage3_classification.singleton_aware_stage3_node import SingletonAwareStage3Node

def test_real_stage2_data():
    """실제 Stage2 결과 데이터로 테스트"""
    
    print("🚀 실제 Stage2 데이터로 향상된 Stage3 테스트 시작")
    print("=" * 60)
    
    # 실제 Stage2 데이터 경로
    data_directory = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/temp_data/stage2_results copy/"
    
    try:
        # Stage3 노드 초기화
        stage3_node = SingletonAwareStage3Node()
        
        # 실제 데이터로 처리
        results = stage3_node.process_column_wise_data(data_directory)
        
        print("\n" + "=" * 60)
        print("📊 최종 결과 요약")
        print("=" * 60)
        
        if results:
            for group_key, result in results.items():
                print(f"\n🔍 {group_key}:")
                print(f"   • 샘플 수: {result.get('n_samples', 'N/A')}")
                print(f"   • 알고리즘: {result.get('algorithm', 'N/A')}")
                print(f"   • 클러스터 수: {result.get('n_clusters', 'N/A')}")
                print(f"   • NMI 점수: {result.get('nmi_score', 'N/A'):.4f}" if result.get('nmi_score') else "   • NMI 점수: N/A")
                print(f"   • ARI 점수: {result.get('ari_score', 'N/A'):.4f}" if result.get('ari_score') else "   • ARI 점수: N/A")
                print(f"   • 결합 점수: {result.get('combined_score', 'N/A'):.4f}" if result.get('combined_score') else "   • 결합 점수: N/A")
                print(f"   • 품질 등급: {result.get('quality_grade', 'N/A')}")
                
                if 'cluster_labels' in result:
                    unique_labels = set(result['cluster_labels'])
                    print(f"   • 고유 라벨 수: {len(unique_labels)}")
                    print(f"   • 라벨 분포: {dict(zip(*np.unique(result['cluster_labels'], return_counts=True)))}")
        else:
            print("❌ 처리된 결과가 없습니다.")
            
        print("\n✅ 실제 데이터 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import numpy as np
    test_real_stage2_data()