"""
MCL Singleton Analysis and Cluster Count Optimization
MCL의 singleton 의미 분석 및 클러스터 수 최적화
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from typing import Dict, Any, List, Tuple
import json

def analyze_clustering_results():
    """현재 클러스터링 결과 분석"""
    
    print("📊 MCL vs Current Results Analysis")
    print("=" * 60)
    
    # 현재 결과 분석
    current_results = {
        '문4': {
            'columns': 2,
            'clusters': 11,  # 5 + 6
            'embeddings': 1544,  # 776 + 768
            'avg_silhouette': 0.133
        },
        '문5': {
            'columns': 10,
            'clusters': 52,  # 6+6+5+7+5+5+6+4+4+4
            'embeddings': 7516,  # 각 컬럼별 합계
            'avg_silhouette': 0.117
        }
    }
    
    total_clusters = sum(q['clusters'] for q in current_results.values())
    total_embeddings = sum(q['embeddings'] for q in current_results.values())
    
    print(f"\n🔍 Current Results Summary:")
    print(f"   Total clusters: {total_clusters}")
    print(f"   Total embeddings: {total_embeddings}")
    print(f"   Cluster ratio: {total_clusters/total_embeddings:.3f} (clusters per embedding)")
    
    for q_id, result in current_results.items():
        clusters_per_column = result['clusters'] / result['columns']
        embeddings_per_cluster = result['embeddings'] / result['clusters']
        print(f"   {q_id}: {result['clusters']} clusters from {result['columns']} columns")
        print(f"     → {clusters_per_column:.1f} clusters/column, {embeddings_per_cluster:.1f} embeddings/cluster")
    
    print(f"\n❗ Issues with Current Approach:")
    print(f"   1. Too many clusters: {total_clusters} clusters for {total_embeddings} embeddings")
    print(f"   2. Small cluster sizes: average {total_embeddings/total_clusters:.1f} embeddings per cluster")
    print(f"   3. No singleton detection: MCL's key advantage lost")
    
    return current_results

def analyze_mcl_singleton_value():
    """MCL의 singleton 의미 분석"""
    
    print(f"\n🎯 MCL Singleton Value Analysis:")
    print(f"=" * 40)
    
    mcl_advantages = {
        "Singleton Detection": {
            "의미": "유사한 항목이 없는 독특한 응답 식별",
            "가치": "outlier/unique response 자동 분리",
            "예시": "설문에서 특이한 응답이나 오타가 포함된 응답"
        },
        "Natural Clustering": {
            "의미": "데이터 자체의 구조를 반영한 클러스터 수",
            "가치": "사전에 클러스터 수를 정하지 않아도 됨",
            "예시": "실제로 의미있는 그룹의 수만큼 클러스터 생성"
        },
        "Noise Handling": {
            "의미": "노이즈나 이상치를 singleton으로 분리",
            "가치": "주요 클러스터의 품질 향상",
            "예시": "잘못 입력된 데이터나 무의미한 응답 분리"
        }
    }
    
    for advantage, details in mcl_advantages.items():
        print(f"\n📌 {advantage}:")
        for key, value in details.items():
            print(f"   {key}: {value}")
    
    print(f"\n⚠️  Current Approach Problems:")
    print(f"   • K-means/Hierarchical: 모든 데이터를 강제로 클러스터에 할당")
    print(f"   • DBSCAN: noise는 식별하지만 singleton의 의미적 가치 무시")
    print(f"   • 너무 많은 클러스터: 해석하기 어려운 세분화")

def propose_optimized_approach():
    """최적화된 접근법 제안"""
    
    print(f"\n💡 Optimized Approach Proposal:")
    print(f"=" * 40)
    
    print(f"\n1. 🎯 Hybrid MCL-Alternative Approach:")
    print(f"   • MCL로 먼저 시도 (singleton 감지)")
    print(f"   • MCL 실패시 제한된 K-means (K=3-5)")
    print(f"   • Singleton threshold 적용")
    
    print(f"\n2. 📊 Cluster Count Optimization:")
    print(f"   • 최대 클러스터 수 제한: 5-7개")
    print(f"   • 최소 클러스터 크기: 전체의 5% 이상")
    print(f"   • Singleton 허용: 전체의 1-3%")
    
    print(f"\n3. 🔍 Quality over Quantity:")
    print(f"   • 의미있는 클러스터 우선")
    print(f"   • 너무 작은 클러스터는 singleton으로 처리")
    print(f"   • 클러스터 간 의미적 차이 확보")
    
    return {
        'max_clusters_per_column': 5,
        'min_cluster_size_ratio': 0.05,
        'singleton_threshold_ratio': 0.03,
        'approach_priority': ['mcl_optimized', 'kmeans_limited', 'hierarchical_limited']
    }

def design_singleton_aware_clustering():
    """Singleton을 고려한 클러스터링 설계"""
    
    print(f"\n🔧 Singleton-Aware Clustering Design:")
    print(f"=" * 45)
    
    design = {
        'step1_mcl_attempt': {
            'description': 'MCL 시도 (최적화된 파라미터)',
            'params': {
                'inflation': [1.1, 1.2, 1.3],  # 더 낮은 값
                'k': [10, 15, 20],  # 적당한 값
                'max_iters': [50, 100]
            },
            'success_criteria': {
                'max_clusters': 7,
                'min_cluster_size': '3% of data',
                'singleton_ratio': 'under 10%'
            }
        },
        'step2_fallback': {
            'description': 'MCL 실패시 대안',
            'algorithms': ['kmeans_with_singleton', 'hierarchical_with_pruning'],
            'post_processing': 'small_cluster_to_singleton'
        },
        'step3_singleton_detection': {
            'description': '후처리 singleton 감지',
            'methods': ['distance_threshold', 'cluster_size_threshold', 'silhouette_threshold']
        }
    }
    
    for step, details in design.items():
        print(f"\n📋 {step}:")
        print(f"   {details['description']}")
        if 'params' in details:
            for param, values in details['params'].items():
                print(f"   • {param}: {values}")
        if 'success_criteria' in details:
            for criteria, value in details['success_criteria'].items():
                print(f"   ✓ {criteria}: {value}")
    
    return design

def calculate_optimal_cluster_counts(current_results):
    """최적 클러스터 수 계산"""
    
    print(f"\n📈 Optimal Cluster Count Calculation:")
    print(f"=" * 40)
    
    optimized_targets = {}
    
    for q_id, result in current_results.items():
        embeddings = result['embeddings']
        columns = result['columns']
        
        # 최적 클러스터 수 계산
        # 경험적 공식: sqrt(n/2) but limited to 3-7
        optimal_per_column = min(7, max(3, int(np.sqrt(embeddings/columns/2))))
        total_optimal = optimal_per_column * columns
        
        # Singleton 허용 (전체의 1-3%)
        singleton_allowance = int(embeddings * 0.02)  # 2%
        
        optimized_targets[q_id] = {
            'current_clusters': result['clusters'],
            'optimal_clusters': total_optimal,
            'clusters_per_column': optimal_per_column,
            'singleton_allowance': singleton_allowance,
            'reduction_ratio': result['clusters'] / total_optimal
        }
        
        print(f"\n{q_id}:")
        print(f"   Current: {result['clusters']} clusters")
        print(f"   Optimal: {total_optimal} clusters ({optimal_per_column}/column)")
        print(f"   Singleton allowance: {singleton_allowance}")
        print(f"   Reduction: {optimized_targets[q_id]['reduction_ratio']:.1f}x")
    
    return optimized_targets

if __name__ == "__main__":
    print("🔍 MCL Singleton Analysis and Optimization")
    print("=" * 60)
    
    # 1. 현재 결과 분석
    current_results = analyze_clustering_results()
    
    # 2. MCL singleton 가치 분석
    analyze_mcl_singleton_value()
    
    # 3. 최적화 접근법 제안
    optimization_config = propose_optimized_approach()
    
    # 4. Singleton 고려 설계
    design = design_singleton_aware_clustering()
    
    # 5. 최적 클러스터 수 계산
    optimal_targets = calculate_optimal_cluster_counts(current_results)
    
    print(f"\n🎯 Recommendations:")
    print(f"=" * 20)
    print(f"1. MCL 재시도: singleton 감지 능력 활용")
    print(f"2. 클러스터 수 제한: 컬럼당 3-5개")
    print(f"3. Singleton 허용: 전체의 1-3%")
    print(f"4. 품질 우선: 해석 가능한 클러스터 생성")
    
    print(f"\n✅ Analysis completed!")
    
    # Configuration for implementation
    final_config = {
        'mcl_retry': True,
        'mcl_params': {
            'inflation_range': [1.1, 1.2, 1.3],
            'k_range': [10, 15, 20],
            'max_iters': [50, 100]
        },
        'fallback_algorithms': ['kmeans_limited', 'hierarchical_limited'],
        'cluster_constraints': {
            'max_per_column': 5,
            'min_cluster_size_ratio': 0.05,
            'singleton_threshold_ratio': 0.03
        },
        'quality_thresholds': {
            'min_silhouette': 0.1,
            'max_total_clusters': 30  # 전체 파이프라인
        }
    }
    
    print(f"\n📋 Implementation Config:")
    print(json.dumps(final_config, indent=2, ensure_ascii=False))