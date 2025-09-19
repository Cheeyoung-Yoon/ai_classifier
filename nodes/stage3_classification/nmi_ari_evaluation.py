"""
NMI & ARI Based Clustering Evaluation System
Silhouette score 대신 NMI(Normalized Mutual Information)와 ARI(Adjusted Rand Index) 기반 평가
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class NMIARIEvaluator:
    """NMI와 ARI 기반 클러스터링 평가자"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'nmi_weight': 0.6,  # NMI 가중치
            'ari_weight': 0.4,  # ARI 가중치
            'min_cluster_size_ratio': 0.03,  # 최소 클러스터 크기 (전체의 3%)
            'max_clusters': 7,   # 최대 클러스터 수
            'min_nmi_threshold': 0.1,  # 최소 NMI 임계값
            'min_ari_threshold': 0.05  # 최소 ARI 임계값
        }
        
    def create_synthetic_ground_truth(self, embeddings: np.ndarray, method: str = 'kmeans') -> np.ndarray:
        """
        실제 ground truth가 없을 때 synthetic ground truth 생성
        실제 데이터의 자연스러운 클러스터 구조를 추정
        """
        n_samples = len(embeddings)
        
        if method == 'kmeans':
            # K-means로 자연스러운 클러스터 수 추정 (3-5개)
            best_k = 3
            best_inertia = float('inf')
            
            for k in range(3, min(6, n_samples//10)):  # 최소 10개씩은 있어야 함
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
                if kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_k = k
            
            # 최적 K로 ground truth 생성
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            ground_truth = kmeans.fit_predict(embeddings)
            
        elif method == 'hierarchical':
            # Hierarchical clustering으로 자연스러운 구조 찾기
            n_clusters = min(5, max(3, n_samples//20))
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            ground_truth = hierarchical.fit_predict(embeddings)
            
        elif method == 'consensus':
            # 여러 방법의 consensus
            k1 = KMeans(n_clusters=3, random_state=42).fit_predict(embeddings)
            k2 = KMeans(n_clusters=4, random_state=43).fit_predict(embeddings)
            k3 = KMeans(n_clusters=5, random_state=44).fit_predict(embeddings)
            
            # 가장 일관성 있는 결과 선택 (간단히 첫 번째 사용)
            ground_truth = k1
            
        return ground_truth
    
    def calculate_nmi_ari(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
        """NMI와 ARI 계산"""
        
        # 유효한 라벨만 사용 (-1 제외)
        valid_mask = (pred_labels != -1) & (true_labels != -1)
        
        if np.sum(valid_mask) < 2:
            return {
                'nmi': 0.0,
                'ari': 0.0,
                'adj_ari': 0.0,
                'combined_score': 0.0,
                'valid_samples': 0
            }
        
        valid_true = true_labels[valid_mask]
        valid_pred = pred_labels[valid_mask]
        
        # NMI 계산
        nmi = normalized_mutual_info_score(valid_true, valid_pred)
        
        # ARI 계산
        ari = adjusted_rand_score(valid_true, valid_pred)
        
        # Adjusted ARI (전체 항목 수 고려)
        total_samples = len(true_labels)
        valid_samples = len(valid_true)
        coverage_penalty = valid_samples / total_samples  # 커버리지 패널티
        
        adj_ari = ari * coverage_penalty
        
        # Combined score (NMI + ARI 가중 평균)
        combined_score = (
            self.config['nmi_weight'] * nmi + 
            self.config['ari_weight'] * adj_ari
        )
        
        return {
            'nmi': nmi,
            'ari': ari,
            'adj_ari': adj_ari,
            'combined_score': combined_score,
            'valid_samples': valid_samples,
            'coverage_ratio': coverage_penalty
        }
    
    def evaluate_clustering_quality(self, embeddings: np.ndarray, pred_labels: np.ndarray, 
                                   ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """클러스터링 품질 종합 평가"""
        
        # Ground truth가 없으면 생성
        if ground_truth is None:
            ground_truth = self.create_synthetic_ground_truth(embeddings, method='consensus')
        
        # 기본 메트릭 계산
        metrics = self.calculate_nmi_ari(ground_truth, pred_labels)
        
        # 클러스터 통계
        unique_labels = np.unique(pred_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(pred_labels == -1)
        n_samples = len(pred_labels)
        
        # 클러스터 크기 분포
        cluster_sizes = []
        for label in unique_labels:
            if label != -1:
                cluster_sizes.append(np.sum(pred_labels == label))
        
        # 품질 판정
        quality_assessment = self._assess_quality(metrics, n_clusters, n_samples, cluster_sizes)
        
        return {
            **metrics,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_samples': n_samples,
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'min_cluster_size': np.min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': np.max(cluster_sizes) if cluster_sizes else 0,
            'quality_assessment': quality_assessment
        }
    
    def _assess_quality(self, metrics: Dict, n_clusters: int, n_samples: int, 
                       cluster_sizes: List[int]) -> str:
        """품질 종합 판정"""
        
        nmi = metrics['nmi']
        adj_ari = metrics['adj_ari']
        combined = metrics['combined_score']
        
        # 임계값 체크
        nmi_good = nmi >= self.config['min_nmi_threshold']
        ari_good = adj_ari >= self.config['min_ari_threshold']
        
        # 클러스터 수 체크
        clusters_reasonable = 2 <= n_clusters <= self.config['max_clusters']
        
        # 클러스터 크기 체크
        min_size = n_samples * self.config['min_cluster_size_ratio']
        sizes_good = all(size >= min_size for size in cluster_sizes) if cluster_sizes else False
        
        if nmi_good and ari_good and clusters_reasonable and sizes_good:
            if combined >= 0.5:
                return "EXCELLENT"
            elif combined >= 0.3:
                return "GOOD"
            else:
                return "ACCEPTABLE"
        elif nmi_good and ari_good:
            return "FAIR"
        else:
            return "POOR"

def test_nmi_ari_evaluator():
    """NMI/ARI 평가자 테스트"""
    
    print("🧪 Testing NMI/ARI Evaluator")
    print("=" * 40)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    
    # 명확한 3개 클러스터
    cluster1 = np.random.normal([0, 0], 0.5, (100, 2))
    cluster2 = np.random.normal([3, 3], 0.5, (100, 2))
    cluster3 = np.random.normal([-2, 3], 0.5, (80, 2))
    
    # 노이즈 포인트들
    noise = np.random.uniform(-5, 5, (20, 2))
    
    test_data = np.vstack([cluster1, cluster2, cluster3, noise])
    
    # Ground truth (실제 클러스터 라벨)
    true_labels = np.array([0]*100 + [1]*100 + [2]*80 + [3]*20)  # 마지막은 노이즈
    
    print(f"Test data: {len(test_data)} points, 3 main clusters + noise")
    
    # 평가자 초기화
    evaluator = NMIARIEvaluator()
    
    # 테스트 1: 완벽한 클러스터링
    print(f"\n📊 Test 1: Perfect Clustering")
    perfect_labels = true_labels.copy()
    perfect_results = evaluator.evaluate_clustering_quality(test_data, perfect_labels, true_labels)
    
    print(f"   NMI: {perfect_results['nmi']:.3f}")
    print(f"   ARI: {perfect_results['ari']:.3f}")
    print(f"   Adj ARI: {perfect_results['adj_ari']:.3f}")
    print(f"   Combined Score: {perfect_results['combined_score']:.3f}")
    print(f"   Quality: {perfect_results['quality_assessment']}")
    
    # 테스트 2: K-means 클러스터링
    print(f"\n📊 Test 2: K-means Clustering")
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(test_data)
    kmeans_results = evaluator.evaluate_clustering_quality(test_data, kmeans_labels, true_labels)
    
    print(f"   NMI: {kmeans_results['nmi']:.3f}")
    print(f"   ARI: {kmeans_results['ari']:.3f}")
    print(f"   Adj ARI: {kmeans_results['adj_ari']:.3f}")
    print(f"   Combined Score: {kmeans_results['combined_score']:.3f}")
    print(f"   Quality: {kmeans_results['quality_assessment']}")
    print(f"   Clusters: {kmeans_results['n_clusters']}")
    
    # 테스트 3: DBSCAN 클러스터링
    print(f"\n📊 Test 3: DBSCAN Clustering")
    dbscan = DBSCAN(eps=0.8, min_samples=5)
    dbscan_labels = dbscan.fit_predict(test_data)
    dbscan_results = evaluator.evaluate_clustering_quality(test_data, dbscan_labels, true_labels)
    
    print(f"   NMI: {dbscan_results['nmi']:.3f}")
    print(f"   ARI: {dbscan_results['ari']:.3f}")
    print(f"   Adj ARI: {dbscan_results['adj_ari']:.3f}")
    print(f"   Combined Score: {dbscan_results['combined_score']:.3f}")
    print(f"   Quality: {dbscan_results['quality_assessment']}")
    print(f"   Clusters: {dbscan_results['n_clusters']}, Noise: {dbscan_results['n_noise']}")
    
    # 테스트 4: 나쁜 클러스터링 (모든 점을 같은 클러스터로)
    print(f"\n📊 Test 4: Poor Clustering (All Same)")
    poor_labels = np.zeros(len(test_data))
    poor_results = evaluator.evaluate_clustering_quality(test_data, poor_labels, true_labels)
    
    print(f"   NMI: {poor_results['nmi']:.3f}")
    print(f"   ARI: {poor_results['ari']:.3f}")
    print(f"   Adj ARI: {poor_results['adj_ari']:.3f}")
    print(f"   Combined Score: {poor_results['combined_score']:.3f}")
    print(f"   Quality: {poor_results['quality_assessment']}")
    
    return {
        'perfect': perfect_results,
        'kmeans': kmeans_results,
        'dbscan': dbscan_results,
        'poor': poor_results
    }

def demonstrate_adj_ari():
    """Adjusted ARI의 전체 항목 수 검증 기능 시연"""
    
    print(f"\n🔍 Adjusted ARI - 전체 항목 수 검증")
    print("=" * 50)
    
    evaluator = NMIARIEvaluator()
    
    # 시나리오 1: 모든 항목이 클러스터링됨
    true_labels_1 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    pred_labels_1 = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])  # 모든 항목 포함
    
    result_1 = evaluator.calculate_nmi_ari(true_labels_1, pred_labels_1)
    print(f"시나리오 1 - 전체 항목 클러스터링:")
    print(f"   ARI: {result_1['ari']:.3f}")
    print(f"   Adj ARI: {result_1['adj_ari']:.3f}")
    print(f"   Coverage: {result_1['coverage_ratio']:.3f}")
    print(f"   Valid samples: {result_1['valid_samples']}/9")
    
    # 시나리오 2: 일부 항목이 노이즈로 분류됨
    true_labels_2 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    pred_labels_2 = np.array([0, 0, 1, 1, 1, 2, 2, -1, -1])  # 2개 노이즈
    
    result_2 = evaluator.calculate_nmi_ari(true_labels_2, pred_labels_2)
    print(f"\n시나리오 2 - 일부 노이즈 포함:")
    print(f"   ARI: {result_2['ari']:.3f}")
    print(f"   Adj ARI: {result_2['adj_ari']:.3f}")
    print(f"   Coverage: {result_2['coverage_ratio']:.3f}")
    print(f"   Valid samples: {result_2['valid_samples']}/9")
    
    # 시나리오 3: 대부분 노이즈
    true_labels_3 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    pred_labels_3 = np.array([0, 0, -1, -1, -1, -1, -1, -1, -1])  # 대부분 노이즈
    
    result_3 = evaluator.calculate_nmi_ari(true_labels_3, pred_labels_3)
    print(f"\n시나리오 3 - 대부분 노이즈:")
    print(f"   ARI: {result_3['ari']:.3f}")
    print(f"   Adj ARI: {result_3['adj_ari']:.3f}")
    print(f"   Coverage: {result_3['coverage_ratio']:.3f}")
    print(f"   Valid samples: {result_3['valid_samples']}/9")
    
    print(f"\n💡 Adjusted ARI 효과:")
    print(f"   - 전체 항목 대비 유효 클러스터링 비율 반영")
    print(f"   - 노이즈가 많을수록 점수 하락")
    print(f"   - 실제 데이터 커버리지 품질 측정")

if __name__ == "__main__":
    print("🎯 NMI & ARI Based Clustering Evaluation")
    print("=" * 50)
    
    # 기본 테스트
    test_results = test_nmi_ari_evaluator()
    
    # Adjusted ARI 시연
    demonstrate_adj_ari()
    
    print(f"\n✅ NMI/ARI Evaluation System Ready!")
    print(f"Key Features:")
    print(f"• NMI (Normalized Mutual Information) - 정보 이론 기반 클러스터 품질")
    print(f"• ARI (Adjusted Rand Index) - 랜덤 보정된 클러스터 일치도")
    print(f"• Adj ARI - 전체 항목 수 고려한 보정된 ARI")
    print(f"• Combined Score - NMI + ARI 가중 결합 점수")
    print(f"• Quality Assessment - 종합 품질 등급 (EXCELLENT ~ POOR)")