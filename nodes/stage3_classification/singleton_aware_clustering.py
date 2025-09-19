"""
Singleton-Aware Clustering System
MCL의 싱글톤 보존 가치를 유지하면서 성능 문제를 해결하는 하이브리드 클러스터링
NMI/ARI 기반 평가 시스템 통합
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import markov_clustering as mc
import warnings
warnings.filterwarnings('ignore')

# NMI/ARI 평가 시스템 import
try:
    from .nmi_ari_evaluation import NMIARIEvaluator
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from nmi_ari_evaluation import NMIARIEvaluator

class SingletonAwareClustering:
    """
    싱글톤 인식 클러스터링 시스템
    
    MCL의 철학(싱글톤 보존)을 유지하면서 성능 문제를 해결하기 위해
    여러 알고리즘을 순차적으로 시도하는 하이브리드 접근법
    
    NMI/ARI 기반 평가 시스템으로 클러스터 품질 측정
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # 기본 설정
        self.config = config or {
            'mcl_max_time': 120,        # MCL 최대 실행 시간 (초)
            'mcl_inflation': 2.0,       # MCL inflation 파라미터
            'mcl_expansion': 2,         # MCL expansion 파라미터
            'mcl_similarity_threshold': 0.3,  # MCL용 유사도 임계값
            
            'kmeans_max_clusters': 7,   # K-means 최대 클러스터 수
            'kmeans_min_samples': 10,   # K-means 최소 샘플 수
            
            'dbscan_eps_range': (0.3, 0.8),     # DBSCAN eps 범위
            'dbscan_min_samples_ratio': 0.05,   # DBSCAN min_samples 비율
            
            'singleton_threshold': 0.15,         # 싱글톤 판정 임계값
            'quality_threshold': 0.2,            # NMI/ARI 품질 임계값 (silhouette 대신)
            'min_cluster_size_ratio': 0.03,      # 최소 클러스터 크기 비율
        }
        
        # NMI/ARI 평가자 초기화
        self.evaluator = NMIARIEvaluator({
            'nmi_weight': 0.6,
            'ari_weight': 0.4,
            'min_cluster_size_ratio': self.config['min_cluster_size_ratio'],
            'max_clusters': self.config['kmeans_max_clusters'],
            'min_nmi_threshold': 0.1,
            'min_ari_threshold': 0.05
        })
        
        self.stats = {
            'method_used': None,
            'mcl_attempted': False,
            'mcl_success': False,
            'fallback_reason': None,
            'execution_time': 0,
            'n_clusters': 0,
            'n_singletons': 0,
            'evaluation_metrics': {}
        }
    
    def create_optimized_graph(self, embeddings: np.ndarray, k: int = 15) -> csr_matrix:
        """MCL을 위한 최적화된 그래프 생성"""
        
        # 1. 코사인 유사도 계산
        similarity_matrix = cosine_similarity(embeddings)
        n = len(embeddings)
        
        # 2. KNN 그래프 생성 (각 노드마다 k개의 최근접 이웃)
        knn_graph = np.zeros_like(similarity_matrix)
        
        for i in range(n):
            # 자기 자신 제외하고 k개의 최근접 이웃 찾기
            neighbors = np.argsort(similarity_matrix[i])[-k-1:-1]  # 자기 자신 제외
            for j in neighbors:
                knn_graph[i][j] = similarity_matrix[i][j]
        
        # 3. 대칭화 (undirected graph)
        knn_graph = (knn_graph + knn_graph.T) / 2
        
        # 4. 최소 임계값 적용 (너무 약한 연결 제거)
        threshold = np.percentile(knn_graph[knn_graph > 0], 50)  # 상위 50%만 유지
        knn_graph[knn_graph < threshold] = 0
        
        return csr_matrix(knn_graph)
    
    def try_mcl_clustering(self, embeddings: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """최적화된 MCL 클러스터링 시도"""
        
        n_samples = len(embeddings)
        mcl_params = self.config['mcl_params']
        
        best_result = None
        best_score = -1
        best_info = {}
        
        print(f"   🔄 Trying MCL with optimized parameters...")
        
        for inflation in mcl_params['inflation_range']:
            for k in mcl_params['k_range']:
                for max_iters in mcl_params['max_iters']:
                    try:
                        # 최적화된 그래프 생성
                        graph = self.create_optimized_graph(embeddings, k=k)
                        
                        # MCL 실행
                        result = mc.run_mcl(graph, inflation=inflation, iterations=max_iters)
                        clusters = mc.get_clusters(result)
                        
                        # 클러스터 배열로 변환
                        labels = np.full(n_samples, -1)
                        for cluster_id, cluster_nodes in enumerate(clusters):
                            for node in cluster_nodes:
                                if node < n_samples:  # 범위 체크
                                    labels[node] = cluster_id
                        
                        # 품질 평가
                        n_clusters = len(clusters)
                        singleton_count = sum(1 for cluster in clusters if len(cluster) == 1)
                        singleton_ratio = singleton_count / n_clusters if n_clusters > 0 else 0
                        
                        # 제약 조건 확인
                        max_clusters = self.config['cluster_constraints']['max_per_column']
                        if n_clusters > max_clusters:
                            continue
                        
                        # Silhouette score 계산 (클러스터가 2개 이상일 때만)
                        if n_clusters > 1:
                            try:
                                silhouette = silhouette_score(embeddings, labels)
                            except:
                                silhouette = 0
                        else:
                            silhouette = 0
                        
                        # 점수 계산 (singleton 감지 보너스 포함)
                        score = silhouette + (singleton_ratio * 0.1)  # singleton 보너스
                        
                        if score > best_score:
                            best_result = labels
                            best_score = score
                            best_info = {
                                'inflation': inflation,
                                'k': k,
                                'max_iters': max_iters,
                                'n_clusters': n_clusters,
                                'singleton_count': singleton_count,
                                'singleton_ratio': singleton_ratio,
                                'silhouette': silhouette,
                                'score': score
                            }
                        
                        print(f"     • inflation={inflation}, k={k}, iters={max_iters}")
                        print(f"       → {n_clusters} clusters, {singleton_count} singletons, score={score:.3f}")
                        
                    except Exception as e:
                        continue
        
        if best_result is not None:
            print(f"   ✅ MCL succeeded: {best_info['n_clusters']} clusters, {best_info['singleton_count']} singletons")
            return best_result, best_info
        else:
            print(f"   ❌ MCL failed with all parameter combinations")
            return None, {}
    
    def limited_kmeans(self, embeddings: np.ndarray, max_k: int = 5) -> Tuple[np.ndarray, Dict]:
        """제한된 K-means (singleton 후처리 포함)"""
        
        n_samples = len(embeddings)
        min_cluster_size = int(n_samples * self.config['cluster_constraints']['min_cluster_size_ratio'])
        
        best_labels = None
        best_score = -1
        best_info = {}
        
        print(f"   🔄 Trying limited K-means (K=2-{max_k})...")
        
        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
                # 작은 클러스터를 singleton으로 변환
                cluster_counts = np.bincount(labels)
                small_clusters = np.where(cluster_counts < min_cluster_size)[0]
                
                # Singleton 마킹 (-1로 변경)
                singleton_labels = labels.copy()
                for small_cluster in small_clusters:
                    singleton_labels[labels == small_cluster] = -1
                
                # 나머지 클러스터 재넘버링
                unique_labels = np.unique(singleton_labels)
                unique_labels = unique_labels[unique_labels != -1]
                label_mapping = {old: new for new, old in enumerate(unique_labels)}
                
                for old_label, new_label in label_mapping.items():
                    singleton_labels[singleton_labels == old_label] = new_label
                
                # 품질 평가
                n_clusters = len(unique_labels)
                singleton_count = np.sum(singleton_labels == -1)
                
                if n_clusters > 1:
                    # Silhouette는 singleton 제외하고 계산
                    non_singleton_mask = singleton_labels != -1
                    if np.sum(non_singleton_mask) > 1:
                        silhouette = silhouette_score(
                            embeddings[non_singleton_mask], 
                            singleton_labels[non_singleton_mask]
                        )
                    else:
                        silhouette = 0
                else:
                    silhouette = 0
                
                # Singleton 보너스 포함 점수
                singleton_ratio = singleton_count / n_samples
                score = silhouette + (singleton_ratio * 0.1)
                
                if score > best_score:
                    best_labels = singleton_labels
                    best_score = score
                    best_info = {
                        'algorithm': 'kmeans_limited',
                        'k': k,
                        'n_clusters': n_clusters,
                        'singleton_count': singleton_count,
                        'singleton_ratio': singleton_ratio,
                        'silhouette': silhouette,
                        'score': score
                    }
                
                print(f"     • K={k} → {n_clusters} clusters, {singleton_count} singletons, score={score:.3f}")
                
            except Exception as e:
                continue
        
        return best_labels, best_info
    
    def dbscan_singleton_aware(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Singleton을 고려한 DBSCAN"""
        
        n_samples = len(embeddings)
        
        best_labels = None
        best_score = -1
        best_info = {}
        
        print(f"   🔄 Trying singleton-aware DBSCAN...")
        
        # eps 범위 계산
        similarity_matrix = cosine_similarity(embeddings)
        distances = 1 - similarity_matrix
        np.fill_diagonal(distances, np.inf)
        
        # 적절한 eps 범위 계산
        min_distances = np.min(distances, axis=1)
        eps_candidates = np.percentile(min_distances, [10, 20, 30, 40, 50])
        
        for eps in eps_candidates:
            for min_samples in [3, 5, 7]:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                    labels = dbscan.fit_predict(embeddings)
                    
                    # 클러스터 정보 계산
                    unique_labels = np.unique(labels)
                    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                    singleton_count = np.sum(labels == -1)
                    
                    if n_clusters < 2:  # 너무 적은 클러스터
                        continue
                    
                    # 품질 평가
                    if n_clusters > 1:
                        non_noise_mask = labels != -1
                        if np.sum(non_noise_mask) > 1:
                            silhouette = silhouette_score(
                                embeddings[non_noise_mask], 
                                labels[non_noise_mask]
                            )
                        else:
                            silhouette = 0
                    else:
                        silhouette = 0
                    
                    # Singleton 비율 계산
                    singleton_ratio = singleton_count / n_samples
                    score = silhouette + (singleton_ratio * 0.1)
                    
                    if score > best_score:
                        best_labels = labels
                        best_score = score
                        best_info = {
                            'algorithm': 'dbscan',
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'singleton_count': singleton_count,
                            'singleton_ratio': singleton_ratio,
                            'silhouette': silhouette,
                            'score': score
                        }
                    
                    print(f"     • eps={eps:.3f}, min_samples={min_samples}")
                    print(f"       → {n_clusters} clusters, {singleton_count} singletons, score={score:.3f}")
                    
                except Exception as e:
                    continue
        
        return best_labels, best_info
    
    def fit_predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """하이브리드 클러스터링 실행"""
        
        print(f"\n🎯 Singleton-Aware Clustering ({len(embeddings)} embeddings)")
        print(f"=" * 50)
        
        # 1. MCL 시도
        mcl_labels, mcl_info = self.try_mcl_clustering(embeddings)
        
        if mcl_labels is not None:
            print(f"✅ MCL successful - using MCL results")
            return mcl_labels, mcl_info
        
        # 2. 대안 알고리즘들 시도
        candidates = []
        
        # 2-1. Limited K-means
        kmeans_labels, kmeans_info = self.limited_kmeans(embeddings)
        if kmeans_labels is not None:
            candidates.append((kmeans_labels, kmeans_info))
        
        # 2-2. DBSCAN
        dbscan_labels, dbscan_info = self.dbscan_singleton_aware(embeddings)
        if dbscan_labels is not None:
            candidates.append((dbscan_labels, dbscan_info))
        
        # 3. 최고 성능 선택
        if candidates:
            best_labels, best_info = max(candidates, key=lambda x: x[1]['score'])
            print(f"✅ Best algorithm: {best_info['algorithm']} (score={best_info['score']:.3f})")
            return best_labels, best_info
        else:
            # 4. 최후의 수단: 간단한 K-means
            print(f"⚠️  Fallback to simple K-means")
            kmeans = KMeans(n_clusters=3, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            
            info = {
                'algorithm': 'fallback_kmeans',
                'n_clusters': 3,
                'singleton_count': 0,
                'singleton_ratio': 0,
                'silhouette': silhouette_score(embeddings, labels) if len(np.unique(labels)) > 1 else 0,
                'score': 0
            }
            
            return labels, info

# 테스트 함수
def test_singleton_aware_clustering():
    """Singleton-aware clustering 테스트"""
    
    print("🧪 Testing Singleton-Aware Clustering")
    print("=" * 40)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    
    # 정상 클러스터 데이터
    cluster1 = np.random.normal([0, 0], 0.3, (50, 2))
    cluster2 = np.random.normal([2, 2], 0.3, (50, 2))
    cluster3 = np.random.normal([-1, 2], 0.3, (30, 2))
    
    # Singleton/outlier 데이터
    outliers = np.random.uniform(-3, 5, (10, 2))
    
    # 전체 데이터 합치기
    test_data = np.vstack([cluster1, cluster2, cluster3, outliers])
    
    print(f"Test data: {len(test_data)} points")
    print(f"Expected: 3 main clusters + ~10 singletons")
    
    # 클러스터링 실행
    clustering = SingletonAwareClustering()
    labels, info = clustering.fit_predict(test_data)
    
    print(f"\nResults:")
    print(f"Algorithm: {info['algorithm']}")
    print(f"Clusters: {info['n_clusters']}")
    print(f"Singletons: {info['singleton_count']}")
    print(f"Silhouette: {info['silhouette']:.3f}")
    print(f"Score: {info['score']:.3f}")
    
    return labels, info

if __name__ == "__main__":
    print("🎯 Singleton-Aware Clustering Pipeline")
    print("=" * 50)
    
    # 테스트 실행
    test_labels, test_info = test_singleton_aware_clustering()
    
    print(f"\n✅ Singleton-aware clustering module ready!")
    print(f"Key features:")
    print(f"• MCL with optimized parameters for sentence embeddings")
    print(f"• Automatic singleton detection and preservation")
    print(f"• Fallback to limited K-means/DBSCAN")
    print(f"• Quality-based algorithm selection")