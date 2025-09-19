"""
Singleton-Aware Clustering System with NMI/ARI Evaluation
MCL의 싱글톤 보존 가치를 유지하면서 성능 문제를 해결하는 하이브리드 클러스터링
NMI/ARI 기반 평가 시스템으로 클러스터 품질 측정
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import time
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import markov_clustering as mc
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# NMI/ARI 평가 시스템 import with improved fallback
try:
    from .nmi_ari_evaluation import NMIARIEvaluator
except ImportError:
    # Graceful fallback without sys.path manipulation
    try:
        from nmi_ari_evaluation import NMIARIEvaluator
    except ImportError:
        logger.warning("NMIARIEvaluator not available. Using dummy evaluator.")
        class NMIARIEvaluator:
            def evaluate(self, labels, ground_truth=None):
                return {"nmi": 0.0, "ari": 0.0, "combined_score": 0.0}

class SingletonAwareClusteringNMI:
    """
    NMI/ARI 기반 평가를 사용하는 싱글톤 인식 클러스터링 시스템
    
    1. MCL 시도 (timeout 포함)
    2. Limited K-means 시도 
    3. DBSCAN 시도
    4. NMI/ARI로 품질 평가하여 최적 결과 선택
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'mcl_max_time': 60,         # MCL 최대 실행 시간 (초)
            'mcl_inflation': 2.0,       # MCL inflation 파라미터
            'mcl_k': 15,                # MCL KNN graph k
            'mcl_similarity_threshold': 0.3,
            
            'kmeans_max_clusters': 7,   # K-means 최대 클러스터 수
            'kmeans_min_samples': 10,   # K-means 최소 샘플 수
            
            'dbscan_eps_range': (0.3, 0.8),
            'dbscan_min_samples_ratio': 0.05,
            
            'singleton_threshold': 0.15,
            'quality_threshold': 0.2,            # NMI/ARI Combined Score 임계값
            'min_cluster_size_ratio': 0.03,
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
    
    def try_knn_csls_mcl_pipeline(self, embeddings: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """원래 테스트된 KNN → CSLS → MCL 파이프라인"""
        
        try:
            logger.info("Trying KNN → CSLS → MCL pipeline...")
            start_time = time.time()
            
            n_samples = len(embeddings)
            if n_samples < 5:
                return None, {'error': 'Not enough samples for KNN-CSLS-MCL'}
            
            # Step 1: KNN Graph 구축
            k = min(self.config['mcl_k'], n_samples - 1)
            nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(embeddings)
            distances, indices = nbrs.kneighbors(embeddings)
            
            # 자기 자신 제외
            distances = distances[:, 1:]
            indices = indices[:, 1:]
            cosine_similarities = 1.0 - distances
            
            # Step 2: CSLS (Cross-domain Similarity Local Scaling)
            # CSLS = 2 * cos(x, y) - r_x - r_y (where r_x = mean similarity to neighbors)
            mean_similarities = np.mean(cosine_similarities, axis=1)
            
            # Step 3: Edge 구성 및 CSLS 점수 계산
            edges = []
            edge_weights = []
            
            for i in range(n_samples):
                for j_idx, j in enumerate(indices[i]):
                    if i == j:
                        continue
                    
                    cos_ij = cosine_similarities[i, j_idx]
                    # CSLS score
                    csls_score = 2.0 * cos_ij - mean_similarities[i] - mean_similarities[j]
                    
                    # 임계값 적용
                    if csls_score > 0.1:  # 최소 CSLS 임계값
                        edges.append((i, j))
                        edge_weights.append(csls_score)
            
            # Step 4: 인접 행렬 구성 (대칭화)
            adjacency = csr_matrix((n_samples, n_samples), dtype=np.float32)
            
            if len(edges) > 0:
                rows, cols = zip(*edges)
                # 대칭화 - 최대값 사용
                adj_data = {}
                for (i, j), weight in zip(edges, edge_weights):
                    key = (min(i, j), max(i, j))
                    if key not in adj_data:
                        adj_data[key] = weight
                    else:
                        adj_data[key] = max(adj_data[key], weight)
                
                # COO 형식으로 구성
                final_rows = []
                final_cols = []
                final_weights = []
                
                for (i, j), weight in adj_data.items():
                    final_rows.extend([i, j])
                    final_cols.extend([j, i])
                    final_weights.extend([weight, weight])
                
                adjacency = csr_matrix((final_weights, (final_rows, final_cols)), 
                                     shape=(n_samples, n_samples), dtype=np.float32)
            
            # Step 5: MCL 클러스터링
            if adjacency.nnz == 0:
                return None, {'error': 'No valid edges after CSLS filtering'}
            
            # MCL 실행
            result = mc.run_mcl(adjacency, inflation=self.config['mcl_inflation'], iterations=100)
            clusters = mc.get_clusters(result)
            
            # 클러스터 레이블 생성
            labels = np.full(n_samples, -1, dtype=int)
            for cluster_id, cluster_nodes in enumerate(clusters):
                for node in cluster_nodes:
                    if node < n_samples:
                        labels[node] = cluster_id
            
            n_clusters = len(clusters)
            n_singletons = np.sum(labels == -1)
            execution_time = time.time() - start_time
            
            print(f"     ✅ KNN-CSLS-MCL: {n_clusters} clusters, {n_singletons} singletons ({execution_time:.1f}s)")
            
            return labels, {
                'algorithm': 'knn_csls_mcl',
                'n_clusters': n_clusters,
                'n_singletons': n_singletons,
                'execution_time': execution_time,
                'n_edges': len(edges),
                'n_final_edges': adjacency.nnz // 2
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"     ❌ KNN-CSLS-MCL failed: {str(e)} ({execution_time:.1f}s)")
            return None, {'error': str(e), 'execution_time': execution_time}

    def try_mcl_with_timeout(self, embeddings: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """시간 제한이 있는 MCL 클러스터링"""
        
        self.stats['mcl_attempted'] = True
        start_time = time.time()
        
        try:
            # 그래프 생성
            similarity_matrix = cosine_similarity(embeddings)
            
            # KNN 그래프로 변환 (성능 최적화)
            k = min(self.config['mcl_k'], len(embeddings) - 1)
            knn_graph = np.zeros_like(similarity_matrix)
            
            for i in range(len(embeddings)):
                neighbors = np.argsort(similarity_matrix[i])[-k-1:-1]
                for j in neighbors:
                    knn_graph[i][j] = similarity_matrix[i][j]
            
            # 대칭화
            knn_graph = (knn_graph + knn_graph.T) / 2
            threshold = np.percentile(knn_graph[knn_graph > 0], 50)
            knn_graph[knn_graph < threshold] = 0
            
            # MCL 실행 (시간 체크)
            graph_csr = csr_matrix(knn_graph)
            result = mc.run_mcl(graph_csr, inflation=self.config['mcl_inflation'], iterations=100)
            clusters = mc.get_clusters(result)
            
            # 클러스터 레이블 생성
            labels = np.full(len(embeddings), -1)
            for cluster_id, cluster_nodes in enumerate(clusters):
                for node in cluster_nodes:
                    if node < len(embeddings):
                        labels[node] = cluster_id
            
            execution_time = time.time() - start_time
            
            # 시간 제한 체크
            if execution_time > self.config['mcl_max_time']:
                print(f"   ⏰ MCL timeout ({execution_time:.1f}s)")
                return None, {'timeout': True, 'execution_time': execution_time}
            
            n_clusters = len(clusters)
            n_singletons = sum(1 for cluster in clusters if len(cluster) == 1)
            
            self.stats['mcl_success'] = True
            
            info = {
                'algorithm': 'mcl',
                'n_clusters': n_clusters,
                'n_singletons': n_singletons,
                'execution_time': execution_time,
                'timeout': False
            }
            
            print(f"   ✅ MCL completed: {n_clusters} clusters, {n_singletons} singletons ({execution_time:.1f}s)")
            return labels, info
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   ❌ MCL failed: {str(e)} ({execution_time:.1f}s)")
            return None, {'error': str(e), 'execution_time': execution_time}
    
    def limited_kmeans_with_nmi_ari(self, embeddings: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """NMI/ARI 기반 Limited K-means"""
        
        n_samples = len(embeddings)
        min_cluster_size = max(2, int(n_samples * self.config['min_cluster_size_ratio']))
        max_k = min(self.config['kmeans_max_clusters'], n_samples // min_cluster_size)
        
        if max_k < 2:
            return None, {'error': 'Not enough samples for clustering'}
        
        best_labels = None
        best_score = -1
        best_info = {}
        
        print(f"   🔄 Trying limited K-means (K=2-{max_k})...")
        
        for k in range(2, max_k + 1):
            try:
                # K-means 실행
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
                # 작은 클러스터를 singleton으로 변환
                cluster_counts = np.bincount(labels)
                small_clusters = np.where(cluster_counts < min_cluster_size)[0]
                
                singleton_labels = labels.copy()
                for small_cluster in small_clusters:
                    singleton_labels[labels == small_cluster] = -1
                
                # 레이블 재정렬
                unique_labels = np.unique(singleton_labels)
                unique_labels = unique_labels[unique_labels != -1]
                
                if len(unique_labels) > 0:
                    label_mapping = {old: new for new, old in enumerate(unique_labels)}
                    for old_label, new_label in label_mapping.items():
                        singleton_labels[singleton_labels == old_label] = new_label
                
                # NMI/ARI 평가
                eval_result = self.evaluator.evaluate_clustering_quality(embeddings, singleton_labels)
                combined_score = eval_result['combined_score']
                
                if combined_score > best_score:
                    best_labels = singleton_labels
                    best_score = combined_score
                    best_info = {
                        'algorithm': 'kmeans_limited',
                        'k': k,
                        'n_clusters': eval_result['n_clusters'],
                        'n_singletons': eval_result['n_noise'],
                        'combined_score': combined_score,
                        'evaluation': eval_result
                    }
                
                print(f"     • K={k} → {eval_result['n_clusters']} clusters, {eval_result['n_noise']} singletons, score={combined_score:.3f}")
                
            except Exception as e:
                continue
        
        return best_labels, best_info
    
    def dbscan_with_nmi_ari(self, embeddings: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """NMI/ARI 기반 DBSCAN"""
        
        n_samples = len(embeddings)
        
        best_labels = None
        best_score = -1
        best_info = {}
        
        print(f"   🔄 Trying DBSCAN...")
        
        # eps 후보 계산
        similarity_matrix = cosine_similarity(embeddings)
        distances = 1 - similarity_matrix
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        eps_candidates = np.percentile(min_distances, [15, 25, 35, 45])
        
        for eps in eps_candidates:
            for min_samples in [3, 5, 7]:
                try:
                    # DBSCAN 실행
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                    labels = dbscan.fit_predict(embeddings)
                    
                    # 기본 검증
                    unique_labels = np.unique(labels)
                    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                    
                    if n_clusters < 2:
                        continue
                    
                    # NMI/ARI 평가
                    eval_result = self.evaluator.evaluate_clustering_quality(embeddings, labels)
                    combined_score = eval_result['combined_score']
                    
                    if combined_score > best_score:
                        best_labels = labels
                        best_score = combined_score
                        best_info = {
                            'algorithm': 'dbscan',
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': eval_result['n_clusters'],
                            'n_singletons': eval_result['n_noise'],
                            'combined_score': combined_score,
                            'evaluation': eval_result
                        }
                    
                    print(f"     • eps={eps:.3f}, min_samples={min_samples} → {eval_result['n_clusters']} clusters, score={combined_score:.3f}")
                    
                except Exception as e:
                    continue
        
        return best_labels, best_info
    
    def cluster_with_nmi_ari_evaluation(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        NMI/ARI 기반 품질 검사로 최적 클러스터링 수행
        
        1. MCL 시도 (time limit 있음)
        2. Limited K-means 시도
        3. DBSCAN 시도
        4. NMI/ARI Combined Score로 최적 결과 선택
        """
        
        start_time = time.time()
        results = []
        
        print(f"🎯 Singleton-aware clustering with NMI/ARI evaluation for {len(embeddings)} samples")
        
        # 1. KNN → CSLS → MCL 파이프라인 시도 (원래 테스트된 순서)
        print(f"📊 Step 1: Trying KNN → CSLS → MCL pipeline...")
        knn_csls_mcl_labels, knn_csls_mcl_info = self.try_knn_csls_mcl_pipeline(embeddings)
        if knn_csls_mcl_labels is not None:
            eval_result = self.evaluator.evaluate_clustering_quality(embeddings, knn_csls_mcl_labels)
            knn_csls_mcl_info.update({
                'method': 'knn_csls_mcl',
                'evaluation': eval_result,
                'combined_score': eval_result['combined_score']
            })
            results.append((knn_csls_mcl_labels, knn_csls_mcl_info))
            print(f"   ✅ KNN-CSLS-MCL: Combined Score = {eval_result['combined_score']:.3f} ({eval_result['quality_assessment']})")
        
        # 2. MCL 시도 (제한된 시간) - 백업용
        print(f"📊 Step 2: Trying MCL (backup)...")
        mcl_labels, mcl_info = self.try_mcl_with_timeout(embeddings)
        if mcl_labels is not None and not mcl_info.get('timeout', False):
            eval_result = self.evaluator.evaluate_clustering_quality(embeddings, mcl_labels)
            mcl_info.update({
                'method': 'mcl_backup',
                'evaluation': eval_result,
                'combined_score': eval_result['combined_score']
            })
            results.append((mcl_labels, mcl_info))
            print(f"   ✅ MCL: Combined Score = {eval_result['combined_score']:.3f} ({eval_result['quality_assessment']})")
        
        # 3. Limited K-means 시도
        print(f"📊 Step 3: Trying Limited K-means...")
        kmeans_labels, kmeans_info = self.limited_kmeans_with_nmi_ari(embeddings)
        if kmeans_labels is not None:
            kmeans_info['method'] = 'kmeans_limited'  # method 키 추가
            eval_result = kmeans_info['evaluation']
            results.append((kmeans_labels, kmeans_info))
            print(f"   ✅ K-means: Combined Score = {eval_result['combined_score']:.3f} ({eval_result['quality_assessment']})")
        
        # 4. DBSCAN 시도
        print(f"📊 Step 4: Trying DBSCAN...")
        dbscan_labels, dbscan_info = self.dbscan_with_nmi_ari(embeddings)
        if dbscan_labels is not None:
            dbscan_info['method'] = 'dbscan'  # method 키 추가
            eval_result = dbscan_info['evaluation']
            results.append((dbscan_labels, dbscan_info))
            print(f"   ✅ DBSCAN: Combined Score = {eval_result['combined_score']:.3f} ({eval_result['quality_assessment']})")
        
        # 5. 최적 결과 선택 (Combined Score 기준)
        if not results:
            print(f"   ❌ All clustering methods failed! Using fallback...")
            # Fallback: Simple K-means
            fallback_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            fallback_labels = fallback_kmeans.fit_predict(embeddings)
            fallback_eval = self.evaluator.evaluate_clustering_quality(embeddings, fallback_labels)
            
            self.stats.update({
                'method_used': 'fallback_kmeans',
                'execution_time': time.time() - start_time,
                'n_clusters': fallback_eval['n_clusters'],
                'n_singletons': fallback_eval['n_noise'],
                'evaluation_metrics': fallback_eval,
                'fallback_reason': 'all_methods_failed'
            })
            
            return fallback_labels, {
                'method': 'fallback_kmeans',
                'evaluation': fallback_eval,
                'combined_score': fallback_eval['combined_score']
            }
        
        # Combined Score로 정렬
        results.sort(key=lambda x: x[1]['combined_score'], reverse=True)
        best_labels, best_info = results[0]
        
        # 통계 업데이트
        self.stats.update({
            'method_used': best_info['method'],
            'execution_time': time.time() - start_time,
            'n_clusters': best_info['evaluation']['n_clusters'],
            'n_singletons': best_info['evaluation']['n_noise'],
            'evaluation_metrics': best_info['evaluation']
        })
        
        print(f"🏆 Best method: {best_info['method']}")
        print(f"   • Combined Score: {best_info['combined_score']:.3f}")
        print(f"   • NMI: {best_info['evaluation']['nmi']:.3f}")
        print(f"   • Adj ARI: {best_info['evaluation']['adj_ari']:.3f}")
        print(f"   • Quality: {best_info['evaluation']['quality_assessment']}")
        print(f"   • Clusters: {best_info['evaluation']['n_clusters']}, Singletons: {best_info['evaluation']['n_noise']}")
        print(f"   • Coverage: {best_info['evaluation']['coverage_ratio']:.3f}")
        
        return best_labels, best_info
    
    def get_stats(self) -> Dict:
        """클러스터링 통계 반환"""
        return self.stats.copy()