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
    
    def _assess_quality(self, combined_score: float) -> str:
        """Combined Score 기반 품질 평가 (업데이트된 기준)"""
        if combined_score >= 0.96:
            return "EXCELLENT"
        elif combined_score >= 0.88:
            return "GOOD"
        elif combined_score >= 0.83:
            return "FAIR"
        elif combined_score >= 0.72:
            return "IMPROVE"
        else:
            return "FAIL"