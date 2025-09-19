"""
Stage3 Classification Pipeline with Optimized Algorithm
MCL 대신 sentence embedding에 최적화된 클러스터링 알고리즘 적용
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, Tuple, List, Optional
import json
from pathlib import Path

class OptimizedClassification:
    """
    MCL 대신 sentence embedding에 최적화된 클러스터링 파이프라인
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with optimized configuration for sentence embeddings
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration optimized for sentence embeddings"""
        return {
            'algorithm': 'adaptive',  # adaptive, kmeans, dbscan, hierarchical
            'kmeans_k_range': [3, 4, 5, 6, 7],  # K 값 후보
            'dbscan_eps_range': [0.1, 0.2, 0.3, 0.4],  # epsilon 후보
            'dbscan_min_samples_range': [3, 5, 7],  # min_samples 후보
            'hierarchical_k_range': [3, 4, 5, 6, 7],  # K 값 후보
            'hierarchical_linkage': ['ward', 'complete', 'average'],  # linkage 방법
            'min_cluster_size': 2,  # 최소 클러스터 크기
            'max_clusters': 10,  # 최대 클러스터 수
            'evaluation_metrics': ['silhouette', 'nmi', 'ari'],  # 평가 지표
            'selection_criteria': 'silhouette'  # 최종 선택 기준
        }
    
    def cluster_embeddings(self, embeddings: np.ndarray, true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Cluster embeddings using optimized algorithm
        
        Args:
            embeddings: Input embeddings (n_samples, n_features)
            true_labels: True labels for evaluation (optional)
            
        Returns:
            Dictionary containing clustering results
        """
        print(f"🔧 Clustering {len(embeddings)} embeddings...")
        
        if self.config['algorithm'] == 'adaptive':
            return self._adaptive_clustering(embeddings, true_labels)
        elif self.config['algorithm'] == 'kmeans':
            return self._kmeans_clustering(embeddings, true_labels)
        elif self.config['algorithm'] == 'dbscan':
            return self._dbscan_clustering(embeddings, true_labels)
        elif self.config['algorithm'] == 'hierarchical':
            return self._hierarchical_clustering(embeddings, true_labels)
        else:
            raise ValueError(f"Unknown algorithm: {self.config['algorithm']}")
    
    def _adaptive_clustering(self, embeddings: np.ndarray, true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Adaptive clustering: 여러 알고리즘을 시도하고 최적 결과 선택
        """
        print("🎯 Running adaptive clustering...")
        
        candidates = []
        
        # 1. K-Means 시도
        kmeans_results = self._kmeans_clustering(embeddings, true_labels)
        candidates.append(('kmeans', kmeans_results))
        
        # 2. DBSCAN 시도
        dbscan_results = self._dbscan_clustering(embeddings, true_labels)
        candidates.append(('dbscan', dbscan_results))
        
        # 3. Hierarchical 시도
        hierarchical_results = self._hierarchical_clustering(embeddings, true_labels)
        candidates.append(('hierarchical', hierarchical_results))
        
        # 최적 결과 선택
        best_algo, best_result = self._select_best_result(candidates, embeddings)
        
        print(f"🏆 Selected algorithm: {best_algo}")
        print(f"   Score: {best_result['score']:.3f}")
        print(f"   Clusters: {best_result['n_clusters']}")
        
        best_result['selected_algorithm'] = best_algo
        return best_result
    
    def _kmeans_clustering(self, embeddings: np.ndarray, true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """K-Means clustering with optimal K selection"""
        print("🔍 Testing K-Means...")
        
        best_score = -1
        best_result = None
        
        for k in self.config['kmeans_k_range']:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # 평가
            score_info = self._evaluate_clustering(embeddings, labels, true_labels)
            
            if score_info['score'] > best_score:
                best_score = score_info['score']
                best_result = {
                    'labels': labels,
                    'n_clusters': k,
                    'algorithm_params': {'k': k},
                    **score_info
                }
        
        return best_result
    
    def _dbscan_clustering(self, embeddings: np.ndarray, true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """DBSCAN clustering with optimal parameter selection"""
        print("🔍 Testing DBSCAN...")
        
        best_score = -1
        best_result = None
        
        for eps in self.config['dbscan_eps_range']:
            for min_samples in self.config['dbscan_min_samples_range']:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                labels = dbscan.fit_predict(embeddings)
                
                # 유효한 클러스터가 있는지 확인
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters < 2 or n_clusters > self.config['max_clusters']:
                    continue
                
                # 평가
                score_info = self._evaluate_clustering(embeddings, labels, true_labels)
                
                if score_info['score'] > best_score:
                    best_score = score_info['score']
                    best_result = {
                        'labels': labels,
                        'n_clusters': n_clusters,
                        'algorithm_params': {'eps': eps, 'min_samples': min_samples},
                        'noise_ratio': np.sum(labels == -1) / len(labels),
                        **score_info
                    }
        
        return best_result
    
    def _hierarchical_clustering(self, embeddings: np.ndarray, true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Hierarchical clustering with optimal parameters"""
        print("🔍 Testing Hierarchical clustering...")
        
        best_score = -1
        best_result = None
        
        for k in self.config['hierarchical_k_range']:
            for linkage in self.config['hierarchical_linkage']:
                try:
                    agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                    labels = agg.fit_predict(embeddings)
                    
                    # 평가
                    score_info = self._evaluate_clustering(embeddings, labels, true_labels)
                    
                    if score_info['score'] > best_score:
                        best_score = score_info['score']
                        best_result = {
                            'labels': labels,
                            'n_clusters': k,
                            'algorithm_params': {'k': k, 'linkage': linkage},
                            **score_info
                        }
                except Exception as e:
                    print(f"⚠️  Skipping {linkage}-{k}: {e}")
                    continue
        
        return best_result
    
    def _evaluate_clustering(self, embeddings: np.ndarray, labels: np.ndarray, true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate clustering results"""
        evaluation = {}
        
        # Silhouette score (unsupervised)
        if len(set(labels)) > 1:
            # DBSCAN에서 노이즈 포인트 제거
            valid_mask = labels != -1
            if np.sum(valid_mask) > 1 and len(set(labels[valid_mask])) > 1:
                try:
                    evaluation['silhouette'] = silhouette_score(embeddings[valid_mask], labels[valid_mask])
                except:
                    evaluation['silhouette'] = -1
            else:
                evaluation['silhouette'] = -1
        else:
            evaluation['silhouette'] = -1
        
        # Supervised metrics (if true labels available)
        if true_labels is not None:
            try:
                evaluation['nmi'] = normalized_mutual_info_score(true_labels, labels)
                evaluation['ari'] = adjusted_rand_score(true_labels, labels)
            except:
                evaluation['nmi'] = 0
                evaluation['ari'] = 0
        else:
            evaluation['nmi'] = 0
            evaluation['ari'] = 0
        
        # Composite score
        if self.config['selection_criteria'] == 'silhouette':
            evaluation['score'] = evaluation['silhouette']
        elif self.config['selection_criteria'] == 'supervised' and true_labels is not None:
            evaluation['score'] = (evaluation['nmi'] + evaluation['ari']) / 2
        else:
            evaluation['score'] = evaluation['silhouette']
        
        return evaluation
    
    def _select_best_result(self, candidates: List[Tuple[str, Dict]], embeddings: np.ndarray) -> Tuple[str, Dict]:
        """Select best clustering result from candidates"""
        
        # Filter out None results
        valid_candidates = [(name, result) for name, result in candidates if result is not None]
        
        if not valid_candidates:
            # Fallback to simple K-means
            print("⚠️  No valid clustering results, using fallback K-means")
            kmeans = KMeans(n_clusters=3, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            score_info = self._evaluate_clustering(embeddings, labels)
            fallback_result = {
                'labels': labels,
                'n_clusters': 3,
                'algorithm_params': {'k': 3},
                **score_info
            }
            return 'kmeans_fallback', fallback_result
        
        # Select best based on score
        best_name, best_result = max(valid_candidates, key=lambda x: x[1]['score'])
        return best_name, best_result
    
    def create_clusters_dataframe(self, embeddings: np.ndarray, labels: np.ndarray, 
                                 original_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create clusters dataframe with statistics"""
        
        df = pd.DataFrame({
            'embedding_id': range(len(embeddings)),
            'cluster_id': labels
        })
        
        # Add original data if available
        if original_data is not None:
            for col in original_data.columns:
                if col not in df.columns:
                    df[col] = original_data[col].iloc[:len(df)].values
        
        # Add cluster statistics
        cluster_stats = []
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:  # DBSCAN noise
                continue
                
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            
            # Intra-cluster similarity
            if len(cluster_embeddings) > 1:
                sim_matrix = cosine_similarity(cluster_embeddings)
                mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
                intra_similarity = sim_matrix[mask].mean()
            else:
                intra_similarity = 1.0
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': np.sum(cluster_mask),
                'intra_similarity': intra_similarity
            })
        
        stats_df = pd.DataFrame(cluster_stats)
        
        return df, stats_df


def create_optimized_classification_pipeline(config: Optional[Dict] = None) -> OptimizedClassification:
    """Create optimized classification pipeline for sentence embeddings"""
    return OptimizedClassification(config)


if __name__ == "__main__":
    # Test the optimized pipeline
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from data_loader import load_data_from_state
    
    print("🧪 Testing Optimized Classification Pipeline")
    print("=" * 60)
    
    # Load test data
    state_file = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/state_history/20250918_113231_081_STAGE2_WORD_문23_COMPLETED_state.json"
    
    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    embeddings, metadata = load_data_from_state(state)
    
    # Limit to 100 samples for testing
    embeddings = embeddings[:100]
    
    # Test the pipeline
    classifier = create_optimized_classification_pipeline()
    result = classifier.cluster_embeddings(embeddings)
    
    print(f"\n🎯 Clustering Results:")
    print(f"   Algorithm: {result.get('selected_algorithm', 'N/A')}")
    print(f"   Clusters: {result['n_clusters']}")
    print(f"   Silhouette: {result['silhouette']:.3f}")
    print(f"   Score: {result['score']:.3f}")
    
    # Create clusters dataframe
    clusters_df, stats_df = classifier.create_clusters_dataframe(embeddings, result['labels'])
    
    print(f"\n📊 Cluster Statistics:")
    print(stats_df.to_string(index=False, float_format='%.3f'))
    
    print("\n✅ Optimized classification pipeline test completed!")