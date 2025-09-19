"""
Column-wise Clustering Pipeline
각 컬럼별로 클러스터링을 수행하고 첫 번째 컬럼 기준으로 매칭하는 파이프라인
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, Tuple, List, Optional, Union
import json
from pathlib import Path
from collections import defaultdict

class ColumnWiseClassification:
    """
    컬럼별 클러스터링 후 매칭하는 파이프라인
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize column-wise classification pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for column-wise clustering"""
        return {
            'algorithm': 'adaptive',  # adaptive, kmeans, dbscan, hierarchical
            'kmeans_k_range': [3, 4, 5, 6, 7],
            'dbscan_eps_range': [0.1, 0.2, 0.3, 0.4],
            'dbscan_min_samples_range': [3, 5, 7],
            'hierarchical_k_range': [3, 4, 5, 6, 7],
            'hierarchical_linkage': ['ward', 'complete', 'average'],
            'max_clusters': 10,
            'min_samples_per_column': 10,  # 컬럼별 최소 샘플 수
            'selection_criteria': 'silhouette',
            'matching_strategy': 'majority_vote',  # majority_vote, weighted_similarity
            'reference_column_index': 0  # 기준 컬럼 인덱스
        }
    
    def cluster_by_columns(self, embeddings_dict: Dict[str, np.ndarray], 
                          metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        각 컬럼별로 클러스터링 수행
        
        Args:
            embeddings_dict: 컬럼별 임베딩 딕셔너리 {column_name: embeddings}
            metadata: 메타데이터
            
        Returns:
            컬럼별 클러스터링 결과
        """
        print(f"🔧 Column-wise clustering for {len(embeddings_dict)} columns...")
        
        column_results = {}
        column_names = list(embeddings_dict.keys())
        
        # 각 컬럼별로 클러스터링 수행
        for i, (col_name, embeddings) in enumerate(embeddings_dict.items()):
            print(f"\n📊 Processing column: {col_name} ({len(embeddings)} samples)")
            
            # 최소 샘플 수 확인
            if len(embeddings) < self.config['min_samples_per_column']:
                print(f"⚠️  Skipping {col_name}: insufficient samples ({len(embeddings)} < {self.config['min_samples_per_column']})")
                continue
            
            # 개별 컬럼 클러스터링
            result = self._cluster_single_column(embeddings, col_name)
            if result is not None:
                column_results[col_name] = result
        
        # 기준 컬럼 확인
        reference_column = column_names[self.config['reference_column_index']]
        if reference_column not in column_results:
            print(f"⚠️  Reference column {reference_column} not found, using first available")
            reference_column = list(column_results.keys())[0] if column_results else None
        
        if not reference_column:
            raise ValueError("No valid clustering results found")
        
        print(f"🎯 Using reference column: {reference_column}")
        
        # 매칭 수행
        matched_results = self._match_clusters_to_reference(
            column_results, reference_column, metadata
        )
        
        return {
            'column_results': column_results,
            'reference_column': reference_column,
            'matched_results': matched_results,
            'total_columns': len(embeddings_dict),
            'processed_columns': len(column_results)
        }
    
    def _cluster_single_column(self, embeddings: np.ndarray, 
                              column_name: str) -> Optional[Dict[str, Any]]:
        """단일 컬럼 클러스터링"""
        try:
            from optimized_classification import OptimizedClassification
            
            # 개별 컬럼용 설정
            column_config = self.config.copy()
            column_config['max_clusters'] = min(self.config['max_clusters'], len(embeddings) // 3)
            
            classifier = OptimizedClassification(column_config)
            result = classifier.cluster_embeddings(embeddings)
            
            if result is not None:
                result['column_name'] = column_name
                result['sample_count'] = len(embeddings)
                
                print(f"   ✅ {column_name}: {result['n_clusters']} clusters, "
                      f"silhouette={result.get('silhouette', -1):.3f}")
                
                return result
            
        except Exception as e:
            print(f"   ❌ Error clustering {column_name}: {e}")
            
        return None
    
    def _match_clusters_to_reference(self, column_results: Dict[str, Dict], 
                                   reference_column: str,
                                   metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        기준 컬럼의 클러스터에 다른 컬럼들의 클러스터를 매칭
        """
        print(f"\n🔗 Matching clusters to reference: {reference_column}")
        
        reference_result = column_results[reference_column]
        reference_labels = reference_result['labels']
        reference_clusters = set(reference_labels)
        
        print(f"   Reference has {len(reference_clusters)} clusters")
        
        # 각 행(row)별 클러스터 정보 수집
        row_mappings = self._build_row_mappings(metadata)
        
        # 기준 컬럼의 각 클러스터에 대해 다른 컬럼들의 정보 매칭
        cluster_profiles = {}
        
        for ref_cluster in reference_clusters:
            if ref_cluster == -1:  # 노이즈 클러스터 스킵
                continue
                
            cluster_profiles[ref_cluster] = self._build_cluster_profile(
                ref_cluster, reference_labels, column_results, row_mappings
            )
        
        # 최종 통합 라벨 생성
        final_labels = self._generate_final_labels(
            reference_labels, cluster_profiles, column_results
        )
        
        return {
            'final_labels': final_labels,
            'cluster_profiles': cluster_profiles,
            'reference_clusters': len(reference_clusters),
            'matching_strategy': self.config['matching_strategy'],
            'row_mappings': row_mappings
        }
    
    def _build_row_mappings(self, metadata: Dict[str, Any]) -> Dict[int, Dict[str, int]]:
        """
        각 행이 어떤 컬럼의 몇 번째 임베딩에 해당하는지 매핑 구성
        """
        row_mappings = {}
        
        # metadata에서 행-컬럼 매핑 정보 추출
        if 'row_mapping' in metadata:
            for global_idx, (original_row, col_idx) in enumerate(metadata['row_mapping']):
                if original_row not in row_mappings:
                    row_mappings[original_row] = {}
                
                # 컬럼명 매핑
                col_name = metadata.get('col_mapping', {}).get(col_idx, f"col_{col_idx}")
                row_mappings[original_row][col_name] = global_idx
        
        return row_mappings
    
    def _build_cluster_profile(self, ref_cluster: int, reference_labels: np.ndarray,
                              column_results: Dict[str, Dict],
                              row_mappings: Dict[int, Dict[str, int]]) -> Dict[str, Any]:
        """
        기준 클러스터에 대한 다른 컬럼들의 클러스터 분포 프로파일 구성
        """
        profile = {
            'reference_cluster': ref_cluster,
            'size': np.sum(reference_labels == ref_cluster),
            'column_distributions': {},
            'majority_clusters': {},
            'confidence_scores': {}
        }
        
        # 기준 클러스터에 속하는 샘플들의 인덱스
        ref_indices = np.where(reference_labels == ref_cluster)[0]
        
        # 각 컬럼별로 해당 샘플들의 클러스터 분포 계산
        for col_name, col_result in column_results.items():
            if col_name not in row_mappings or len(row_mappings[col_name]) == 0:
                continue
                
            col_labels = col_result['labels']
            
            # 기준 클러스터 샘플들이 이 컬럼에서 어떤 클러스터에 속하는지 확인
            mapped_clusters = []
            
            for ref_idx in ref_indices:
                # row_mappings를 통해 해당 샘플이 이 컬럼의 몇 번째 임베딩에 해당하는지 찾기
                # 간단화: 인덱스 기반 매핑 (실제로는 더 복잡한 매핑 필요)
                if ref_idx < len(col_labels):
                    mapped_clusters.append(col_labels[ref_idx])
            
            if mapped_clusters:
                # 클러스터 분포 계산
                unique_clusters, counts = np.unique(mapped_clusters, return_counts=True)
                distribution = dict(zip(unique_clusters.astype(int), counts.astype(int)))
                
                # 가장 많은 클러스터 찾기
                majority_cluster = unique_clusters[np.argmax(counts)]
                confidence = np.max(counts) / len(mapped_clusters)
                
                profile['column_distributions'][col_name] = distribution
                profile['majority_clusters'][col_name] = int(majority_cluster)
                profile['confidence_scores'][col_name] = float(confidence)
        
        return profile
    
    def _generate_final_labels(self, reference_labels: np.ndarray,
                              cluster_profiles: Dict[int, Dict],
                              column_results: Dict[str, Dict]) -> np.ndarray:
        """
        최종 통합 라벨 생성
        """
        if self.config['matching_strategy'] == 'majority_vote':
            return self._majority_vote_labels(reference_labels, cluster_profiles)
        elif self.config['matching_strategy'] == 'weighted_similarity':
            return self._weighted_similarity_labels(reference_labels, cluster_profiles, column_results)
        else:
            # 기본적으로는 reference 라벨 사용
            return reference_labels
    
    def _majority_vote_labels(self, reference_labels: np.ndarray,
                             cluster_profiles: Dict[int, Dict]) -> np.ndarray:
        """
        다수결 투표 방식으로 최종 라벨 결정
        """
        # 현재는 단순히 reference 라벨 반환
        # 실제로는 각 컬럼의 클러스터 정보를 종합해서 결정
        return reference_labels
    
    def _weighted_similarity_labels(self, reference_labels: np.ndarray,
                                   cluster_profiles: Dict[int, Dict],
                                   column_results: Dict[str, Dict]) -> np.ndarray:
        """
        가중 유사도 방식으로 최종 라벨 결정
        """
        # 현재는 단순히 reference 라벨 반환
        # 실제로는 실루엣 스코어 등을 가중치로 사용
        return reference_labels
    
    def create_results_dataframe(self, results: Dict[str, Any],
                                embeddings_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        결과를 DataFrame으로 변환
        """
        data = []
        
        # 기준 컬럼 정보
        reference_column = results['reference_column']
        matched_results = results['matched_results']
        final_labels = matched_results['final_labels']
        
        for i, label in enumerate(final_labels):
            row_data = {
                'sample_index': i,
                'final_cluster': label,
                'reference_column': reference_column
            }
            
            # 각 컬럼별 클러스터 정보 추가
            for col_name, col_result in results['column_results'].items():
                if i < len(col_result['labels']):
                    row_data[f'{col_name}_cluster'] = col_result['labels'][i]
                    row_data[f'{col_name}_algorithm'] = col_result.get('selected_algorithm', 'unknown')
            
            data.append(row_data)
        
        return pd.DataFrame(data)


def create_column_wise_classification_pipeline(config: Optional[Dict] = None) -> ColumnWiseClassification:
    """Create column-wise classification pipeline"""
    return ColumnWiseClassification(config)


if __name__ == "__main__":
    # Test the column-wise pipeline
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from data_loader import load_data_from_state
    
    print("🧪 Testing Column-wise Classification Pipeline")
    print("=" * 60)
    
    # Load test data
    state_file = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/state_history/20250918_113231_081_STAGE2_WORD_문23_COMPLETED_state.json"
    
    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    embeddings, metadata = load_data_from_state(state)
    
    print(f"Total embeddings loaded: {embeddings.shape}")
    
    # 컬럼별로 임베딩 분리 (시뮬레이션)
    # 실제로는 metadata에서 컬럼 정보를 추출해야 함
    n_columns = 5
    samples_per_column = len(embeddings) // n_columns
    
    embeddings_dict = {}
    for i in range(n_columns):
        start_idx = i * samples_per_column
        end_idx = min((i + 1) * samples_per_column, len(embeddings))
        embeddings_dict[f'column_{i+1}'] = embeddings[start_idx:end_idx]
    
    print(f"Split into {len(embeddings_dict)} columns:")
    for col_name, col_emb in embeddings_dict.items():
        print(f"  {col_name}: {col_emb.shape}")
    
    # Test column-wise clustering
    classifier = create_column_wise_classification_pipeline()
    results = classifier.cluster_by_columns(embeddings_dict, metadata)
    
    print(f"\n🎯 Column-wise Clustering Results:")
    print(f"   Processed columns: {results['processed_columns']}/{results['total_columns']}")
    print(f"   Reference column: {results['reference_column']}")
    
    # Show results for each column
    for col_name, col_result in results['column_results'].items():
        print(f"   {col_name}: {col_result['n_clusters']} clusters, "
              f"algorithm={col_result.get('selected_algorithm', 'unknown')}")
    
    # Create results dataframe
    results_df = classifier.create_results_dataframe(results, embeddings_dict)
    print(f"\n📊 Results DataFrame: {results_df.shape}")
    print(results_df.head())
    
    print("\n✅ Column-wise classification pipeline test completed!")