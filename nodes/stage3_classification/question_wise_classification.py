"""
Question-wise Classification Pipeline
각 문항(질문)별로 개별적으로 클러스터링을 수행하는 파이프라인
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

class QuestionWiseClassification:
    """
    문항별 개별 클러스터링 파이프라인
    각 문항의 컬럼들만 사용해서 해당 문항에 대한 클러스터링 수행
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize question-wise classification pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for question-wise clustering"""
        return {
            'algorithm': 'adaptive',  # adaptive, kmeans, dbscan, hierarchical
            'kmeans_k_range': [3, 4, 5, 6, 7],
            'dbscan_eps_range': [0.1, 0.2, 0.3, 0.4],
            'dbscan_min_samples_range': [3, 5, 7],
            'hierarchical_k_range': [3, 4, 5, 6, 7],
            'hierarchical_linkage': ['ward', 'complete', 'average'],
            'max_clusters': 10,
            'min_samples_per_question': 20,  # 문항별 최소 샘플 수
            'selection_criteria': 'silhouette',
            'clustering_strategy': 'individual_columns',  # individual_columns, combined_embeddings
            'reference_column_strategy': 'largest'  # largest, first, most_diverse
        }
    
    def cluster_by_questions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        각 문항별로 개별 클러스터링 수행
        
        Args:
            state: LangGraph state containing matched_questions
            
        Returns:
            문항별 클러스터링 결과
        """
        print(f"🔧 Question-wise clustering...")
        
        if "matched_questions" not in state:
            raise ValueError("No matched_questions found in state")
        
        matched_questions = state["matched_questions"]
        question_results = {}
        
        # 각 문항별로 클러스터링 수행
        for question_id, question_data in matched_questions.items():
            print(f"\n📊 Processing question: {question_id}")
            
            try:
                result = self._cluster_single_question(question_id, question_data)
                if result is not None:
                    question_results[question_id] = result
                    print(f"   ✅ {question_id}: Clustering completed")
                else:
                    print(f"   ⚠️  {question_id}: Clustering skipped")
            except Exception as e:
                print(f"   ❌ {question_id}: Error - {e}")
                continue
        
        return {
            'question_results': question_results,
            'total_questions': len(matched_questions),
            'processed_questions': len(question_results),
            'clustering_strategy': self.config['clustering_strategy']
        }
    
    def _cluster_single_question(self, question_id: str, question_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        단일 문항에 대한 클러스터링 수행
        
        Args:
            question_id: 문항 ID
            question_data: 문항 데이터
            
        Returns:
            클러스터링 결과 또는 None
        """
        # Stage2 데이터 확인
        if "stage2_data" not in question_data:
            print(f"   ⚠️  No stage2_data found for {question_id}")
            return None
        
        stage2_data = question_data["stage2_data"]
        if not isinstance(stage2_data, dict) or "csv_path" not in stage2_data:
            print(f"   ⚠️  Invalid stage2_data format for {question_id}")
            return None
        
        csv_path = stage2_data["csv_path"]
        
        try:
            # CSV 데이터 로드
            df = pd.read_csv(csv_path)
            print(f"   📁 Loaded: {len(df)} rows from {Path(csv_path).name}")
            
            # 임베딩 컬럼별로 추출
            embeddings_by_column = self._extract_question_embeddings(df, question_id)
            
            if not embeddings_by_column:
                print(f"   ⚠️  No valid embeddings found for {question_id}")
                return None
            
            # 클러스터링 전략에 따라 처리
            if self.config['clustering_strategy'] == 'individual_columns':
                return self._cluster_individual_columns(question_id, embeddings_by_column, df)
            elif self.config['clustering_strategy'] == 'combined_embeddings':
                return self._cluster_combined_embeddings(question_id, embeddings_by_column, df)
            else:
                print(f"   ❌ Unknown clustering strategy: {self.config['clustering_strategy']}")
                return None
                
        except Exception as e:
            print(f"   ❌ Error processing {question_id}: {e}")
            return None
    
    def _extract_question_embeddings(self, df: pd.DataFrame, question_id: str) -> Dict[str, np.ndarray]:
        """
        문항의 DataFrame에서 임베딩 추출
        
        Args:
            df: 문항 DataFrame
            question_id: 문항 ID
            
        Returns:
            컬럼별 임베딩 딕셔너리
        """
        embeddings_by_column = {}
        
        # 임베딩 컬럼 찾기
        embedding_columns = [col for col in df.columns if 'embed' in col.lower()]
        print(f"   🔍 Found {len(embedding_columns)} embedding columns")
        
        for col in embedding_columns:
            try:
                embeddings = []
                
                for idx, value in df[col].items():
                    if pd.isna(value):
                        continue
                        
                    try:
                        # 문자열로 저장된 임베딩을 배열로 변환
                        if isinstance(value, str):
                            import ast
                            embedding = ast.literal_eval(value)
                        else:
                            embedding = value
                        
                        # numpy 배열로 변환
                        embedding_array = np.array(embedding, dtype=np.float32)
                        
                        # 1차원 배열인지 확인
                        if embedding_array.ndim == 1 and len(embedding_array) > 0:
                            embeddings.append(embedding_array)
                            
                    except Exception as e:
                        continue  # 변환 실패한 임베딩은 스킵
                
                if embeddings:
                    # 모든 임베딩을 2D 배열로 변환
                    embeddings_array = np.vstack(embeddings)
                    
                    # 컬럼명 정리
                    clean_col_name = col.replace('embed_', '').replace('_embed', '')
                    embeddings_by_column[clean_col_name] = embeddings_array
                    
                    print(f"     📊 {clean_col_name}: {len(embeddings_array)} embeddings")
                    
            except Exception as e:
                print(f"     ❌ Error processing column {col}: {e}")
                continue
        
        return embeddings_by_column
    
    def _cluster_individual_columns(self, question_id: str, 
                                   embeddings_by_column: Dict[str, np.ndarray],
                                   df: pd.DataFrame) -> Dict[str, Any]:
        """
        각 컬럼별로 개별 클러스터링 수행
        
        Args:
            question_id: 문항 ID
            embeddings_by_column: 컬럼별 임베딩
            df: 원본 DataFrame
            
        Returns:
            클러스터링 결과
        """
        print(f"   🎯 Individual column clustering for {question_id}")
        
        column_results = {}
        
        # 각 컬럼별로 클러스터링
        for col_name, embeddings in embeddings_by_column.items():
            if len(embeddings) < self.config['min_samples_per_question']:
                print(f"     ⚠️  Skipping {col_name}: insufficient samples ({len(embeddings)})")
                continue
            
            try:
                # 개별 컬럼 클러스터링
                result = self._cluster_single_column(embeddings, f"{question_id}_{col_name}")
                if result is not None:
                    column_results[col_name] = result
                    print(f"     ✅ {col_name}: {result['n_clusters']} clusters")
                    
            except Exception as e:
                print(f"     ❌ Error clustering {col_name}: {e}")
                continue
        
        if not column_results:
            return None
        
        # 기준 컬럼 선택
        reference_column = self._select_reference_column(embeddings_by_column)
        print(f"   📌 Reference column: {reference_column}")
        
        return {
            'question_id': question_id,
            'clustering_type': 'individual_columns',
            'column_results': column_results,
            'reference_column': reference_column,
            'total_columns': len(embeddings_by_column),
            'clustered_columns': len(column_results),
            'original_dataframe_rows': len(df)
        }
    
    def _cluster_combined_embeddings(self, question_id: str,
                                   embeddings_by_column: Dict[str, np.ndarray],
                                   df: pd.DataFrame) -> Dict[str, Any]:
        """
        모든 컬럼의 임베딩을 합쳐서 단일 클러스터링 수행
        
        Args:
            question_id: 문항 ID
            embeddings_by_column: 컬럼별 임베딩
            df: 원본 DataFrame
            
        Returns:
            클러스터링 결과
        """
        print(f"   🎯 Combined embeddings clustering for {question_id}")
        
        # 모든 임베딩을 합치기
        all_embeddings = []
        embedding_mapping = []  # (row_idx, col_name) 매핑
        
        for col_name, embeddings in embeddings_by_column.items():
            for i, embedding in enumerate(embeddings):
                all_embeddings.append(embedding)
                embedding_mapping.append((i, col_name))
        
        if len(all_embeddings) < self.config['min_samples_per_question']:
            print(f"     ⚠️  Insufficient total samples: {len(all_embeddings)}")
            return None
        
        # 합쳐진 임베딩으로 클러스터링
        combined_embeddings = np.vstack(all_embeddings)
        result = self._cluster_single_column(combined_embeddings, question_id)
        
        if result is None:
            return None
        
        # 결과를 컬럼별로 분리
        column_labels = {}
        for i, (row_idx, col_name) in enumerate(embedding_mapping):
            if col_name not in column_labels:
                column_labels[col_name] = []
            column_labels[col_name].append(result['labels'][i])
        
        return {
            'question_id': question_id,
            'clustering_type': 'combined_embeddings',
            'combined_result': result,
            'column_labels': column_labels,
            'embedding_mapping': embedding_mapping,
            'total_embeddings': len(all_embeddings),
            'original_dataframe_rows': len(df)
        }
    
    def _cluster_single_column(self, embeddings: np.ndarray, name: str) -> Optional[Dict[str, Any]]:
        """단일 컬럼/데이터셋 클러스터링"""
        try:
            from optimized_classification import OptimizedClassification
            
            # 컬럼용 설정
            column_config = self.config.copy()
            column_config['max_clusters'] = min(self.config['max_clusters'], len(embeddings) // 3)
            
            classifier = OptimizedClassification(column_config)
            result = classifier.cluster_embeddings(embeddings)
            
            if result is not None:
                result['name'] = name
                result['sample_count'] = len(embeddings)
                return result
                
        except Exception as e:
            print(f"     ❌ Error clustering {name}: {e}")
            
        return None
    
    def _select_reference_column(self, embeddings_by_column: Dict[str, np.ndarray]) -> str:
        """기준 컬럼 선택"""
        if not embeddings_by_column:
            raise ValueError("No columns available")
        
        strategy = self.config['reference_column_strategy']
        
        if strategy == 'largest':
            return max(embeddings_by_column.keys(), 
                      key=lambda k: len(embeddings_by_column[k]))
        elif strategy == 'first':
            return list(embeddings_by_column.keys())[0]
        elif strategy == 'most_diverse':
            max_variance = -1
            best_column = None
            
            for col_name, embeddings in embeddings_by_column.items():
                if len(embeddings) > 10:
                    variance = np.var(embeddings, axis=0).mean()
                    if variance > max_variance:
                        max_variance = variance
                        best_column = col_name
            
            return best_column or list(embeddings_by_column.keys())[0]
        else:
            return list(embeddings_by_column.keys())[0]
    
    def create_results_summary(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        결과 요약 DataFrame 생성
        
        Args:
            results: cluster_by_questions 결과
            
        Returns:
            요약 DataFrame
        """
        summary_data = []
        
        for question_id, question_result in results['question_results'].items():
            base_info = {
                'question_id': question_id,
                'clustering_type': question_result['clustering_type'],
                'original_rows': question_result['original_dataframe_rows']
            }
            
            if question_result['clustering_type'] == 'individual_columns':
                # 개별 컬럼 결과
                for col_name, col_result in question_result['column_results'].items():
                    row_data = base_info.copy()
                    row_data.update({
                        'column_name': col_name,
                        'n_clusters': col_result['n_clusters'],
                        'algorithm': col_result.get('selected_algorithm', 'unknown'),
                        'silhouette_score': col_result.get('silhouette', -1),
                        'sample_count': col_result['sample_count']
                    })
                    summary_data.append(row_data)
            
            elif question_result['clustering_type'] == 'combined_embeddings':
                # 합쳐진 결과
                combined_result = question_result['combined_result']
                row_data = base_info.copy()
                row_data.update({
                    'column_name': 'combined',
                    'n_clusters': combined_result['n_clusters'],
                    'algorithm': combined_result.get('selected_algorithm', 'unknown'),
                    'silhouette_score': combined_result.get('silhouette', -1),
                    'sample_count': question_result['total_embeddings']
                })
                summary_data.append(row_data)
        
        return pd.DataFrame(summary_data)


def create_question_wise_classification_pipeline(config: Optional[Dict] = None) -> QuestionWiseClassification:
    """Create question-wise classification pipeline"""
    return QuestionWiseClassification(config)


if __name__ == "__main__":
    # Test the question-wise pipeline
    import json
    
    print("🧪 Testing Question-wise Classification Pipeline")
    print("=" * 60)
    
    # Load test data
    state_file = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/state_history/20250918_113231_081_STAGE2_WORD_문23_COMPLETED_state.json"
    
    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    # Test question-wise clustering
    classifier = create_question_wise_classification_pipeline()
    
    # 빠른 테스트를 위해 설정 조정
    classifier.config.update({
        'min_samples_per_question': 10,
        'kmeans_k_range': [3, 4, 5],
        'clustering_strategy': 'individual_columns'
    })
    
    results = classifier.cluster_by_questions(state)
    
    print(f"\n🎯 Question-wise Clustering Results:")
    print(f"   Processed questions: {results['processed_questions']}/{results['total_questions']}")
    print(f"   Strategy: {results['clustering_strategy']}")
    
    # 결과 요약 출력
    summary_df = classifier.create_results_summary(results)
    print(f"\n📊 Results Summary:")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("   No clustering results to display")
    
    # 각 문항별 상세 결과
    for question_id, question_result in results['question_results'].items():
        print(f"\n📋 {question_id} Details:")
        if question_result['clustering_type'] == 'individual_columns':
            for col_name, col_result in question_result['column_results'].items():
                print(f"   {col_name}: {col_result['n_clusters']} clusters, "
                      f"algorithm={col_result.get('selected_algorithm', 'unknown')}, "
                      f"silhouette={col_result.get('silhouette', -1):.3f}")
    
    print("\n✅ Question-wise classification pipeline test completed!")