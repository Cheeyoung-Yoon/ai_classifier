"""
Final Stage3 Node with Question-wise Clustering
문항별 개별 클러스터링을 사용하는 최종 Stage3 노드
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from question_wise_classification import create_question_wise_classification_pipeline

def stage3_classification_final_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final Stage3 classification node using question-wise clustering
    각 문항별로 개별 클러스터링을 수행하는 최종 노드
    
    Args:
        state: LangGraph state containing matched_questions
        
    Returns:
        Updated state with stage3 results
    """
    print("🔬 Starting Stage3 Classification (Question-wise)")
    print("=" * 60)
    
    try:
        # 1. Configure question-wise classification pipeline
        config = {
            'algorithm': 'adaptive',
            'kmeans_k_range': [3, 4, 5, 6],
            'dbscan_eps_range': [0.1, 0.2, 0.3],
            'dbscan_min_samples_range': [3, 5, 7],
            'hierarchical_k_range': [3, 4, 5, 6],
            'hierarchical_linkage': ['ward', 'complete'],
            'max_clusters': 8,
            'min_samples_per_question': 10,
            'selection_criteria': 'silhouette',
            'clustering_strategy': 'individual_columns',  # 각 컬럼별 개별 클러스터링
            'reference_column_strategy': 'largest'
        }
        
        # 2. Create and run classification pipeline
        print("🎯 Running question-wise clustering...")
        classifier = create_question_wise_classification_pipeline(config)
        results = classifier.cluster_by_questions(state)
        
        # 3. Create results summary
        summary_df = classifier.create_results_summary(results)
        
        # 4. Calculate overall statistics
        total_clusters = 0
        total_algorithms = set()
        total_embeddings = 0
        avg_silhouette = 0
        valid_silhouettes = 0
        
        question_summaries = {}
        
        for question_id, question_result in results['question_results'].items():
            question_summary = {
                'question_id': question_id,
                'clustering_type': question_result['clustering_type'],
                'total_columns': question_result.get('total_columns', 0),
                'clustered_columns': question_result.get('clustered_columns', 0),
                'original_rows': question_result['original_dataframe_rows']
            }
            
            if question_result['clustering_type'] == 'individual_columns':
                # 개별 컬럼 통계
                question_clusters = 0
                question_embeddings = 0
                question_silhouettes = []
                
                for col_name, col_result in question_result['column_results'].items():
                    question_clusters += col_result['n_clusters']
                    question_embeddings += col_result['sample_count']
                    total_algorithms.add(col_result.get('selected_algorithm', 'unknown'))
                    
                    if 'silhouette' in col_result and col_result['silhouette'] > -1:
                        question_silhouettes.append(col_result['silhouette'])
                
                question_summary.update({
                    'total_clusters': question_clusters,
                    'total_embeddings': question_embeddings,
                    'avg_silhouette': np.mean(question_silhouettes) if question_silhouettes else -1,
                    'column_results': question_result['column_results']
                })
                
                total_clusters += question_clusters
                total_embeddings += question_embeddings
                
                if question_silhouettes:
                    avg_silhouette += sum(question_silhouettes)
                    valid_silhouettes += len(question_silhouettes)
            
            question_summaries[question_id] = question_summary
        
        # Overall averages
        overall_avg_silhouette = avg_silhouette / valid_silhouettes if valid_silhouettes > 0 else -1
        
        # 5. Update state with comprehensive results
        state.update({
            # Core results
            'stage3_status': 'completed',
            'stage3_approach': 'question_wise_clustering',
            'stage3_total_questions': results['total_questions'],
            'stage3_processed_questions': results['processed_questions'],
            'stage3_clustering_strategy': results['clustering_strategy'],
            
            # Overall statistics
            'stage3_total_clusters': total_clusters,
            'stage3_total_embeddings': total_embeddings,
            'stage3_avg_silhouette_score': float(overall_avg_silhouette),
            'stage3_algorithms_used': list(total_algorithms),
            
            # Detailed results
            'stage3_question_summaries': question_summaries,
            'stage3_results_summary': summary_df.to_dict('records') if not summary_df.empty else [],
            'stage3_full_results': results,
            
            # Metadata
            'stage3_timestamp': pd.Timestamp.now().isoformat(),
            'stage3_config': config
        })
        
        print(f"\n✅ Stage3 Classification Completed")
        print(f"   Approach: Question-wise clustering")
        print(f"   Processed: {results['processed_questions']}/{results['total_questions']} questions")
        print(f"   Total clusters: {total_clusters}")
        print(f"   Total embeddings: {total_embeddings}")
        print(f"   Average silhouette: {overall_avg_silhouette:.3f}")
        print(f"   Algorithms used: {list(total_algorithms)}")
        
        # Per-question summary
        for question_id, summary in question_summaries.items():
            if summary['clustering_type'] == 'individual_columns':
                print(f"   {question_id}: {summary['clustered_columns']}/{summary['total_columns']} columns, "
                      f"{summary['total_clusters']} clusters, "
                      f"silhouette={summary['avg_silhouette']:.3f}")
        
        return state
        
    except Exception as e:
        print(f"❌ Error in Stage3 Classification: {e}")
        import traceback
        traceback.print_exc()
        
        # Update state with error information
        state.update({
            'stage3_status': 'error',
            'stage3_error': str(e),
            'stage3_approach': 'question_wise_clustering',
            'stage3_total_questions': 0,
            'stage3_processed_questions': 0,
            'stage3_timestamp': pd.Timestamp.now().isoformat()
        })
        
        return state


# Backward compatibility functions
def run_stage3_classification(state: Dict[str, Any]) -> Dict[str, Any]:
    """Backward compatibility wrapper"""
    return stage3_classification_final_node(state)

def stage3_classification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Alternative function name"""
    return stage3_classification_final_node(state)


if __name__ == "__main__":
    # Test the final stage3 node
    print("🧪 Testing Final Stage3 Node")
    print("=" * 50)
    
    # Load test state
    state_file = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/state_history/20250918_113231_081_STAGE2_WORD_문23_COMPLETED_state.json"
    
    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    # Run stage3 node with limited scope for testing
    print("Running limited test (first 2 questions only)...")
    
    # Limit to questions with embeddings for quick test
    limited_questions = {}
    embedding_questions = ['문4', '문5', '문17-1', '문23']  # Questions known to have embeddings
    count = 0
    
    for q_id, q_data in state['matched_questions'].items():
        if q_id in embedding_questions and count < 2:  # Only first 2 embedding questions
            limited_questions[q_id] = q_data
            count += 1
        elif count >= 2:
            break
    
    state['matched_questions'] = limited_questions
    
    # Run stage3 node
    updated_state = stage3_classification_final_node(state)
    
    # Print results
    print(f"\n🎯 Results:")
    print(f"   Status: {updated_state.get('stage3_status', 'unknown')}")
    print(f"   Approach: {updated_state.get('stage3_approach', 'unknown')}")
    print(f"   Processed: {updated_state.get('stage3_processed_questions', 0)}/{updated_state.get('stage3_total_questions', 0)} questions")
    print(f"   Total clusters: {updated_state.get('stage3_total_clusters', 0)}")
    print(f"   Average silhouette: {updated_state.get('stage3_avg_silhouette_score', -1):.3f}")
    
    print("\n✅ Final Stage3 node test completed!")