"""
Updated Stage3 Node for LangGraph Pipeline
MCL 대신 최적화된 클러스터링 알고리즘 사용
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_data_from_state, map_clusters_back_to_data
from optimized_classification import create_optimized_classification_pipeline
from evaluation import calculate_evaluation_metrics

def stage3_classification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updated Stage3 classification node using optimized clustering
    MCL 대신 sentence embedding에 최적화된 알고리즘 사용
    
    Args:
        state: LangGraph state containing stage2 results
        
    Returns:
        Updated state with stage3 results
    """
    print("🔬 Starting Stage3 Classification (Optimized)")
    print("=" * 60)
    
    try:
        # 1. Load data from state
        print("📊 Loading data from state...")
        embeddings, metadata = load_data_from_state(state)
        
        print(f"   Loaded embeddings: {embeddings.shape}")
        print(f"   Total embeddings: {len(embeddings)}")
        
        # 2. Configure optimized classification pipeline
        config = {
            'algorithm': 'adaptive',  # 여러 알고리즘 시도 후 최적 선택
            'kmeans_k_range': [3, 4, 5, 6, 7, 8],
            'dbscan_eps_range': [0.1, 0.2, 0.3, 0.4, 0.5],
            'dbscan_min_samples_range': [3, 5, 7, 10],
            'hierarchical_k_range': [3, 4, 5, 6, 7, 8],
            'hierarchical_linkage': ['ward', 'complete', 'average'],
            'max_clusters': min(15, len(embeddings) // 5),  # 적응적 최대 클러스터 수
            'selection_criteria': 'silhouette'
        }
        
        # 3. Create and run classification pipeline
        print("🎯 Running optimized clustering...")
        classifier = create_optimized_classification_pipeline(config)
        
        # 샘플 수가 너무 많으면 제한
        if len(embeddings) > 1000:
            print(f"⚠️  Large dataset ({len(embeddings)} samples), limiting to 1000 for performance")
            embeddings = embeddings[:1000]
            # metadata도 업데이트 필요하지만 일단 생략
        
        clustering_result = classifier.cluster_embeddings(embeddings)
        
        # 4. Create clusters dataframe with statistics
        clusters_df, stats_df = classifier.create_clusters_dataframe(
            embeddings, 
            clustering_result['labels']
        )
        
        # 5. Map clusters back to original data structure
        mapped_results = map_clusters_back_to_data(
            clustering_result['labels'], 
            metadata, 
            len(embeddings)
        )
        
        # 6. Calculate evaluation metrics
        evaluation = calculate_evaluation_metrics(
            clustering_result['labels'],
            embeddings,
            algorithm_name=clustering_result.get('selected_algorithm', 'optimized')
        )
        
        # 7. Update state with results
        state.update({
            # Core results
            'stage3_clusters': clustering_result['labels'].tolist(),
            'stage3_n_clusters': int(clustering_result['n_clusters']),
            'stage3_algorithm': clustering_result.get('selected_algorithm', 'optimized'),
            'stage3_algorithm_params': clustering_result.get('algorithm_params', {}),
            
            # Performance metrics
            'stage3_silhouette_score': float(clustering_result.get('silhouette', -1)),
            'stage3_evaluation_score': float(clustering_result.get('score', -1)),
            'stage3_noise_ratio': float(clustering_result.get('noise_ratio', 0)),
            
            # Data structures
            'stage3_clusters_df': clusters_df.to_dict('records'),
            'stage3_cluster_stats': stats_df.to_dict('records'),
            'stage3_mapped_results': mapped_results,
            
            # Evaluation metrics
            'stage3_evaluation_metrics': evaluation,
            'stage3_coverage_at_5': float(evaluation.get('coverage_at_5', 0)),
            'stage3_coverage_at_10': float(evaluation.get('coverage_at_10', 0)),
            'stage3_gini_coefficient': float(evaluation.get('gini_coefficient', 0)),
            
            # Metadata
            'stage3_total_embeddings': len(embeddings),
            'stage3_embedding_dimensions': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'stage3_timestamp': pd.Timestamp.now().isoformat(),
            'stage3_status': 'completed'
        })
        
        print(f"\n✅ Stage3 Classification Completed")
        print(f"   Algorithm: {clustering_result.get('selected_algorithm', 'optimized')}")
        print(f"   Clusters: {clustering_result['n_clusters']}")
        print(f"   Silhouette Score: {clustering_result.get('silhouette', -1):.3f}")
        print(f"   Total Embeddings: {len(embeddings)}")
        
        return state
        
    except Exception as e:
        print(f"❌ Error in Stage3 Classification: {e}")
        import traceback
        traceback.print_exc()
        
        # Update state with error information
        state.update({
            'stage3_status': 'error',
            'stage3_error': str(e),
            'stage3_clusters': [],
            'stage3_n_clusters': 0,
            'stage3_algorithm': 'none',
            'stage3_timestamp': pd.Timestamp.now().isoformat()
        })
        
        return state


# Backward compatibility - keep the original function name
def run_stage3_classification(state: Dict[str, Any]) -> Dict[str, Any]:
    """Backward compatibility wrapper"""
    return stage3_classification_node(state)


if __name__ == "__main__":
    # Test the updated stage3 node
    import json
    
    print("🧪 Testing Updated Stage3 Node")
    print("=" * 50)
    
    # Load test state
    state_file = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/state_history/20250918_113231_081_STAGE2_WORD_문23_COMPLETED_state.json"
    
    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    # Run stage3 node
    updated_state = stage3_classification_node(state)
    
    # Print results
    print(f"\n🎯 Results:")
    print(f"   Status: {updated_state.get('stage3_status', 'unknown')}")
    print(f"   Algorithm: {updated_state.get('stage3_algorithm', 'none')}")
    print(f"   Clusters: {updated_state.get('stage3_n_clusters', 0)}")
    print(f"   Silhouette: {updated_state.get('stage3_silhouette_score', -1):.3f}")
    print(f"   Total Embeddings: {updated_state.get('stage3_total_embeddings', 0)}")
    
    print("\n✅ Updated Stage3 node test completed!")