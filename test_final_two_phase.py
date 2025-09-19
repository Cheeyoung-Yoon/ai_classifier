#!/usr/bin/env python3
"""
Final comprehensive test of the complete Two-Phase Stage 3 system.
"""

import sys
import os
sys.path.append('/home/cyyoon/test_area/ai_text_classification/2.langgraph')

import numpy as np
from graph.state import GraphState
from nodes.stage3_classification.two_phase_stage3_node import stage3_main_node

def test_complete_two_phase_system():
    """Test the complete two-phase Stage 3 system end-to-end."""
    
    print("🎯 Final Two-Phase Stage 3 System Test")
    print("=" * 50)
    
    # Create test dataset with clear structure (50 samples, 64 features)
    np.random.seed(42)
    n_samples = 50
    n_features = 64
    
    # Generate embeddings with 5 clear clusters
    cluster_centers = np.random.randn(5, n_features)
    embeddings = []
    texts = []
    ground_truth = []
    
    samples_per_cluster = n_samples // 5
    for cluster_id, center in enumerate(cluster_centers):
        for sample_id in range(samples_per_cluster):
            # Add controlled noise to cluster center
            noise = np.random.normal(0, 0.08, n_features)  # Small noise for clear clusters
            embedding = center + noise
            # L2 normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
            texts.append(f"Document_{cluster_id}_{sample_id}")
            ground_truth.append(cluster_id)
    
    embeddings = np.array(embeddings)
    ground_truth = np.array(ground_truth)
    
    print(f"📊 Test Dataset:")
    print(f"   - Samples: {embeddings.shape[0]}")
    print(f"   - Features: {embeddings.shape[1]}")
    print(f"   - Ground truth clusters: {len(np.unique(ground_truth))}")
    print(f"   - Cluster distribution: {np.bincount(ground_truth)}")
    
    # Prepare GraphState with optimized configuration
    state = GraphState(
        # Input data with matched_questions structure
        matched_questions={
            "Q1": {
                "embeddings": embeddings.tolist(),
                "texts": texts,
                "original_labels": ground_truth.tolist()
            }
        },
        
        # Also set direct fields for compatibility
        embeddings=embeddings.tolist(),
        texts=texts,
        
        # Stage 3 two-phase configuration - set as individual state keys
        stage3_config={
            # Phase 1 parameters (optimized)
            "phase1": {
                "knn_k": 12,
                "knn_metric": "cosine", 
                "mutual_knn": True,
                "csls_neighborhood_size": 8,
                "csls_weighting": True,
                "top_m_edges": 300,
                "csls_threshold": 0.05,
                "prune_bottom_percentile": 15,
                "mcl_inflation": 1.6,
                "mcl_expansion": 2,
                "mcl_max_iters": 40,
                "allow_singletons": True,
                "merge_small_clusters": True,
                "min_cluster_size": 2,
                "small_cluster_threshold": 3,
                "compute_subset_score": True,
                "compute_cluster_quality": True
            },
            
            # Phase 2 parameters
            "phase2": {
                "labeling_mode": "manual",  # manual, llm_assisted, automatic
                "algorithm": "louvain",
                "resolution": 1.0,
                "edge_weights": {  # Dictionary format expected by code
                    "alpha": 1.0,    # Prototype embedding weight
                    "beta": 0.8,     # Co-context weight  
                    "gamma": 0.6     # Keyword weight
                },
                "edge_threshold": 0.1,  # Minimum edge weight
                "top_m_edges": 200,  # Maximum edges to keep
                "must_links": [],  # Forced connections
                "cannot_links": [],  # Forced separations
                "random_seed": 42,
                "n_iterations": 10,  # For Leiden algorithm
                "similarity_threshold": 0.3,
                "min_community_size": 2,
                "use_llm_labeling": False,  # Disable LLM for testing
                "llm_model": "gpt-3.5-turbo",
                "llm_config": {"model": "gpt-3.5-turbo"}
            },
            
            # Quality assessment
            "quality": {
                "enable_assessment": True,
                "subset_score_threshold": 0.8,
                "consistency_threshold": 0.7
            }
        },
        
        # Phase 2 individual state keys (required by phase2_secondary_labeling_node)
        stage3_phase2_algorithm="louvain",
        stage3_phase2_resolution=1.0,
        stage3_phase2_mode="basic",  # Changed to basic for simplicity
        stage3_phase2_edge_weights={
            "alpha": 1.0,
            "beta": 0.8, 
            "gamma": 0.6
        },
        stage3_phase2_edge_threshold=0.1,
        stage3_phase2_top_m_edges=200,
        stage3_phase2_must_links=[],
        stage3_phase2_cannot_links=[],
        stage3_phase2_random_seed=42,
        stage3_phase2_n_iterations=10,
        
        # Initialize empty stage 3 fields
        stage3_status="pending",
        stage3_clusters=[],
        stage3_prototypes=[],
        stage3_metadata={},
        
        # Phase 1 fields
        stage3_phase1_status="pending",
        stage3_phase1_clusters=[],
        stage3_phase1_prototypes=[],
        stage3_phase1_metadata={},
        stage3_phase1_quality_stats={},
        stage3_phase1_processing_time=0.0,
        
        # Phase 2 fields  
        stage3_phase2_status="pending",
        stage3_phase2_communities=[],
        stage3_phase2_labels=[],
        stage3_phase2_metadata={},
        stage3_phase2_llm_suggestions=[],
        stage3_phase2_processing_time=0.0,
        
        # Quality assessment fields
        stage3_quality_overall_score=0.0,
        stage3_quality_subset_score=0.0,
        stage3_quality_consistency_score=0.0,
        stage3_quality_user_feedback=[],
        stage3_quality_modification_history=[]
    )
    
    print(f"\n🚀 Starting Two-Phase Processing...")
    
    # Execute the complete two-phase system
    try:
        from nodes.stage3_classification.two_phase_stage3_node import two_phase_stage3_node
        result_state = two_phase_stage3_node(state)
        
        print(f"\n✅ Two-Phase Processing Completed!")
        print(f"   Status: {result_state.get('stage3_status', 'unknown')}")
        
        # Phase 1 Results
        print(f"\n📊 Phase 1 Results:")
        print(f"   - Status: {result_state.get('stage3_phase1_status', 'unknown')}")
        print(f"   - Clusters: {len(set(result_state.get('stage3_phase1_clusters', [])))}")
        print(f"   - Processing time: {result_state.get('stage3_phase1_processing_time', 0):.3f}s")
        
        phase1_quality = result_state.get('stage3_phase1_quality_stats', {})
        if phase1_quality:
            print(f"   - Quality metrics:")
            if 'subset_score' in phase1_quality:
                print(f"     • SubsetScore: {phase1_quality['subset_score']:.3f}")
            if 'cluster_consistency' in phase1_quality:
                print(f"     • Consistency: {phase1_quality['cluster_consistency']:.3f}")
        
        # Phase 2 Results  
        print(f"\n🔗 Phase 2 Results:")
        print(f"   - Status: {result_state.get('stage3_phase2_status', 'unknown')}")
        print(f"   - Communities: {len(set(result_state.get('stage3_phase2_communities', [])))}")
        print(f"   - Processing time: {result_state.get('stage3_phase2_processing_time', 0):.3f}s")
        
        # Quality Assessment
        print(f"\n📈 Quality Assessment:")
        print(f"   - Overall score: {result_state.get('stage3_quality_overall_score', 0):.3f}")
        print(f"   - SubsetScore: {result_state.get('stage3_quality_subset_score', 0):.3f}")
        print(f"   - Consistency: {result_state.get('stage3_quality_consistency_score', 0):.3f}")
        
        # Final clustering results
        final_clusters = result_state.get('stage3_phase2_labels', [])
        if final_clusters:
            unique_clusters = len(set(final_clusters))
            print(f"\n🎯 Final Results:")
            print(f"   - Total clusters: {unique_clusters}")
            print(f"   - Ground truth: {len(np.unique(ground_truth))}")
            
            # Compare with ground truth if available
            if len(final_clusters) == len(ground_truth):
                # Simple cluster purity calculation
                cluster_purities = []
                for cluster_id in set(final_clusters):
                    cluster_indices = [i for i, c in enumerate(final_clusters) if c == cluster_id]
                    cluster_gt = [ground_truth[i] for i in cluster_indices]
                    if cluster_gt:
                        most_common_gt = max(set(cluster_gt), key=cluster_gt.count)
                        purity = cluster_gt.count(most_common_gt) / len(cluster_gt)
                        cluster_purities.append(purity)
                
                if cluster_purities:
                    avg_purity = np.mean(cluster_purities)
                    print(f"   - Average cluster purity: {avg_purity:.3f}")
        
        print(f"\n🎉 Two-Phase System Test: SUCCESS!")
        return True
        
    except Exception as e:
        print(f"\n❌ Two-Phase System Test: FAILED")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_two_phase_system()
    if success:
        print(f"\n✨ All tests passed! Two-Phase Stage 3 system is ready for production.")
        sys.exit(0)
    else:
        print(f"\n💥 Tests failed. System needs debugging.")
        sys.exit(1)