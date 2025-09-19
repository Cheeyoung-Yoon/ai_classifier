"""
Simple test to debug Phase 1 issues
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/cyyoon/test_area/ai_text_classification/2.langgraph')

from nodes.stage3_classification.phase1_primary_labeling import Phase1PrimaryLabeling

def simple_phase1_test():
    """Simple Phase 1 test with minimal data."""
    print("Testing Phase 1 with larger data...")
    
    # Create test data similar to the main test but smaller
    np.random.seed(42)
    n_samples = 30  # Smaller than 75
    n_features = 64  # Smaller than 128
    
    # Create 3 clusters
    embeddings = []
    true_labels = []
    
    # Cluster 1
    center1 = np.random.randn(n_features)
    center1 = center1 / np.linalg.norm(center1)
    for i in range(10):
        emb = center1 + np.random.normal(0, 0.1, n_features)
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
        true_labels.append(0)
    
    # Cluster 2
    center2 = np.random.randn(n_features)
    center2 = center2 / np.linalg.norm(center2)
    for i in range(10):
        emb = center2 + np.random.normal(0, 0.1, n_features)
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
        true_labels.append(1)
    
    # Cluster 3
    center3 = np.random.randn(n_features)
    center3 = center3 / np.linalg.norm(center3)
    for i in range(10):
        emb = center3 + np.random.normal(0, 0.1, n_features)
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
        true_labels.append(2)
    
    embeddings = np.array(embeddings)
    true_labels = np.array(true_labels)
    texts = [f"Sample text {i}" for i in range(n_samples)]
    
    print(f"Created {len(embeddings)} samples with {embeddings.shape[1]} features")
    
    # Use similar config to the main test
    config = {
        "knn_k": 15,  # Reasonable k
        "knn_metric": "cosine",
        "mutual_knn": True,
        "csls_neighborhood_size": 5,
        "csls_threshold": 0.1,
        "top_m_edges": 10,
        "prune_bottom_percentile": 30,
        "mcl_inflation": 2.0,
        "mcl_expansion": 2,
        "mcl_max_iters": 100,
        "allow_singletons": True,
        "merge_small_clusters": True,
        "min_cluster_size": 2,
        "small_cluster_threshold": 3,
        "compute_subset_score": True,
        "compute_cluster_quality": True
    }
    
    try:
        phase1 = Phase1PrimaryLabeling(config)
        result = phase1.process_embeddings(embeddings, texts, true_labels)
        
        print(f"Result status: {result['status']}")
        
        if result["status"] == "completed":
            print(f"✅ Success!")
            print(f"   Clusters: {result['n_clusters']}")
            print(f"   Singletons: {result['n_singletons']}")
            print(f"   Time: {result['processing_time']:.2f}s")
            
            # Show quality stats
            if "quality_stats" in result:
                quality = result["quality_stats"]
                if "subset_score" in quality:
                    print(f"   SubsetScore: {quality['subset_score']['score']:.3f}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_phase1_test()