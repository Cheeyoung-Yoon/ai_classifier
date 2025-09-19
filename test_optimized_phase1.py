#!/usr/bin/env python3
"""
Optimized Phase 1 test with better parameters for larger datasets.
"""

import sys
import os
sys.path.append('/home/cyyoon/test_area/ai_text_classification/2.langgraph')

import numpy as np
from nodes.stage3_classification.phase1_primary_labeling import Phase1PrimaryLabeling

def test_optimized_phase1():
    """Test Phase 1 with optimized parameters for larger datasets."""
    
    # Create larger test dataset (60 samples, 64 features)
    np.random.seed(42)
    n_samples = 60
    n_features = 64
    
    # Generate embeddings with clear cluster structure
    cluster_centers = np.random.randn(6, n_features)  # 6 clusters
    embeddings = []
    ground_truth = []
    
    samples_per_cluster = n_samples // 6
    for i, center in enumerate(cluster_centers):
        for j in range(samples_per_cluster):
            # Add noise to cluster center
            noise = np.random.normal(0, 0.1, n_features)
            embedding = center + noise
            # L2 normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
            ground_truth.append(i)
    
    # Handle remaining samples
    remaining = n_samples - len(embeddings)
    for i in range(remaining):
        center = cluster_centers[i % len(cluster_centers)]
        noise = np.random.normal(0, 0.1, n_features)
        embedding = center + noise
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
        ground_truth.append(i % len(cluster_centers))
    
    embeddings = np.array(embeddings)
    ground_truth = np.array(ground_truth)
    
    print(f"Generated dataset: {embeddings.shape[0]} samples, {embeddings.shape[1]} features")
    print(f"Ground truth clusters: {len(np.unique(ground_truth))}")
    
    # Optimized configuration for larger datasets
    optimized_config = {
        # kNN parameters - reduced k for stability
        "knn_k": 15,  # Reduced from 30 for better stability
        "knn_metric": "cosine",
        "mutual_knn": True,
        
        # CSLS parameters
        "csls_k": 10,  # Reduced for stability
        "csls_neighborhood_size": 10,  # Added missing parameter
        "csls_weighting": True,
        "top_m_edges": 500,  # Maximum edges to keep
        "csls_threshold": 0.1,  # Minimum CSLS score
        "prune_bottom_percentile": 10,  # Prune bottom 10% edges
        
        # Edge processing
        "edge_pruning_threshold": 0.1,  # Increased threshold
        "edge_scaling_factor": 1.5,
        "min_edge_weight": 0.01,
        
        # MCL parameters - more conservative settings
        "mcl_inflation": 1.8,  # Reduced inflation for larger clusters
        "mcl_expansion": 2,
        "mcl_pruning": 1e-2,  # Less aggressive pruning
        "mcl_max_iters": 50,   # Reduced iterations
        "mcl_convergence_check": True,
        
        # Singleton and small cluster handling
        "allow_singletons": True,
        "merge_small_clusters": True,
        "min_cluster_size": 2,
        "small_cluster_threshold": 3,
        
        # Quality assessment
        "compute_subset_score": True,
        "compute_cluster_quality": True
    }
    
    # Initialize Phase 1 with optimized config
    phase1 = Phase1PrimaryLabeling(config=optimized_config)
    
    # Process embeddings
    print("\n🚀 Starting Phase 1 processing with optimized parameters...")
    results = phase1.process_embeddings(
        embeddings=embeddings,
        texts=[f"sample_{i}" for i in range(len(embeddings))],
        ground_truth_labels=ground_truth.tolist()
    )
    
    # Display results
    if results["status"] == "completed":
        print(f"\n✅ Phase 1 completed successfully!")
        print(f"   📊 Clusters found: {results['n_clusters']}")
        print(f"   🔍 Singletons: {results['n_singletons']}")
        print(f"   ⏱️  Processing time: {results['processing_time']:.2f}s")
        
        # Quality metrics
        if "quality_stats" in results:
            quality = results["quality_stats"]
            print(f"   📈 Quality metrics:")
            if "subset_score" in quality:
                print(f"      - SubsetScore: {quality['subset_score']:.3f}")
            if "cluster_consistency" in quality:
                print(f"      - Cluster consistency: {quality['cluster_consistency']:.3f}")
        
        # Cluster distribution
        labels = np.array(results["cluster_labels"])
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        print(f"   📋 Cluster sizes: {dict(zip(unique_labels, counts))}")
        
        return True
        
    else:
        print(f"\n❌ Phase 1 failed: {results.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    success = test_optimized_phase1()
    if success:
        print("\n🎉 Optimized Phase 1 test passed!")
    else:
        print("\n💥 Optimized Phase 1 test failed!")
        sys.exit(1)