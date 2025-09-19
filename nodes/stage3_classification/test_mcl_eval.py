"""
Test MCL with evaluation using better parameters for synthetic data.
"""
import numpy as np
from evaluation import mcl_scoring_function
from mcl_pipeline import MCLPipeline, auto_train_mcl

def test_mcl_with_realistic_data():
    """Test MCL with synthetic data that should cluster well."""
    print("🧪 Testing MCL with Realistic Clustering Data")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Create well-separated clusters
    n_samples_per_cluster = 15
    n_clusters = 3
    n_features = 50
    
    embeddings = []
    true_labels = []
    
    centers = [
        np.array([2, 2] + [0] * (n_features-2)),
        np.array([-2, -2] + [0] * (n_features-2)),
        np.array([2, -2] + [0] * (n_features-2))
    ]
    
    for cluster_id, center in enumerate(centers):
        for _ in range(n_samples_per_cluster):
            # Add small noise around center
            point = center + np.random.randn(n_features) * 0.3
            embeddings.append(point)
            true_labels.append(cluster_id)
    
    embeddings = np.array(embeddings)
    true_labels = np.array(true_labels)
    
    print(f"Created {len(embeddings)} samples with {n_clusters} well-separated clusters")
    
    # Test 1: Manual MCL with reasonable parameters
    print("\n1. Manual MCL Test:")
    try:
        mcl = MCLPipeline(inflation=2.0, max_iters=100)
        mcl.fit(embeddings, k=10)  # Smaller k for better clustering
        
        scores = mcl_scoring_function(true_labels, mcl.cluster_labels)
        summary = mcl.get_cluster_summary()
        
        print(f"MCL found {summary['n_clusters']} clusters")
        print(f"NMI: {scores['metrics']['nmi']:.3f}")
        print(f"ARI: {scores['metrics']['ari']:.3f}")
        print(f"Singleton ratio: {scores['cluster_stats']['pred_singleton_ratio']:.3f}")
        
    except Exception as e:
        print(f"Manual MCL failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Auto-train with limited search space
    print("\n2. Auto-train MCL Test (limited search):")
    try:
        result = auto_train_mcl(
            embeddings, 
            search_iterations=6, 
            true_labels=true_labels
        )
        
        print(f"Best parameters: {result.get('best_parameters', {})}")
        print(f"Best score: {result.get('best_score', 0):.3f}")
        
        if 'best_evaluation' in result:
            eval_results = result['best_evaluation']
            if 'metrics' in eval_results:
                metrics = eval_results['metrics']
                print(f"Best NMI: {metrics.get('nmi', 0):.3f}")
                print(f"Best ARI: {metrics.get('ari', 0):.3f}")
        
    except Exception as e:
        print(f"Auto-train MCL failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mcl_with_realistic_data()
    print("\n✅ MCL evaluation testing completed!")