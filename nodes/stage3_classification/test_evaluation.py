"""
Test script for trail3 evaluation capabilities.
Demonstrates NMI/ARI scoring with MCL clustering.
"""
import numpy as np
from evaluation import mcl_scoring_function
from mcl_pipeline import auto_train_mcl

def test_evaluation_with_synthetic_data():
    """Test evaluation with synthetic clustered data."""
    np.random.seed(42)
    
    # Create synthetic data with known clusters
    n_samples_per_cluster = 20
    n_clusters = 4
    n_features = 128
    
    embeddings = []
    true_labels = []
    
    for cluster_id in range(n_clusters):
        # Create cluster center
        center = np.random.randn(n_features) * 2
        
        # Generate points around center
        cluster_points = []
        for _ in range(n_samples_per_cluster):
            point = center + np.random.randn(n_features) * 0.5
            cluster_points.append(point)
            true_labels.append(cluster_id)
        
        embeddings.extend(cluster_points)
    
    embeddings = np.array(embeddings)
    true_labels = np.array(true_labels)
    
    print(f"Created synthetic data: {embeddings.shape[0]} samples, {n_clusters} true clusters")
    
    # Test direct evaluation function
    print("\n=== Testing Direct Evaluation Function ===")
    # Create some predicted labels for testing
    predicted_labels = np.random.randint(0, n_clusters + 1, size=len(true_labels))
    
    try:
        scores = mcl_scoring_function(true_labels, predicted_labels)
        print(f"Direct evaluation scores:")
        for key, value in scores.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Direct evaluation failed: {e}")
    
    # Test MCL auto-training with evaluation
    print("\n=== Testing MCL Auto-Training with Evaluation ===")
    try:
        result = auto_train_mcl(embeddings, search_iterations=8, true_labels=true_labels)
        
        print(f"Auto-training completed:")
        print(f"  Best parameters: {result.get('best_parameters', {})}")
        print(f"  Best score: {result.get('best_score', 0):.3f}")
        
        if 'best_evaluation' in result:
            eval_results = result['best_evaluation']
            print(f"  Best evaluation:")
            for key, value in eval_results.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value}")
        
        print(f"  Search iterations: {result.get('search_iterations', 0)}")
        
    except Exception as e:
        print(f"Auto-training with evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def test_edge_cases():
    """Test evaluation with edge cases."""
    print("\n=== Testing Edge Cases ===")
    
    # Test with singleton clusters
    true_labels = np.array([0, 1, 2, 3, 4])  # All singletons
    pred_labels = np.array([0, 0, 1, 1, 2])  # Some grouping
    
    try:
        scores = mcl_scoring_function(true_labels, pred_labels)
        print(f"Singleton test scores:")
        for key, value in scores.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Singleton test failed: {e}")
    
    # Test with perfect clustering
    true_labels = np.array([0, 0, 1, 1, 2, 2])
    pred_labels = np.array([0, 0, 1, 1, 2, 2])  # Perfect match
    
    try:
        scores = mcl_scoring_function(true_labels, pred_labels)
        print(f"Perfect clustering scores:")
        for key, value in scores.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Perfect clustering test failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing Trail3 Evaluation Capabilities")
    print("=" * 50)
    
    test_evaluation_with_synthetic_data()
    test_edge_cases()
    
    print("\n✅ Evaluation testing completed!")