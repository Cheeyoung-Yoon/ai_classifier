"""
Quick test of the evaluation with a simple manual clustering to verify metrics.
"""
import numpy as np
from evaluation import mcl_scoring_function, ClusterEvaluator

def test_simple_evaluation():
    """Test evaluation with simple, controlled examples."""
    print("🧪 Testing Simple Evaluation Examples")
    print("=" * 50)
    
    # Test 1: Perfect clustering
    print("\n1. Perfect Clustering Test:")
    true_labels = np.array([0, 0, 1, 1, 2, 2])
    pred_labels = np.array([0, 0, 1, 1, 2, 2])
    
    scores = mcl_scoring_function(true_labels, pred_labels)
    print(f"True labels:  {true_labels}")
    print(f"Pred labels:  {pred_labels}")
    print(f"NMI: {scores['metrics']['nmi']:.3f}")
    print(f"ARI: {scores['metrics']['ari']:.3f}")
    print(f"Expected: NMI=1.0, ARI=1.0")
    
    # Test 2: Completely wrong clustering
    print("\n2. Random Clustering Test:")
    true_labels = np.array([0, 0, 1, 1, 2, 2])
    pred_labels = np.array([3, 4, 5, 6, 7, 8])  # All different clusters
    
    scores = mcl_scoring_function(true_labels, pred_labels)
    print(f"True labels:  {true_labels}")
    print(f"Pred labels:  {pred_labels}")
    print(f"NMI: {scores['metrics']['nmi']:.3f}")
    print(f"ARI: {scores['metrics']['ari']:.3f}")
    print(f"Expected: Low scores")
    
    # Test 3: Partial match
    print("\n3. Partial Match Test:")
    true_labels = np.array([0, 0, 0, 1, 1, 1])
    pred_labels = np.array([2, 2, 3, 3, 3, 3])  # Some overlap
    
    scores = mcl_scoring_function(true_labels, pred_labels)
    print(f"True labels:  {true_labels}")
    print(f"Pred labels:  {pred_labels}")
    print(f"NMI: {scores['metrics']['nmi']:.3f}")
    print(f"ARI: {scores['metrics']['ari']:.3f}")
    print(f"Expected: Moderate scores")
    
    # Test 4: Test ClusterEvaluator directly
    print("\n4. ClusterEvaluator Class Test:")
    evaluator = ClusterEvaluator(true_labels)
    result = evaluator.evaluate_clustering(pred_labels)
    
    print(f"Evaluator result keys: {list(result.keys())}")
    print(f"Has ground truth: {result['has_ground_truth']}")
    print(f"Metrics: {result['metrics']}")

if __name__ == "__main__":
    test_simple_evaluation()
    print("\n✅ Simple evaluation tests completed!")