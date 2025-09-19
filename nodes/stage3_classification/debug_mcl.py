"""
Debug MCL clustering with very simple data.
"""
import numpy as np
from mcl_pipeline import MCLPipeline
from evaluation import mcl_scoring_function

def debug_mcl():
    """Debug MCL with extremely simple data."""
    print("🔍 Debugging MCL Clustering")
    print("=" * 40)
    
    # Create 2 very distinct clusters
    cluster1 = np.array([[0, 0], [0.1, 0.1], [0.1, 0]])
    cluster2 = np.array([[10, 10], [10.1, 10.1], [10, 10.1]])
    
    embeddings = np.vstack([cluster1, cluster2])
    true_labels = np.array([0, 0, 0, 1, 1, 1])
    
    print(f"Data shape: {embeddings.shape}")
    print(f"Data:\n{embeddings}")
    print(f"True labels: {true_labels}")
    
    # Test with different parameters
    for inflation in [1.5, 2.0, 3.0]:
        for k in [2, 3, 5]:
            print(f"\nTesting inflation={inflation}, k={k}:")
            
            try:
                mcl = MCLPipeline(inflation=inflation, max_iters=50)
                mcl.fit(embeddings, k=k)
                
                print(f"  Found {len(np.unique(mcl.cluster_labels))} clusters")
                print(f"  Labels: {mcl.cluster_labels}")
                
                # Evaluate
                scores = mcl_scoring_function(true_labels, mcl.cluster_labels)
                print(f"  NMI: {scores['metrics']['nmi']:.3f}")
                print(f"  ARI: {scores['metrics']['ari']:.3f}")
                
            except Exception as e:
                print(f"  Failed: {e}")

if __name__ == "__main__":
    debug_mcl()