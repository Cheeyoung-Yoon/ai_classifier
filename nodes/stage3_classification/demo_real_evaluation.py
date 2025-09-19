"""
Demonstrate trail3 evaluation capabilities using real data structure.
Shows how evaluation integrates with auto_train mode.
"""
import os
import pandas as pd
import numpy as np
from mcl_pipeline import auto_train_mcl
from evaluation import mcl_scoring_function

def demonstrate_trail3_evaluation():
    """Demonstrate evaluation with real data structure but simplified test."""
    print("🎯 Trail3 Evaluation Demonstration")
    print("=" * 45)
    
    # Use real stage2 file path
    stage2_file = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/temp_data/stage2_results/stage2_문4_img_20250918_112537.csv"
    
    if not os.path.exists(stage2_file):
        print(f"❌ Stage2 file not found: {stage2_file}")
        return
        
    print(f"Using real data file: {os.path.basename(stage2_file)}")
    
    # Load and prepare data
    df = pd.read_csv(stage2_file)
    
    # Extract embeddings from first embed column
    embed_col = 'embed_text_1'
    embeddings = []
    
    for idx, row in df.iterrows():
        embed_str = row[embed_col]
        if pd.notna(embed_str) and embed_str != '':
            try:
                if isinstance(embed_str, str) and embed_str.startswith('['):
                    embed_array = eval(embed_str)  # In production, use ast.literal_eval
                    embeddings.append(embed_array)
                    if len(embeddings) >= 100:  # Limit for demo
                        break
            except:
                continue
                
    if len(embeddings) < 20:
        print(f"❌ Not enough embeddings extracted: {len(embeddings)}")
        return
        
    embeddings = np.array(embeddings)
    print(f"Extracted {len(embeddings)} embeddings, shape: {embeddings.shape}")
    
    # Create synthetic ground truth for demonstration
    n_samples = len(embeddings)
    n_true_clusters = 4
    cluster_size = n_samples // n_true_clusters
    
    true_labels = []
    for i in range(n_true_clusters):
        start = i * cluster_size
        end = start + cluster_size if i < n_true_clusters - 1 else n_samples
        true_labels.extend([i] * (end - start))
    
    true_labels = np.array(true_labels)
    print(f"Created ground truth: {n_true_clusters} clusters, distribution: {np.bincount(true_labels)}")
    
    # Demonstration 1: Direct evaluation function
    print(f"\n--- Direct Evaluation Function Demo ---")
    
    # Test with different clustering scenarios
    test_scenarios = {
        "Perfect": true_labels,
        "Good (90% correct)": create_noisy_labels(true_labels, error_rate=0.1),
        "Poor (random)": np.random.randint(0, 6, size=len(true_labels)),
    }
    
    for name, pred_labels in test_scenarios.items():
        scores = mcl_scoring_function(true_labels, pred_labels)
        print(f"{name:15} NMI: {scores['metrics']['nmi']:.3f}, ARI: {scores['metrics']['ari']:.3f}")
    
    # Demonstration 2: MCL Auto-train with evaluation
    print(f"\n--- MCL Auto-train with Evaluation Demo ---")
    
    try:
        # Run auto-train with true labels for evaluation
        result = auto_train_mcl(
            embeddings[:50],  # Use subset for faster demo
            search_iterations=6,
            true_labels=true_labels[:50]
        )
        
        print(f"Auto-train completed:")
        print(f"  Best parameters: {result.get('best_parameters', {})}")
        print(f"  Best composite score: {result.get('best_score', 0):.3f}")
        
        if 'best_evaluation' in result:
            eval_results = result['best_evaluation']
            if 'metrics' in eval_results:
                metrics = eval_results['metrics']
                print(f"  Best NMI: {metrics.get('nmi', 0):.3f}")
                print(f"  Best ARI: {metrics.get('ari', 0):.3f}")
            
            if 'cluster_stats' in eval_results:
                stats = eval_results['cluster_stats']
                print(f"  Singleton ratio: {stats.get('pred_singleton_ratio', 0):.3f}")
        
        print(f"  Tested {result.get('search_iterations', 0)} parameter combinations")
        
    except Exception as e:
        print(f"❌ Auto-train demo failed: {e}")
    
    # Demonstration 3: Show how to interpret results
    print(f"\n--- How to Interpret Evaluation Results ---")
    print(f"📊 NMI (Normalized Mutual Information):")
    print(f"   • 1.0 = Perfect clustering (identical to ground truth)")
    print(f"   • 0.8+ = Very good clustering") 
    print(f"   • 0.5-0.8 = Moderate clustering")
    print(f"   • <0.5 = Poor clustering")
    
    print(f"\n📊 ARI (Adjusted Rand Index):")
    print(f"   • 1.0 = Perfect clustering") 
    print(f"   • 0.7+ = Very good clustering")
    print(f"   • 0.3-0.7 = Moderate clustering")
    print(f"   • <0.3 = Poor clustering")
    
    print(f"\n📊 Composite Score (used for parameter optimization):")
    print(f"   • (NMI + ARI) / 2")
    print(f"   • Auto-train selects parameters that maximize this score")
    
    print(f"\n✅ Trail3 evaluation demonstration completed!")
    print(f"✅ Ready for production use with real LangGraph pipeline!")

def create_noisy_labels(true_labels, error_rate=0.1):
    """Create labels with some errors for testing."""
    noisy = true_labels.copy()
    n_errors = int(len(noisy) * error_rate)
    error_indices = np.random.choice(len(noisy), size=n_errors, replace=False)
    
    for idx in error_indices:
        # Change to a different random label
        original = noisy[idx]
        new_label = np.random.choice([l for l in np.unique(true_labels) if l != original])
        noisy[idx] = new_label
    
    return noisy

if __name__ == "__main__":
    np.random.seed(42)  # For reproducible demo
    demonstrate_trail3_evaluation()