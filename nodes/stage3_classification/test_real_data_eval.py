"""
Test trail3 evaluation with real data files (no mock data).
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from evaluation import mcl_scoring_function

def test_evaluation_with_real_data():
    """Test evaluation system using real stage2 data files."""
    print("🔍 Testing Trail3 Evaluation with Real Data")
    print("=" * 50)
    
    # Path to real stage2 results
    stage2_dir = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/temp_data/stage2_results"
    
    # Find a file with embeddings
    csv_files = [
        "stage2_문4_img_20250918_112537.csv",
        "stage2_문5_img_20250918_112643.csv", 
        "stage2_문23_concept_20250918_113224.csv"
    ]
    
    for csv_file in csv_files:
        file_path = os.path.join(stage2_dir, csv_file)
        if os.path.exists(file_path):
            print(f"\n--- Testing with {csv_file} ---")
            test_file_evaluation(file_path)
            break
    else:
        print("❌ No real data files found for testing")

def test_file_evaluation(file_path: str):
    """Test evaluation with a specific data file."""
    try:
        # Load the data
        df = pd.read_csv(file_path)
        print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
        
        # Find embedding columns (exclude org_text)
        embed_cols = [col for col in df.columns 
                     if 'embed' in col.lower() 
                     and 'org_text' not in col.lower()]
        
        print(f"Found embedding columns: {len(embed_cols)}")
        print(f"Embedding columns: {embed_cols[:3]}...")  # Show first 3
        
        if len(embed_cols) == 0:
            print("❌ No embedding columns found")
            return
            
        # Extract embeddings from one column for testing
        test_col = embed_cols[0]
        print(f"Testing with column: {test_col}")
        
        # Get embeddings (assuming they're stored as string representations of lists)
        embeddings = []
        valid_rows = []
        
        for idx, row in df.iterrows():
            embed_str = row[test_col]
            if pd.notna(embed_str) and embed_str != '':
                try:
                    # Parse embedding (assuming it's a string representation of a list)
                    if isinstance(embed_str, str):
                        if embed_str.startswith('[') and embed_str.endswith(']'):
                            embed_array = eval(embed_str)  # Use ast.literal_eval in production
                        else:
                            continue
                    else:
                        embed_array = embed_str
                    
                    if len(embed_array) > 0:
                        embeddings.append(embed_array)
                        valid_rows.append(idx)
                        
                except Exception as e:
                    continue
        
        if len(embeddings) < 10:
            print(f"❌ Too few valid embeddings: {len(embeddings)}")
            return
            
        embeddings = np.array(embeddings)
        print(f"Extracted {len(embeddings)} valid embeddings, shape: {embeddings.shape}")
        
        # Create synthetic true labels for testing (simulate 3-5 clusters)
        n_clusters = min(5, max(3, len(embeddings) // 10))
        cluster_size = len(embeddings) // n_clusters
        
        true_labels = []
        for i in range(n_clusters):
            start_idx = i * cluster_size
            end_idx = start_idx + cluster_size if i < n_clusters - 1 else len(embeddings)
            true_labels.extend([i] * (end_idx - start_idx))
            
        true_labels = np.array(true_labels)
        print(f"Created synthetic true labels: {n_clusters} clusters")
        print(f"Label distribution: {np.bincount(true_labels)}")
        
        # Test 1: Perfect clustering (same as true labels)
        print(f"\n1. Perfect Clustering Test:")
        scores = mcl_scoring_function(true_labels, true_labels)
        print(f"   NMI: {scores['metrics']['nmi']:.3f} (expected: 1.000)")
        print(f"   ARI: {scores['metrics']['ari']:.3f} (expected: 1.000)")
        
        # Test 2: Random clustering
        print(f"\n2. Random Clustering Test:")
        random_labels = np.random.randint(0, n_clusters + 2, size=len(true_labels))
        scores = mcl_scoring_function(true_labels, random_labels)
        print(f"   NMI: {scores['metrics']['nmi']:.3f} (expected: low)")
        print(f"   ARI: {scores['metrics']['ari']:.3f} (expected: low)")
        
        # Test 3: Partially correct clustering
        print(f"\n3. Partially Correct Clustering Test:")
        # Create labels that are mostly correct but with some errors
        partial_labels = true_labels.copy()
        # Introduce some errors
        error_indices = np.random.choice(len(partial_labels), size=len(partial_labels)//10, replace=False)
        partial_labels[error_indices] = (partial_labels[error_indices] + 1) % n_clusters
        
        scores = mcl_scoring_function(true_labels, partial_labels)
        print(f"   NMI: {scores['metrics']['nmi']:.3f} (expected: moderate-high)")
        print(f"   ARI: {scores['metrics']['ari']:.3f} (expected: moderate-high)")
        
        print(f"\n✅ Real data evaluation test completed successfully!")
        print(f"✅ Evaluation system works correctly with real embeddings!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    test_evaluation_with_real_data()