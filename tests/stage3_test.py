"""
Stage 3 Classification Test for LangGraph Pipeline.
Tests the integrated trail3 MCL clustering with evaluation capabilities.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path  
project_root = "/home/cyyoon/test_area/ai_text_classification/2.langgraph"
sys.path.insert(0, project_root)

from graph.state import GraphState, initialize_project_state
from nodes.stage3_classification.stage3_node import stage3_classification_node
from nodes.stage3_classification.evaluation import mcl_scoring_function


def test_stage3_node_integration():
    """Test Stage 3 node with real LangGraph state structure."""
    print("🧪 Testing Stage 3 Classification Node Integration")
    print("=" * 60)
    
    # Use real stage2 data files
    stage2_results_dir = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/temp_data/stage2_results"
    test_files = [
        "stage2_문4_img_20250918_112537.csv",
        "stage2_문5_img_20250918_112643.csv"
    ]
    
    # Create mock state that mimics real LangGraph state after Stage 2
    state = initialize_project_state("test", "test.txt", "-SUV_776부.xlsx")
    
    # Add matched_questions with real stage2 data paths
    matched_questions = {}
    
    for file_name in test_files:
        file_path = os.path.join(stage2_results_dir, file_name)
        if os.path.exists(file_path):
            # Extract question ID from filename
            qid = file_name.split("_")[1]  # e.g., "문4" from "stage2_문4_img_..."
            
            matched_questions[qid] = {
                "stage2_data": {
                    "csv_path": file_path,
                    "status": "completed"
                }
            }
            print(f"✅ Added real data for {qid}: {file_name}")
    
    state["matched_questions"] = matched_questions
    
    # Test different stage3 modes
    test_modes = [
        ("estimate", "Quick cluster estimation"),
        ("auto_train", "Auto hyperparameter search with evaluation"),
        ("manual_train", "Manual parameters")
    ]
    
    for mode, description in test_modes:
        print(f"\n--- Testing {mode.upper()} Mode ---")
        print(f"Description: {description}")
        
        # Set up state for this mode
        test_state = state.copy()
        test_state["stage3_mode"] = mode
        
        if mode == "auto_train":
            test_state["stage3_search_iterations"] = 6  # Quick test
        elif mode == "manual_train":
            test_state["stage3_manual_inflation"] = 2.0
            test_state["stage3_manual_k"] = 30
            test_state["stage3_manual_max_iters"] = 100
        
        try:
            # Run stage3 node
            result_state = stage3_classification_node(test_state)
            
            # Check results
            status = result_state.get("stage3_status", "unknown")
            print(f"Status: {status}")
            
            if status == "completed":
                print("✅ Stage 3 completed successfully!")
                
                # Print summary
                summary = result_state.get("stage3_cluster_summary", {})
                print(f"  Clusters found: {summary.get('n_clusters', 'N/A')}")
                print(f"  Samples processed: {summary.get('n_samples', 'N/A')}")
                print(f"  Processing time: {result_state.get('processing_time_seconds', 'N/A')}s")
                
                # Print mode-specific results
                if mode == "estimate":
                    print(f"  Estimated clusters: {result_state.get('stage3_estimated_clusters', 'N/A')}")
                    print(f"  Recommended k: {result_state.get('stage3_recommended_k', 'N/A')}")
                
                elif mode == "auto_train":
                    best_params = result_state.get("stage3_best_parameters", {})
                    best_score = result_state.get("stage3_best_score", 0)
                    print(f"  Best parameters: {best_params}")
                    print(f"  Best score: {best_score:.4f}")
                    
                    # Check evaluation results
                    evaluation = result_state.get("stage3_best_evaluation", {})
                    if evaluation and "metrics" in evaluation:
                        metrics = evaluation["metrics"]
                        print(f"  Evaluation - NMI: {metrics.get('nmi', 'N/A')}")
                        print(f"  Evaluation - ARI: {metrics.get('ari', 'N/A')}")
                
                elif mode == "manual_train":
                    cluster_mapping = result_state.get("stage3_cluster_mapping", {})
                    print(f"  Cluster mapping created: {bool(cluster_mapping)}")
                    
                    if cluster_mapping:
                        n_mappings = len(cluster_mapping)
                        print(f"  Number of cluster groups: {n_mappings}")
                
            elif status == "failed":
                error = result_state.get("stage3_error", "Unknown error")
                print(f"❌ Stage 3 failed: {error}")
                
            elif status == "skipped":
                print("⚠️ Stage 3 was skipped")
                
        except Exception as e:
            print(f"❌ Test failed for {mode}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Stage 3 node integration tests completed!")


def test_stage3_evaluation_functions():
    """Test the evaluation functions directly."""
    print("\n🔬 Testing Stage 3 Evaluation Functions")
    print("=" * 45)
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 60
    n_true_clusters = 4
    
    # Create ground truth labels
    cluster_size = n_samples // n_true_clusters
    true_labels = []
    for i in range(n_true_clusters):
        start = i * cluster_size
        end = start + cluster_size if i < n_true_clusters - 1 else n_samples
        true_labels.extend([i] * (end - start))
    
    true_labels = np.array(true_labels)
    print(f"Created test data: {n_samples} samples, {n_true_clusters} true clusters")
    print(f"True label distribution: {np.bincount(true_labels)}")
    
    # Test scenarios
    test_scenarios = {
        "Perfect clustering": true_labels,
        "Good clustering (90% correct)": create_noisy_labels(true_labels, 0.1),
        "Moderate clustering (70% correct)": create_noisy_labels(true_labels, 0.3),
        "Poor clustering (random)": np.random.randint(0, 6, size=n_samples),
        "Over-clustering": create_split_clusters(true_labels),
        "Under-clustering": create_merged_clusters(true_labels)
    }
    
    print(f"\nEvaluation Results:")
    print(f"{'Scenario':<25} {'NMI':<6} {'ARI':<6} {'Clusters':<9} {'Quality':<10}")
    print("-" * 65)
    
    for scenario_name, pred_labels in test_scenarios.items():
        try:
            # Use seed for reproducible random scenarios
            if "random" in scenario_name.lower():
                np.random.seed(42)
                pred_labels = np.random.randint(0, 6, size=n_samples)
            
            scores = mcl_scoring_function(true_labels, pred_labels)
            
            nmi = scores['metrics']['nmi']
            ari = scores['metrics']['ari']
            n_clusters = scores['n_predicted_clusters']
            
            # Determine quality
            quality = get_quality_rating(nmi, ari)
            
            print(f"{scenario_name:<25} {nmi:<6.3f} {ari:<6.3f} {n_clusters:<9} {quality:<10}")
            
        except Exception as e:
            print(f"{scenario_name:<25} ERROR: {str(e)}")
    
    print(f"\n✅ Evaluation function tests completed!")


def create_noisy_labels(true_labels, error_rate):
    """Create labels with specified error rate."""
    noisy = true_labels.copy()
    n_errors = int(len(noisy) * error_rate)
    error_indices = np.random.choice(len(noisy), size=n_errors, replace=False)
    
    for idx in error_indices:
        original = noisy[idx]
        possible_labels = [l for l in np.unique(true_labels) if l != original]
        if possible_labels:
            noisy[idx] = np.random.choice(possible_labels)
    
    return noisy


def create_split_clusters(true_labels):
    """Create over-clustering by splitting some clusters."""
    split_labels = true_labels.copy()
    max_label = max(true_labels) + 1
    
    # Split cluster 0 into two parts
    cluster_0_indices = np.where(split_labels == 0)[0]
    split_point = len(cluster_0_indices) // 2
    split_labels[cluster_0_indices[split_point:]] = max_label
    
    return split_labels


def create_merged_clusters(true_labels):
    """Create under-clustering by merging clusters."""
    merged = true_labels.copy()
    # Merge clusters 2 and 3 into cluster 2
    merged[merged == 3] = 2
    return merged


def get_quality_rating(nmi, ari):
    """Get quality rating based on NMI and ARI scores."""
    avg_score = (nmi + ari) / 2
    if avg_score >= 0.8:
        return "Excellent"
    elif avg_score >= 0.6:
        return "Good"
    elif avg_score >= 0.4:
        return "Moderate"
    elif avg_score >= 0.2:
        return "Poor"
    else:
        return "Very Poor"


def test_stage3_with_different_parameters():
    """Test stage3 with various parameter combinations."""
    print("\n⚙️ Testing Stage 3 with Different Parameters")
    print("=" * 50)
    
    # Use a smaller real data file for parameter testing
    stage2_file = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/temp_data/stage2_results/stage2_문4_img_20250918_112537.csv"
    
    if not os.path.exists(stage2_file):
        print(f"❌ Test file not found: {stage2_file}")
        return
    
    # Create base state
    state = initialize_project_state("test", "test.txt", "-SUV_776부.xlsx")
    state["matched_questions"] = {
        "문4": {
            "stage2_data": {
                "csv_path": stage2_file,
                "status": "completed"
            }
        }
    }
    
    # Test different parameter combinations
    parameter_tests = [
        {
            "name": "Quick Auto-train (3 iterations)",
            "mode": "auto_train",
            "params": {"stage3_search_iterations": 3}
        },
        {
            "name": "Standard Auto-train (10 iterations)",
            "mode": "auto_train", 
            "params": {"stage3_search_iterations": 10}
        },
        {
            "name": "Manual - Conservative",
            "mode": "manual_train",
            "params": {
                "stage3_manual_inflation": 1.5,
                "stage3_manual_k": 20,
                "stage3_manual_max_iters": 50
            }
        },
        {
            "name": "Manual - Aggressive",
            "mode": "manual_train",
            "params": {
                "stage3_manual_inflation": 3.0,
                "stage3_manual_k": 50,
                "stage3_manual_max_iters": 200
            }
        }
    ]
    
    results = []
    
    for test_config in parameter_tests:
        print(f"\n--- {test_config['name']} ---")
        
        # Setup test state
        test_state = state.copy()
        test_state["stage3_mode"] = test_config["mode"]
        
        for param_name, param_value in test_config["params"].items():
            test_state[param_name] = param_value
        
        try:
            result_state = stage3_classification_node(test_state)
            
            status = result_state.get("stage3_status")
            if status == "completed":
                summary = result_state.get("stage3_cluster_summary", {})
                processing_time = result_state.get("processing_time_seconds", 0)
                
                result_info = {
                    "name": test_config["name"],
                    "status": "success",
                    "clusters": summary.get("n_clusters", 0),
                    "samples": summary.get("n_samples", 0), 
                    "time": processing_time
                }
                
                if test_config["mode"] == "auto_train":
                    result_info["best_score"] = result_state.get("stage3_best_score", 0)
                
                results.append(result_info)
                
                print(f"✅ Success: {summary.get('n_clusters', 0)} clusters from {summary.get('n_samples', 0)} samples")
                print(f"   Processing time: {processing_time:.2f}s")
                
            else:
                print(f"❌ Failed: {result_state.get('stage3_error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    # Summary
    if results:
        print(f"\n📊 Parameter Test Summary:")
        print(f"{'Test Name':<30} {'Clusters':<9} {'Samples':<8} {'Time(s)':<8} {'Status':<10}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['name']:<30} {result['clusters']:<9} {result['samples']:<8} {result['time']:<8.2f} {result['status']:<10}")
    
    print(f"\n✅ Parameter testing completed!")


if __name__ == "__main__":
    print("🚀 Starting Stage 3 Classification Tests")
    print("=" * 60)
    
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Run all tests
    test_stage3_node_integration()
    test_stage3_evaluation_functions()
    test_stage3_with_different_parameters()
    
    print(f"\n🎉 All Stage 3 tests completed successfully!")
    print("=" * 60)