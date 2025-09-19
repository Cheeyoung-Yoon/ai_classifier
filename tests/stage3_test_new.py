"""
Stage 3 Classification Test for LangGraph Pipeline.
Tests the integrated state-based clustering with evaluation capabilities.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path using config
try:
    from config.config import settings
    project_root = Path(settings.PROJECT_DATA_BASE_DIR).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except ImportError:
    # Fallback for when config is not available
    project_root = "/home/cyyoon/test_area/ai_text_classification/2.langgraph"
    sys.path.insert(0, project_root)

from graph.state import GraphState, initialize_project_state
from nodes.stage3_classification.state_based_stage3_node import state_based_stage3_node
from nodes.stage3_classification.clustering_service import Stage3ClusteringService


def test_stage3_node_integration():
    """Test State-based Stage 3 node with mock matched_questions data."""
    print("🧪 Testing State-based Stage 3 Classification Node")
    print("=" * 60)
    
    # Create mock matched_questions data with embeddings
    print("📊 Creating mock matched_questions data with embeddings...")
    
    matched_questions = {
        "문4": {
            "question_info": {
                "id": "문4",
                "type": "img",
                "text": "Test question 4"
            },
            "stage2_data": {
                "embeddings": np.random.rand(50, 384).tolist(),  # Mock embeddings
                "texts": [f"Test text {i}" for i in range(50)],
                "labels": [f"label_{i % 5}" for i in range(50)],  # 5 different labels
                "status": "completed"
            }
        },
        "문5": {
            "question_info": {
                "id": "문5", 
                "type": "img",
                "text": "Test question 5"
            },
            "stage2_data": {
                "embeddings": np.random.rand(30, 384).tolist(),  # Mock embeddings
                "texts": [f"Test text {i}" for i in range(30)],
                "labels": [f"label_{i % 3}" for i in range(30)],  # 3 different labels
                "status": "completed"
            }
        }
    }
    
    # Initialize state
    state = initialize_project_state("test_project", "test.txt", "test_data.xlsx")
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
            test_state["stage3_search_iterations"] = 3  # Quick test
        elif mode == "manual_train":
            test_state["stage3_manual_inflation"] = 2.0
            test_state["stage3_manual_k"] = 10
            test_state["stage3_manual_max_iters"] = 50
        
        try:
            # Run the state-based stage3 node
            print(f"🔄 Running state-based stage3 node...")
            result_state = state_based_stage3_node(test_state)
            
            # Check results
            status = result_state.get("stage3_status", "unknown")
            print(f"✅ Status: {status}")
            
            if status == "completed":
                processing_time = result_state.get("processing_time_seconds", 0)
                print(f"⏱️  Processing time: {processing_time:.2f}s")
                
                # Display mode-specific results
                if mode == "estimate":
                    estimated_clusters = result_state.get("stage3_estimated_clusters", 0)
                    recommended_k = result_state.get("stage3_recommended_k", 0)
                    print(f"📊 Estimated clusters: {estimated_clusters}")
                    print(f"📊 Recommended k: {recommended_k}")
                    
                elif mode == "auto_train":
                    best_score = result_state.get("stage3_best_score", 0)
                    best_params = result_state.get("stage3_best_parameters", {})
                    print(f"🏆 Best score: {best_score:.3f}")
                    print(f"🔧 Best parameters: {best_params}")
                    
                elif mode == "manual_train":
                    cluster_labels = result_state.get("stage3_cluster_labels", [])
                    cluster_mapping = result_state.get("stage3_cluster_mapping", {})
                    print(f"🏷️  Cluster labels count: {len(cluster_labels)}")
                    print(f"🗂️  Questions processed: {len(cluster_mapping)}")
                
                # Common metrics
                data_summary = result_state.get("stage3_data_summary", {})
                if data_summary:
                    print(f"📈 Data summary: {data_summary}")
                    
            elif status == "failed":
                error = result_state.get("stage3_error", "Unknown error")
                print(f"❌ Error: {error}")
                
        except Exception as e:
            print(f"💥 Exception in {mode} mode: {str(e)}")
            continue
    
    print(f"\n✅ Stage 3 integration test completed!")


def test_clustering_service_directly():
    """Test the clustering service directly with mock data."""
    print("\n🔬 Testing Clustering Service Directly")
    print("=" * 50)
    
    # Create mock matched_questions
    matched_questions = {
        "test_question": {
            "stage2_data": {
                "embeddings": np.random.rand(20, 384).tolist(),
                "texts": [f"Sample text {i}" for i in range(20)],
                "labels": [f"class_{i % 4}" for i in range(20)],
                "status": "completed"
            }
        }
    }
    
    # Test the service
    service = Stage3ClusteringService()
    
    # Test estimate mode
    result = service.process_matched_questions(matched_questions, "estimate")
    print(f"Estimate result: {result.get('stage3_status')}")
    
    if result.get("stage3_status") == "completed":
        print(f"Estimated clusters: {result.get('stage3_estimated_clusters', 0)}")
    
    print("✅ Direct service test completed!")


if __name__ == "__main__":
    test_stage3_node_integration()
    test_clustering_service_directly()