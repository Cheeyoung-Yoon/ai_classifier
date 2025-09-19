"""
Test script for the new Stage 3 Two-Phase Classification System
Tests the complete kNN → CSLS → MCL → Community Detection pipeline.
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path

# Add project root to path
try:
    from config.config import settings
    project_root = Path(settings.PROJECT_DATA_BASE_DIR).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except ImportError:
    project_root = "/home/cyyoon/test_area/ai_text_classification/2.langgraph"
    sys.path.insert(0, project_root)

from graph.state import GraphState, initialize_project_state
from nodes.stage3_classification import (
    stage3_main_node,
    two_phase_stage3_node,
    phase1_primary_labeling_node,
    phase2_secondary_labeling_node,
    quality_assessment_node
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_embeddings_data(n_samples: int = 100, n_features: int = 384, n_clusters: int = 5):
    """Create mock embeddings data for testing."""
    np.random.seed(42)  # For reproducibility
    
    # Create cluster centers
    centers = np.random.randn(n_clusters, n_features)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)  # L2 normalize
    
    # Generate samples around centers
    embeddings = []
    labels = []
    texts = []
    
    samples_per_cluster = n_samples // n_clusters
    remaining_samples = n_samples % n_clusters
    
    for cluster_id in range(n_clusters):
        n_cluster_samples = samples_per_cluster + (1 if cluster_id < remaining_samples else 0)
        
        for i in range(n_cluster_samples):
            # Add noise to center
            noise = np.random.randn(n_features) * 0.1
            sample = centers[cluster_id] + noise
            sample = sample / np.linalg.norm(sample)  # L2 normalize
            
            embeddings.append(sample)
            labels.append(cluster_id)
            texts.append(f"Sample text {len(embeddings)} from cluster {cluster_id}")
    
    return np.array(embeddings), labels, texts


def create_test_state():
    """Create test state with mock data."""
    # Create mock embeddings
    embeddings, ground_truth_labels, texts = create_mock_embeddings_data(
        n_samples=150, n_features=384, n_clusters=6
    )
    
    # Create matched_questions structure
    matched_questions = {
        "Q1": {
            "embeddings": embeddings[:30].tolist(),  # First part
            "texts": texts[:30],
            "original_labels": ground_truth_labels[:30],
            "question_text": "Mock question 1",
            "data_column": "column1"
        },
        "Q2": {
            "embeddings": embeddings[30:60].tolist(),  # Second part
            "texts": texts[30:60],
            "original_labels": ground_truth_labels[30:60],
            "question_text": "Mock question 2", 
            "data_column": "column2"
        }
    }
    
    # Initialize state
    state = initialize_project_state(
        project_name="stage3_test",
        survey_filename="test_survey.txt",
        data_filename="test_data.xlsx"
    )
    
    # Add matched questions
    state["matched_questions"] = matched_questions
    
    # Set Stage 3 mode and parameters
    state["stage3_mode"] = "two_phase"
    
    # Phase 1 parameters
    state["stage3_phase1_knn_k"] = 30
    state["stage3_phase1_csls_threshold"] = 0.05
    state["stage3_phase1_mcl_inflation"] = 1.8
    state["stage3_phase1_top_m"] = 15
    
    # Phase 2 parameters
    state["stage3_phase2_algorithm"] = "louvain"
    state["stage3_phase2_resolution"] = 1.2
    state["stage3_phase2_mode"] = "llm_assisted"
    state["stage3_phase2_edge_weights"] = {"alpha": 0.6, "beta": 0.2, "gamma": 0.2}
    
    return state


def test_phase1_only():
    """Test Phase 1 primary labeling only."""
    print("\n" + "="*60)
    print("🧪 Testing Phase 1 Primary Labeling")
    print("="*60)
    
    state = create_test_state()
    
    try:
        result_state = phase1_primary_labeling_node(state)
        
        status = result_state.get("stage3_phase1_status")
        print(f"Status: {status}")
        
        if status == "completed":
            groups = result_state.get("stage3_phase1_groups", {})
            prototypes = result_state.get("stage3_phase1_prototypes", {})
            quality_stats = result_state.get("stage3_phase1_quality_stats", {})
            
            print("✅ Phase 1 completed successfully!")
            
            for question_id, result in groups.items():
                if result.get("status") == "completed":
                    print(f"\n📊 Question {question_id}:")
                    print(f"   Clusters: {result.get('n_clusters', 0)}")
                    print(f"   Singletons: {result.get('n_singletons', 0)}")
                    print(f"   Samples: {result.get('n_samples', 0)}")
                    print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
                    
                    # Show quality metrics if available
                    if question_id in quality_stats:
                        quality = quality_stats[question_id]
                        if "subset_score" in quality:
                            subset_score = quality["subset_score"].get("score", 0)
                            print(f"   SubsetScore: {subset_score:.3f}")
        else:
            error = result_state.get("stage3_error", "Unknown error")
            print(f"❌ Phase 1 failed: {error}")
            
        return result_state
        
    except Exception as e:
        print(f"❌ Phase 1 test failed: {str(e)}")
        return None


def test_phase2_only():
    """Test Phase 2 secondary labeling with Phase 1 results."""
    print("\n" + "="*60)
    print("🧪 Testing Phase 2 Secondary Labeling")
    print("="*60)
    
    # First run Phase 1
    state = create_test_state()
    state = phase1_primary_labeling_node(state)
    
    if state.get("stage3_phase1_status") != "completed":
        print("❌ Phase 1 failed, cannot test Phase 2")
        return None
    
    try:
        result_state = phase2_secondary_labeling_node(state)
        
        status = result_state.get("stage3_phase2_status")
        print(f"Status: {status}")
        
        if status == "completed":
            labels = result_state.get("stage3_phase2_labels", {})
            
            print("✅ Phase 2 completed successfully!")
            print(f"   Generated {len(labels)} labels")
            
            for label_id, label_data in labels.items():
                print(f"\n🏷️ Label: {label_id}")
                print(f"   Name: {label_data.get('name', 'N/A')}")
                print(f"   Size: {label_data.get('size', 0)}")
                print(f"   Groups: {len(label_data.get('group_ids', []))}")
                print(f"   Keywords: {label_data.get('keywords', [])[:3]}")  # Show first 3
                
        else:
            error = result_state.get("stage3_error", "Unknown error")
            print(f"❌ Phase 2 failed: {error}")
            
        return result_state
        
    except Exception as e:
        print(f"❌ Phase 2 test failed: {str(e)}")
        return None


def test_quality_assessment():
    """Test quality assessment on complete results."""
    print("\n" + "="*60)
    print("🧪 Testing Quality Assessment")
    print("="*60)
    
    # Run full pipeline first
    state = create_test_state()
    state = two_phase_stage3_node(state)
    
    if state.get("stage3_status") != "completed":
        print("❌ Two-phase pipeline failed, cannot test quality assessment")
        return None
    
    try:
        result_state = quality_assessment_node(state)
        
        phase1_quality = result_state.get("stage3_phase1_quality", {})
        phase2_quality = result_state.get("stage3_phase2_quality", {})
        
        print("✅ Quality assessment completed!")
        
        # Show Phase 1 quality
        if phase1_quality:
            print("\n📊 Phase 1 Quality:")
            for question_id, quality in phase1_quality.items():
                if "overall_quality" in quality:
                    overall = quality["overall_quality"]
                    score = overall.get("weighted_score", 0)
                    level = overall.get("quality_level", "unknown")
                    print(f"   {question_id}: {score:.3f} ({level})")
        
        # Show Phase 2 quality
        if phase2_quality and "overall_quality" in phase2_quality:
            overall = phase2_quality["overall_quality"]
            score = overall.get("weighted_score", 0)
            level = overall.get("quality_level", "unknown")
            print(f"\n📊 Phase 2 Quality: {score:.3f} ({level})")
        
        return result_state
        
    except Exception as e:
        print(f"❌ Quality assessment test failed: {str(e)}")
        return None


def test_full_pipeline():
    """Test the complete two-phase pipeline."""
    print("\n" + "="*80)
    print("🚀 Testing Complete Two-Phase Pipeline")
    print("="*80)
    
    state = create_test_state()
    
    try:
        result_state = stage3_main_node(state)
        
        status = result_state.get("stage3_status")
        print(f"Overall Status: {status}")
        
        if status == "completed":
            print("✅ Two-Phase Pipeline completed successfully!")
            
            # Phase 1 summary
            phase1_groups = result_state.get("stage3_phase1_groups", {})
            total_clusters = sum(
                result.get("n_clusters", 0) 
                for result in phase1_groups.values() 
                if result.get("status") == "completed"
            )
            
            # Phase 2 summary
            phase2_labels = result_state.get("stage3_phase2_labels", {})
            
            print(f"\n📊 Pipeline Summary:")
            print(f"   Phase 1 Primary Clusters: {total_clusters}")
            print(f"   Phase 2 Semantic Labels: {len(phase2_labels)}")
            
            # Quality summary
            phase1_quality = result_state.get("stage3_phase1_quality", {})
            phase2_quality = result_state.get("stage3_phase2_quality", {})
            
            if phase1_quality:
                avg_phase1_score = np.mean([
                    q.get("overall_quality", {}).get("weighted_score", 0) 
                    for q in phase1_quality.values() if "overall_quality" in q
                ])
                print(f"   Phase 1 Avg Quality: {avg_phase1_score:.3f}")
            
            if phase2_quality and "overall_quality" in phase2_quality:
                phase2_score = phase2_quality["overall_quality"].get("weighted_score", 0)
                print(f"   Phase 2 Quality: {phase2_score:.3f}")
            
            print("\n🎉 All tests completed successfully!")
            
        else:
            error = result_state.get("stage3_error", "Unknown error")
            print(f"❌ Pipeline failed: {error}")
            
        return result_state
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {str(e)}")
        return None


def main():
    """Run all tests."""
    print("🧪 Stage 3 Two-Phase Classification System Tests")
    print("Testing new kNN → CSLS → MCL → Community Detection pipeline")
    
    try:
        # Test individual phases
        test_phase1_only()
        test_phase2_only()
        test_quality_assessment()
        
        # Test complete pipeline
        test_full_pipeline()
        
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)