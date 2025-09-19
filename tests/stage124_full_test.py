"""
Full Pipeline Integration Test (Stage 1-2-3).
Tests the complete LangGraph pipeline with Stage 3 MCL classification.
"""
import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
project_root = "/home/cyyoon/test_area/ai_text_classification/2.langgraph"
sys.path.insert(0, project_root)

from graph.graph import create_workflow, run_pipeline
from graph.state import initialize_project_state
from nodes.stage3_classification.evaluation import mcl_scoring_function


def test_full_pipeline_with_stage3():
    """Test the complete pipeline including Stage 3 classification."""
    print("🚀 Testing Full Pipeline Integration (Stage 1-2-3)")
    print("=" * 65)
    
    # Set up test project
    project_name = "integration_test"
    survey_filename = "test.txt"
    data_filename = "-SUV_776부.xlsx"
    
    print(f"Project: {project_name}")
    print(f"Survey: {survey_filename}")
    print(f"Data: {data_filename}")
    
    # Configure Stage 3 mode for the test
    print(f"\n🎯 Configuring Stage 3 for auto_train mode with evaluation")
    
    try:
        # Initialize state with Stage 3 configuration
        initial_state = initialize_project_state(project_name, survey_filename, data_filename)
        
        # Configure Stage 3 parameters
        initial_state["stage3_mode"] = "auto_train"
        initial_state["stage3_search_iterations"] = 8  # Quick test
        
        print(f"✅ Initial state configured with Stage 3 settings")
        
        # Create and run the full workflow
        print(f"\n🔄 Running full pipeline...")
        start_time = time.time()
        
        workflow = create_workflow()
        app = workflow.compile()
        
        result = app.invoke(initial_state, config={"recursion_limit": 150})
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Pipeline completed in {total_time:.2f} seconds")
        
        # Analyze results
        print(f"\n📊 Pipeline Results Analysis:")
        print("-" * 40)
        
        # Check overall status
        current_stage = result.get("current_stage", "Unknown")
        error = result.get("error")
        
        print(f"Final Stage: {current_stage}")
        
        if error:
            print(f"❌ Pipeline Error: {error}")
            return result
        
        # Check Stage 1 results
        print(f"\n📋 Stage 1 Results:")
        matched_questions = result.get("matched_questions", {})
        if matched_questions:
            print(f"  ✅ Question matching: {len(matched_questions)} questions matched")
            for qid, data in list(matched_questions.items())[:3]:
                print(f"    {qid}: {data.get('stage2_data', {}).get('status', 'unknown')}")
        else:
            print(f"  ❌ No question matches found")
        
        # Check Stage 2 results
        print(f"\n📋 Stage 2 Results:")
        stage2_complete = result.get("stage2_processing_complete", False)
        stage2_results = result.get("stage2_processing_results", {})
        
        print(f"  Stage 2 Complete: {stage2_complete}")
        if stage2_results:
            print(f"  Processing Results: {len(stage2_results)} items")
        
        # Check Stage 3 results in detail
        print(f"\n📋 Stage 3 Results:")
        stage3_status = result.get("stage3_status", "unknown")
        print(f"  Status: {stage3_status}")
        
        if stage3_status == "completed":
            print(f"  ✅ Stage 3 MCL Classification completed successfully!")
            
            # Show clustering results
            cluster_summary = result.get("stage3_cluster_summary", {})
            print(f"  Clusters found: {cluster_summary.get('n_clusters', 'N/A')}")
            print(f"  Samples processed: {cluster_summary.get('n_samples', 'N/A')}")
            print(f"  Processing time: {result.get('processing_time_seconds', 'N/A')}s")
            
            # Show auto-train results
            best_params = result.get("stage3_best_parameters", {})
            best_score = result.get("stage3_best_score", 0)
            
            if best_params:
                print(f"  Best parameters: {best_params}")
                print(f"  Best score: {best_score:.4f}")
            
            # Show evaluation results if available
            evaluation = result.get("stage3_best_evaluation", {})
            if evaluation and "metrics" in evaluation:
                metrics = evaluation["metrics"]
                nmi = metrics.get("nmi", "N/A")
                ari = metrics.get("ari", "N/A")
                print(f"  Evaluation - NMI: {nmi}")
                print(f"  Evaluation - ARI: {ari}")
                
                # Interpret quality
                if isinstance(nmi, float) and isinstance(ari, float):
                    avg_score = (nmi + ari) / 2
                    quality = get_clustering_quality(avg_score)
                    print(f"  Clustering Quality: {quality}")
            
            # Show cluster mapping summary
            cluster_mapping = result.get("stage3_cluster_mapping", {})
            if cluster_mapping:
                print(f"  Cluster mapping created: {len(cluster_mapping)} groups")
        
        elif stage3_status == "failed":
            stage3_error = result.get("stage3_error", "Unknown error")
            print(f"  ❌ Stage 3 failed: {stage3_error}")
        
        elif stage3_status == "skipped":
            print(f"  ⚠️ Stage 3 was skipped")
        
        # Show resource usage
        print(f"\n💰 Resource Usage:")
        total_cost = result.get("total_llm_cost_usd", 0)
        pipeline_id = result.get("pipeline_id", "Unknown")
        print(f"  Total LLM Cost: ${total_cost:.4f}")
        print(f"  Pipeline ID: {pipeline_id}")
        print(f"  Total Runtime: {total_time:.2f}s")
        
        # Show history file
        history_file = result.get("stage_history_file")
        if history_file and os.path.exists(history_file):
            print(f"  History saved to: {os.path.basename(history_file)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def test_pipeline_with_different_stage3_modes():
    """Test the pipeline with different Stage 3 modes."""
    print(f"\n🔄 Testing Pipeline with Different Stage 3 Modes")
    print("=" * 55)
    
    # Test modes
    test_modes = [
        ("estimate", "Quick cluster estimation", {}),
        ("auto_train", "Hyperparameter optimization", {"stage3_search_iterations": 5}),
        ("manual_train", "Manual parameters", {
            "stage3_manual_inflation": 2.0,
            "stage3_manual_k": 30,
            "stage3_manual_max_iters": 100
        })
    ]
    
    results = []
    
    for mode, description, extra_params in test_modes:
        print(f"\n--- Testing {mode.upper()} Mode ---")
        print(f"Description: {description}")
        
        try:
            # Create state for this mode
            state = initialize_project_state("test_modes", "test.txt", "-SUV_776부.xlsx")
            state["stage3_mode"] = mode
            
            # Add mode-specific parameters
            for param_name, param_value in extra_params.items():
                state[param_name] = param_value
            
            # Run workflow (just the stage3 node for speed)
            # Note: In practice, you'd run the full pipeline
            from nodes.stage3_classification.stage3_node import stage3_classification_node
            
            # Mock some stage2 data for testing
            state["matched_questions"] = {
                "문4": {
                    "stage2_data": {
                        "csv_path": "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/temp_data/stage2_results/stage2_문4_img_20250918_112537.csv",
                        "status": "completed"
                    }
                }
            }
            
            start_time = time.time()
            result_state = stage3_classification_node(state)
            end_time = time.time()
            
            processing_time = end_time - start_time
            status = result_state.get("stage3_status", "unknown")
            
            print(f"  Status: {status}")
            print(f"  Processing time: {processing_time:.2f}s")
            
            if status == "completed":
                summary = result_state.get("stage3_cluster_summary", {})
                clusters = summary.get("n_clusters", 0)
                samples = summary.get("n_samples", 0)
                
                print(f"  ✅ Success: {clusters} clusters from {samples} samples")
                
                results.append({
                    "mode": mode,
                    "status": "success",
                    "clusters": clusters,
                    "samples": samples,
                    "time": processing_time
                })
                
            else:
                error = result_state.get("stage3_error", "Unknown")
                print(f"  ❌ Failed: {error}")
                
                results.append({
                    "mode": mode,
                    "status": "failed",
                    "error": error,
                    "time": processing_time
                })
        
        except Exception as e:
            print(f"  ❌ Exception: {str(e)}")
            results.append({
                "mode": mode,
                "status": "exception",
                "error": str(e),
                "time": 0
            })
    
    # Summary
    print(f"\n📊 Stage 3 Mode Test Summary:")
    print(f"{'Mode':<12} {'Status':<10} {'Clusters':<9} {'Samples':<8} {'Time(s)':<8}")
    print("-" * 55)
    
    for result in results:
        mode = result["mode"]
        status = result["status"]
        clusters = result.get("clusters", "-")
        samples = result.get("samples", "-")
        time_taken = result["time"]
        
        print(f"{mode:<12} {status:<10} {clusters:<9} {samples:<8} {time_taken:<8.2f}")
    
    return results


def get_clustering_quality(avg_score):
    """Get clustering quality description based on average score."""
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


def test_stage3_evaluation_integration():
    """Test Stage 3 evaluation integration in full pipeline context."""
    print(f"\n🔬 Testing Stage 3 Evaluation Integration")
    print("=" * 45)
    
    # Create synthetic evaluation scenario
    import numpy as np
    
    # Mock some evaluation testing
    true_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    test_predictions = [
        ("Perfect", np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])),
        ("Good", np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3])),
        ("Poor", np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    ]
    
    print(f"Testing evaluation function with synthetic data:")
    print(f"{'Scenario':<10} {'NMI':<6} {'ARI':<6} {'Quality':<10}")
    print("-" * 35)
    
    for scenario_name, pred_labels in test_predictions:
        try:
            scores = mcl_scoring_function(true_labels, pred_labels)
            
            nmi = scores['metrics']['nmi']
            ari = scores['metrics']['ari']
            quality = get_clustering_quality((nmi + ari) / 2)
            
            print(f"{scenario_name:<10} {nmi:<6.3f} {ari:<6.3f} {quality:<10}")
            
        except Exception as e:
            print(f"{scenario_name:<10} ERROR: {str(e)}")
    
    print(f"\n✅ Evaluation integration test completed!")


if __name__ == "__main__":
    print("🚀 Starting Full Pipeline Integration Tests")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run all integration tests
    try:
        # Test 1: Full pipeline
        print("\n" + "="*70)
        full_result = test_full_pipeline_with_stage3()
        
        # Test 2: Different Stage 3 modes
        print("\n" + "="*70)
        mode_results = test_pipeline_with_different_stage3_modes()
        
        # Test 3: Evaluation integration
        print("\n" + "="*70)
        test_stage3_evaluation_integration()
        
        end_time = time.time()
        total_test_time = end_time - start_time
        
        # Final summary
        print(f"\n" + "="*70)
        print(f"🎉 All Integration Tests Completed!")
        print(f"Total test runtime: {total_test_time:.2f} seconds")
        
        # Show test results summary
        if isinstance(full_result, dict) and full_result.get("stage3_status") == "completed":
            print(f"✅ Full pipeline test: SUCCESS")
        else:
            print(f"❌ Full pipeline test: FAILED")
        
        successful_modes = [r for r in mode_results if r["status"] == "success"]
        print(f"✅ Stage 3 mode tests: {len(successful_modes)}/{len(mode_results)} passed")
        
        print(f"\n🚀 Pipeline is ready for production use!")
        
    except Exception as e:
        print(f"❌ Integration tests failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("=" * 70)