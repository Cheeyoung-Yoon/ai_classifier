"""
Main classification node for Stage 3 MCL clustering.
Handles all three modes in a single clean interface.
"""
import time
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    from .data_loader import load_data_from_state, map_clusters_back_to_data
    from .mcl_pipeline import estimate_clusters, auto_train_mcl, manual_train_mcl
except ImportError:
    # Direct execution fallback
    from data_loader import load_data_from_state, map_clusters_back_to_data
    from mcl_pipeline import estimate_clusters, auto_train_mcl, manual_train_mcl


def stage3_classify(state: Dict[str, Any]) -> Dict[str, Any]:
    """Main Stage 3 classification node - handles all modes.
    
    Args:
        state: LangGraph state with validated classification request
        
    Returns:
        Updated state with classification results
    """
    start_time = time.time()
    mode = state.get("stage3_validated_mode", state.get("stage3_mode", "estimate"))
    
    try:
        print(f"🎯 Stage 3 Classification: Starting {mode} mode")
        
        # Load data from state
        embeddings, metadata = load_data_from_state(state)
        print(f"📊 Loaded {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions")
        
        # Route to appropriate mode
        if mode == "estimate":
            result = _estimate_mode(embeddings, metadata)
            
        elif mode == "auto_train":
            search_iterations = state.get("stage3_search_iterations", 10)
            result = _auto_train_mode(embeddings, metadata, search_iterations)
            
        elif mode == "manual_train":
            manual_params = {
                "inflation": state.get("stage3_manual_inflation", 2.0),
                "k": state.get("stage3_manual_k", 50),
                "max_iters": state.get("stage3_manual_max_iters", 100)
            }
            result = _manual_train_mode(embeddings, metadata, manual_params)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Add timing and finalize result
        result["processing_time_seconds"] = round(time.time() - start_time, 2)
        result["stage3_status"] = "completed"
        
        print(f"✅ Stage 3 Classification: {mode} completed in {result['processing_time_seconds']}s")
        
        return {**state, **result}
        
    except Exception as e:
        error_msg = f"Stage 3 classification failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        return {
            **state,
            "stage3_status": "failed",
            "stage3_error": error_msg,
            "processing_time_seconds": round(time.time() - start_time, 2)
        }


def _estimate_mode(embeddings: np.ndarray, metadata: Dict) -> Dict[str, Any]:
    """Handle estimate mode - quick cluster estimation.
    
    Args:
        embeddings: Input embeddings
        metadata: Data metadata
        
    Returns:
        Estimation results
    """
    estimation = estimate_clusters(embeddings)
    
    return {
        "stage3_mode": "estimate",
        "stage3_estimated_clusters": estimation.get("estimated_clusters", 0),
        "stage3_recommended_k": estimation.get("recommended_k", 50),
        "stage3_estimation_details": estimation,
        "stage3_data_summary": {
            "n_samples": embeddings.shape[0],
            "n_features": embeddings.shape[1],
            "metadata": metadata
        }
    }


def _auto_train_mode(embeddings: np.ndarray, metadata: Dict, search_iterations: int) -> Dict[str, Any]:
    """Handle auto_train mode - hyperparameter search.
    
    Args:
        embeddings: Input embeddings
        metadata: Data metadata
        search_iterations: Number of search iterations
        
    Returns:
        Auto training results
    """
    # Check if true labels are available in metadata for evaluation
    true_labels = metadata.get("true_labels", None)
    
    training_result = auto_train_mcl(embeddings, search_iterations, true_labels=true_labels)
    
    # Get cluster labels if training succeeded
    cluster_labels = None
    cluster_mapping = None
    
    if training_result.get("best_parameters"):
        best_params = training_result["best_parameters"]
        manual_result = manual_train_mcl(embeddings, **best_params)
        if manual_result.get("status") == "success":
            cluster_labels = manual_result["cluster_labels"]
            # Map clusters back to original data with IDs
            cluster_mapping = map_clusters_back_to_data(cluster_labels, metadata)
    
    return {
        "stage3_mode": "auto_train",
        "stage3_best_parameters": training_result.get("best_parameters", {}),
        "stage3_best_score": training_result.get("best_score", 0),
        "stage3_best_evaluation": training_result.get("best_evaluation", {}),
        "stage3_cluster_labels": cluster_labels.tolist() if cluster_labels is not None else None,
        "stage3_cluster_mapping": cluster_mapping,  # Detailed ID mapping
        "stage3_cluster_summary": training_result.get("best_summary", {}),
        "stage3_search_results": training_result.get("all_results", []),
        "stage3_search_iterations": search_iterations,
        "stage3_data_summary": {
            "n_samples": embeddings.shape[0],
            "n_features": embeddings.shape[1],
            "metadata": metadata,
            "has_true_labels": true_labels is not None
        }
    }


def _manual_train_mode(embeddings: np.ndarray, metadata: Dict, manual_params: Dict) -> Dict[str, Any]:
    """Handle manual_train mode - user specified parameters.
    
    Args:
        embeddings: Input embeddings
        metadata: Data metadata
        manual_params: User-specified parameters
        
    Returns:
        Manual training results
    """
    training_result = manual_train_mcl(embeddings, **manual_params)
    
    if training_result.get("status") != "success":
        raise Exception(f"Manual training failed: {training_result.get('error')}")
    
    cluster_labels = training_result["cluster_labels"]
    # Map clusters back to original data with IDs
    cluster_mapping = map_clusters_back_to_data(cluster_labels, metadata)
    
    return {
        "stage3_mode": "manual_train",
        "stage3_manual_parameters": manual_params,
        "stage3_cluster_labels": cluster_labels.tolist(),
        "stage3_cluster_mapping": cluster_mapping,  # Detailed ID mapping
        "stage3_cluster_summary": training_result["summary"],
        "stage3_data_summary": {
            "n_samples": embeddings.shape[0],
            "n_features": embeddings.shape[1],
            "metadata": metadata
        }
    }