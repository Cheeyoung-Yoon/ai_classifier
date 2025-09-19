"""
Stage3 Clustering Service
Dedicated clustering service that reads embeddings from state and performs clustering operations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import time
from pathlib import Path

from .singleton_aware_clustering_nmi import SingletonAwareClusteringNMI
from .nmi_ari_evaluation import NMIARIEvaluator

logger = logging.getLogger(__name__)


class Stage3ClusteringService:
    """
    Service for performing Stage3 clustering operations using state-based data.
    Replaces filesystem-based CSV/glob logic with direct state integration.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.clustering_engine = SingletonAwareClusteringNMI(self.config)
        self.evaluator = NMIARIEvaluator()
        
    def process_matched_questions(
        self, 
        matched_questions: Dict[str, Any],
        mode: str = "estimate"
    ) -> Dict[str, Any]:
        """
        Process matched questions data and perform clustering.
        
        Args:
            matched_questions: Dictionary of matched questions with embeddings
            mode: Clustering mode ('estimate', 'auto_train', 'manual_train')
            
        Returns:
            Dictionary with clustering results and updated stage3 fields
        """
        start_time = time.time()
        
        try:
            # Extract embeddings from matched questions
            embeddings_data = self._extract_embeddings_from_state(matched_questions)
            
            if not embeddings_data:
                return {
                    "stage3_status": "failed",
                    "stage3_error": "No valid embeddings found in matched_questions",
                    "processing_time_seconds": time.time() - start_time
                }
            
            # Perform clustering based on mode
            if mode == "estimate":
                result = self._estimate_clusters(embeddings_data)
            elif mode == "auto_train":
                result = self._auto_train_clustering(embeddings_data)
            elif mode == "manual_train":
                result = self._manual_train_clustering(embeddings_data)
            else:
                return {
                    "stage3_status": "failed", 
                    "stage3_error": f"Unknown mode: {mode}",
                    "processing_time_seconds": time.time() - start_time
                }
            
            # Add common fields
            result["stage3_mode"] = mode
            result["processing_time_seconds"] = time.time() - start_time
            result["stage3_data_summary"] = self._create_data_summary(embeddings_data)
            
            logger.info(f"Stage3 clustering completed in {result['processing_time_seconds']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Stage3 clustering failed: {str(e)}")
            return {
                "stage3_status": "failed",
                "stage3_error": str(e),
                "processing_time_seconds": time.time() - start_time
            }
    
    def _extract_embeddings_from_state(self, matched_questions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract embeddings data from matched_questions state.
        
        Args:
            matched_questions: State data containing question information
            
        Returns:
            Dictionary with processed embeddings data
        """
        embeddings_data = {}
        
        for question_id, question_data in matched_questions.items():
            try:
                # Check if question has stage2 data
                stage2_data = question_data.get("stage2_data", {})
                if not stage2_data:
                    logger.warning(f"No stage2_data found for question {question_id}")
                    continue
                
                # Try to load embeddings from CSV path if available
                csv_path = stage2_data.get("csv_path")
                if csv_path and Path(csv_path).exists():
                    df = pd.read_csv(csv_path)
                    
                    # Extract embeddings (assuming they're in columns starting with 'embed_')
                    embedding_cols = [col for col in df.columns if col.startswith('embed_')]
                    if embedding_cols:
                        embeddings = df[embedding_cols].values
                        embeddings_data[question_id] = {
                            "embeddings": embeddings,
                            "texts": df.get("text", df.get("processed_text", [])).tolist(),
                            "original_labels": df.get("label", df.get("category", [])).tolist(),
                            "question_info": question_data.get("question_info", {})
                        }
                        logger.info(f"Loaded {len(embeddings)} embeddings for question {question_id}")
                    else:
                        logger.warning(f"No embedding columns found in {csv_path}")
                
                # Alternative: check if embeddings are directly in state
                elif "embeddings" in stage2_data:
                    embeddings_data[question_id] = {
                        "embeddings": np.array(stage2_data["embeddings"]),
                        "texts": stage2_data.get("texts", []),
                        "original_labels": stage2_data.get("labels", []),
                        "question_info": question_data.get("question_info", {})
                    }
                    logger.info(f"Loaded embeddings from state for question {question_id}")
                
            except Exception as e:
                logger.error(f"Failed to process question {question_id}: {str(e)}")
                continue
        
        return embeddings_data
    
    def _estimate_clusters(self, embeddings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate optimal number of clusters."""
        if not embeddings_data:
            return {"stage3_status": "failed", "stage3_error": "No embeddings data available"}
        
        # Combine all embeddings for estimation
        all_embeddings = []
        for question_data in embeddings_data.values():
            all_embeddings.extend(question_data["embeddings"])
        
        if not all_embeddings:
            return {"stage3_status": "failed", "stage3_error": "No embeddings found"}
        
        embeddings_matrix = np.array(all_embeddings)
        
        # Use clustering engine to estimate clusters
        result = self.clustering_engine.estimate_clusters(embeddings_matrix)
        
        return {
            "stage3_status": "completed",
            "stage3_estimated_clusters": result.get("estimated_clusters", 3),
            "stage3_recommended_k": result.get("recommended_k", 3),
            "stage3_cluster_summary": result
        }
    
    def _auto_train_clustering(self, embeddings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform auto hyperparameter tuning with evaluation."""
        if not embeddings_data:
            return {"stage3_status": "failed", "stage3_error": "No embeddings data available"}
        
        best_overall_score = -1
        best_overall_params = None
        best_overall_evaluation = None
        question_results = {}
        
        for question_id, question_data in embeddings_data.items():
            embeddings = question_data["embeddings"]
            original_labels = question_data.get("original_labels", [])
            
            # Perform auto training for this question
            result = self.clustering_engine.auto_train(
                embeddings, 
                ground_truth_labels=original_labels if original_labels else None
            )
            
            question_results[question_id] = result
            
            # Track best overall result
            if result.get("best_score", -1) > best_overall_score:
                best_overall_score = result["best_score"]
                best_overall_params = result.get("best_parameters", {})
                best_overall_evaluation = result.get("best_evaluation", {})
        
        return {
            "stage3_status": "completed",
            "stage3_search_iterations": self.config.get("auto_train_iterations", 10),
            "stage3_best_parameters": best_overall_params,
            "stage3_best_score": best_overall_score,
            "stage3_best_evaluation": best_overall_evaluation,
            "stage3_cluster_mapping": question_results
        }
    
    def _manual_train_clustering(self, embeddings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform clustering with manual parameters."""
        if not embeddings_data:
            return {"stage3_status": "failed", "stage3_error": "No embeddings data available"}
        
        # Get manual parameters from config
        inflation = self.config.get("stage3_manual_inflation", 2.0)
        k = self.config.get("stage3_manual_k", 15)
        max_iters = self.config.get("stage3_manual_max_iters", 100)
        
        question_results = {}
        cluster_labels = []
        
        for question_id, question_data in embeddings_data.items():
            embeddings = question_data["embeddings"]
            original_labels = question_data.get("original_labels", [])
            
            # Perform manual clustering
            result = self.clustering_engine.manual_train(
                embeddings,
                inflation=inflation,
                k=k,
                max_iters=max_iters,
                ground_truth_labels=original_labels if original_labels else None
            )
            
            question_results[question_id] = result
            cluster_labels.extend(result.get("cluster_labels", []))
        
        return {
            "stage3_status": "completed",
            "stage3_manual_inflation": inflation,
            "stage3_manual_k": k,
            "stage3_manual_max_iters": max_iters,
            "stage3_cluster_labels": cluster_labels,
            "stage3_cluster_mapping": question_results
        }
    
    def _create_data_summary(self, embeddings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of processed data."""
        total_embeddings = 0
        total_questions = len(embeddings_data)
        
        for question_data in embeddings_data.values():
            total_embeddings += len(question_data.get("embeddings", []))
        
        return {
            "total_questions": total_questions,
            "total_embeddings": total_embeddings,
            "average_embeddings_per_question": total_embeddings / max(total_questions, 1)
        }