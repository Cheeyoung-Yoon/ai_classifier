"""
NMI and ARI scoring utilities for Trail 3 MCL clustering evaluation.
Based on the existing cluster evaluation implementation.
"""
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


class ClusterEvaluator:
    """Evaluator for clustering results using NMI and ARI metrics."""
    
    def __init__(self, gold_labels: Optional[Union[np.ndarray, List, pd.Series]] = None):
        """Initialize evaluator with optional ground truth labels.
        
        Args:
            gold_labels: Ground truth cluster labels (optional)
        """
        self.gold_labels = None
        if gold_labels is not None:
            self.gold_labels = np.array(gold_labels)
    
    def evaluate_clustering(self, predicted_labels: Union[np.ndarray, List, pd.Series]) -> Dict:
        """Evaluate clustering results against ground truth (if available).
        
        Args:
            predicted_labels: Predicted cluster labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = np.array(predicted_labels)
        n_pred_clusters = len(np.unique(y_pred))
        
        # If no ground truth, return basic statistics
        if self.gold_labels is None:
            return {
                "has_ground_truth": False,
                "n_predicted_clusters": n_pred_clusters,
                "n_samples": len(y_pred),
                "metrics": {
                    "nmi": np.nan,
                    "ari": np.nan,
                    "adj_nmi": np.nan
                }
            }
        
        # Ensure same length
        if len(self.gold_labels) != len(y_pred):
            raise ValueError(f"Ground truth ({len(self.gold_labels)}) and predicted ({len(y_pred)}) labels must have same length")
        
        y_true = self.gold_labels
        n_true_clusters = len(np.unique(y_true))
        
        # Calculate metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            nmi = normalized_mutual_info_score(y_true, y_pred)
            ari = adjusted_rand_score(y_true, y_pred)
            
            # Adjusted NMI (geometric mean normalization)
            adj_nmi = normalized_mutual_info_score(y_true, y_pred, average_method='geometric')
        
        # Calculate cluster size statistics
        true_counts = np.bincount(y_true)
        pred_counts = np.bincount(y_pred)
        
        # Identify singleton clusters
        true_singletons = (true_counts == 1).sum()
        pred_singletons = (pred_counts == 1).sum()
        
        true_singleton_ratio = true_singletons / max(len(true_counts), 1)
        pred_singleton_ratio = pred_singletons / max(len(pred_counts), 1)
        
        return {
            "has_ground_truth": True,
            "n_true_clusters": n_true_clusters,
            "n_predicted_clusters": n_pred_clusters,
            "n_samples": len(y_pred),
            "metrics": {
                "nmi": float(nmi),
                "ari": float(ari),
                "adj_nmi": float(adj_nmi)
            },
            "cluster_stats": {
                "true_singleton_ratio": float(true_singleton_ratio),
                "pred_singleton_ratio": float(pred_singleton_ratio),
                "true_singletons": int(true_singletons),
                "pred_singletons": int(pred_singletons)
            }
        }
    
    def evaluate_with_masks(self, predicted_labels: Union[np.ndarray, List, pd.Series]) -> Dict:
        """Comprehensive evaluation with different masking strategies.
        
        Args:
            predicted_labels: Predicted cluster labels
            
        Returns:
            Dictionary with detailed evaluation metrics
        """
        if self.gold_labels is None:
            basic_result = self.evaluate_clustering(predicted_labels)
            return {
                "all": basic_result["metrics"],
                "no_true_single": basic_result["metrics"].copy(),
                "no_pred_single": basic_result["metrics"].copy(),
                "no_any_single": basic_result["metrics"].copy(),
                "cluster_stats": {
                    "true_singleton_ratio": np.nan,
                    "pred_singleton_ratio": basic_result.get("cluster_stats", {}).get("pred_singleton_ratio", np.nan)
                }
            }
        
        y_true = self.gold_labels
        y_pred = np.array(predicted_labels)
        
        # Calculate cluster counts
        true_counts = np.bincount(y_true)
        pred_counts = np.bincount(y_pred)
        
        # Create masks to exclude singletons
        mask_true_non_single = (true_counts[y_true] >= 2)
        mask_pred_non_single = (pred_counts[y_pred] >= 2)
        
        # Evaluate with different masks
        all_metrics = self._masked_metrics(y_true, y_pred, None)
        no_true_single = self._masked_metrics(y_true, y_pred, mask_true_non_single)
        no_pred_single = self._masked_metrics(y_true, y_pred, mask_pred_non_single)
        no_any_single = self._masked_metrics(y_true, y_pred, mask_true_non_single & mask_pred_non_single)
        
        # Calculate singleton ratios
        true_singleton_ratio = float((true_counts == 1).sum() / max(true_counts.size, 1))
        pred_singleton_ratio = float((pred_counts == 1).sum() / max(pred_counts.size, 1))
        
        return {
            "all": all_metrics,
            "no_true_single": no_true_single,
            "no_pred_single": no_pred_single,
            "no_any_single": no_any_single,
            "cluster_stats": {
                "true_singleton_ratio": true_singleton_ratio,
                "pred_singleton_ratio": pred_singleton_ratio
            }
        }
    
    def _masked_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray]) -> Dict:
        """Calculate metrics with optional masking.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            mask: Boolean mask to apply (None for no masking)
            
        Returns:
            Dictionary with metrics
        """
        if mask is not None:
            if not mask.any():
                # No samples left after masking
                return {
                    "nmi": np.nan,
                    "ari": np.nan,
                    "adj_nmi": np.nan,
                    "n_true": 0,
                    "n_pred": 0,
                    "n_samples": 0
                }
            
            y_true_masked = y_true[mask]
            y_pred_masked = y_pred[mask]
        else:
            y_true_masked = y_true
            y_pred_masked = y_pred
        
        # Calculate metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            nmi = normalized_mutual_info_score(y_true_masked, y_pred_masked)
            ari = adjusted_rand_score(y_true_masked, y_pred_masked)
            adj_nmi = normalized_mutual_info_score(y_true_masked, y_pred_masked, average_method='geometric')
        
        return {
            "nmi": float(nmi),
            "ari": float(ari),
            "adj_nmi": float(adj_nmi),
            "n_true": len(np.unique(y_true_masked)),
            "n_pred": len(np.unique(y_pred_masked)),
            "n_samples": len(y_true_masked)
        }


def quick_evaluate(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> Dict:
    """Quick evaluation function for clustering results.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with NMI and ARI scores
    """
    evaluator = ClusterEvaluator(y_true)
    return evaluator.evaluate_clustering(y_pred)


def compare_clusterings(predictions: Dict[str, Union[np.ndarray, List]], 
                       ground_truth: Optional[Union[np.ndarray, List]] = None) -> pd.DataFrame:
    """Compare multiple clustering results.
    
    Args:
        predictions: Dictionary of {method_name: predicted_labels}
        ground_truth: Ground truth labels (optional)
        
    Returns:
        DataFrame comparing all methods
    """
    evaluator = ClusterEvaluator(ground_truth)
    
    results = []
    for method_name, pred_labels in predictions.items():
        eval_result = evaluator.evaluate_clustering(pred_labels)
        
        row = {
            "method": method_name,
            "n_clusters": eval_result["n_predicted_clusters"],
            "n_samples": eval_result["n_samples"],
            **eval_result["metrics"]
        }
        
        if "cluster_stats" in eval_result:
            row.update(eval_result["cluster_stats"])
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Sort by NMI if available, otherwise by method name
    if not df["nmi"].isna().all():
        df = df.sort_values("nmi", ascending=False)
    else:
        df = df.sort_values("method")
    
    return df


def mcl_scoring_function(true_labels: np.ndarray, predicted_labels: np.ndarray) -> Dict:
    """Scoring function for MCL clustering evaluation with NMI/ARI metrics.
    
    Args:
        true_labels: Ground truth cluster labels
        predicted_labels: Predicted cluster labels from MCL
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = ClusterEvaluator(true_labels)
    return evaluator.evaluate_clustering(predicted_labels)


def mcl_optimization_score(embeddings: np.ndarray, cluster_labels: np.ndarray, 
                          ground_truth: Optional[np.ndarray] = None) -> float:
    """Scoring function for MCL hyperparameter optimization.
    
    Args:
        embeddings: Input embeddings
        cluster_labels: Predicted cluster labels
        ground_truth: Ground truth labels (optional)
        
    Returns:
        Score for optimization (higher is better)
    """
    n_samples = len(cluster_labels)
    n_clusters = len(np.unique(cluster_labels))
    
    if ground_truth is not None:
        # Use NMI if ground truth is available
        evaluator = ClusterEvaluator(ground_truth)
        result = evaluator.evaluate_clustering(cluster_labels)
        return result["metrics"]["nmi"]
    
    else:
        # Heuristic scoring without ground truth
        # Prefer moderate number of clusters with balanced sizes
        
        # Target cluster count (heuristic)
        target_clusters = max(2, min(n_samples // 10, 50))
        
        # Cluster count score (closer to target is better)
        cluster_score = 1.0 / (1.0 + abs(n_clusters - target_clusters) / target_clusters)
        
        # Cluster balance score (prefer balanced cluster sizes)
        cluster_counts = np.bincount(cluster_labels)
        balance_score = 1.0 - (cluster_counts.std() / max(cluster_counts.mean(), 1))
        balance_score = max(0, balance_score)
        
        # Singleton penalty (too many single-item clusters is bad)
        singleton_count = (cluster_counts == 1).sum()
        singleton_ratio = singleton_count / n_clusters
        singleton_score = max(0, 1.0 - singleton_ratio)
        
        # Combined score
        total_score = (cluster_score + balance_score + singleton_score) / 3
        
        return float(total_score)