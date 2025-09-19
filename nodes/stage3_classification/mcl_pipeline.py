"""
Clean MCL Pipeline for Stage 3 Classification (Trail 3).
Simplified version with just the essentials.
"""
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    from .evaluation import mcl_scoring_function, mcl_optimization_score
except ImportError:
    from evaluation import mcl_scoring_function, mcl_optimization_score


class MCLPipeline:
    """Clean MCL clustering pipeline."""
    
    def __init__(self, inflation: float = 2.0, max_iters: int = 100):
        """Initialize MCL pipeline.
        
        Args:
            inflation: MCL inflation parameter (default 2.0)
            max_iters: Maximum iterations for MCL convergence
        """
        self.inflation = inflation
        self.max_iters = max_iters
        self.embeddings = None
        self.knn_graph = None
        self.cluster_labels = None
        
    def fit(self, embeddings: np.ndarray, k: int = 50) -> 'MCLPipeline':
        """Fit MCL clustering on embeddings.
        
        Args:
            embeddings: Input embedding matrix (n_samples, n_features)
            k: Number of nearest neighbors for graph construction
            
        Returns:
            Self for method chaining
        """
        self.embeddings = embeddings
        n_samples = embeddings.shape[0]
        
        # Build k-NN graph
        k_actual = min(k, n_samples - 1)
        knn = NearestNeighbors(n_neighbors=k_actual + 1, metric='cosine')
        knn.fit(embeddings)
        
        # Get adjacency matrix (exclude self-loops)
        distances, indices = knn.kneighbors(embeddings)
        adjacency = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(1, len(indices[i])):  # Skip self (index 0)
                neighbor = indices[i][j]
                # Convert cosine distance to similarity
                similarity = 1.0 - distances[i][j]
                adjacency[i][neighbor] = max(0, similarity)
                adjacency[neighbor][i] = max(0, similarity)  # Make symmetric
        
        self.knn_graph = adjacency
        
        # Run MCL clustering
        self.cluster_labels = self._mcl_cluster(adjacency)
        
        return self
    
    def _mcl_cluster(self, adjacency: np.ndarray) -> np.ndarray:
        """Run MCL clustering algorithm.
        
        Args:
            adjacency: Adjacency matrix
            
        Returns:
            Cluster labels for each node
        """
        # Add small diagonal for stability
        matrix = adjacency + np.eye(adjacency.shape[0]) * 0.01
        
        # Normalize columns
        col_sums = matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / col_sums
        
        # MCL iterations
        for iteration in range(self.max_iters):
            old_matrix = matrix.copy()
            
            # Expansion (matrix multiplication)
            matrix = np.linalg.matrix_power(matrix, 2)
            
            # Inflation
            matrix = np.power(matrix, self.inflation)
            
            # Normalize columns
            col_sums = matrix.sum(axis=0)
            col_sums[col_sums == 0] = 1
            matrix = matrix / col_sums
            
            # Check convergence
            if np.allclose(matrix, old_matrix, atol=1e-6):
                break
        
        # Extract clusters from final matrix
        return self._extract_clusters(matrix)
    
    def _extract_clusters(self, matrix: np.ndarray) -> np.ndarray:
        """Extract cluster labels from converged MCL matrix.
        
        Args:
            matrix: Converged MCL matrix
            
        Returns:
            Cluster labels
        """
        n_nodes = matrix.shape[0]
        labels = np.full(n_nodes, -1, dtype=int)
        
        # Find attractors (columns with non-zero values)
        attractors = []
        for col in range(n_nodes):
            if matrix[:, col].sum() > 1e-6:
                attractors.append(col)
        
        # Assign each node to its strongest attractor
        cluster_id = 0
        for attractor in attractors:
            # Find nodes attracted to this attractor
            attracted_nodes = np.where(matrix[:, attractor] > 1e-6)[0]
            
            for node in attracted_nodes:
                if labels[node] == -1:  # Not yet assigned
                    labels[node] = cluster_id
            
            if len(attracted_nodes) > 0:
                cluster_id += 1
        
        # Handle unassigned nodes (assign to their own clusters)
        for i in range(n_nodes):
            if labels[i] == -1:
                labels[i] = cluster_id
                cluster_id += 1
        
        return labels
    
    def get_cluster_summary(self) -> Dict:
        """Get clustering summary statistics.
        
        Returns:
            Summary dictionary with cluster information
        """
        if self.cluster_labels is None:
            return {"error": "Model not fitted yet"}
        
        unique_labels = np.unique(self.cluster_labels)
        n_clusters = len(unique_labels)
        
        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[int(label)] = int(np.sum(self.cluster_labels == label))
        
        return {
            "n_clusters": n_clusters,
            "n_samples": len(self.cluster_labels),
            "cluster_sizes": cluster_sizes,
            "largest_cluster": max(cluster_sizes.values()),
            "smallest_cluster": min(cluster_sizes.values())
        }


def estimate_clusters(embeddings: np.ndarray, k_range: Tuple[int, int] = (20, 100)) -> Dict:
    """Estimate optimal number of clusters using multiple k values.
    
    Args:
        embeddings: Input embeddings
        k_range: Range of k values to try (min_k, max_k)
        
    Returns:
        Dictionary with estimation results
    """
    k_min, k_max = k_range
    n_samples = embeddings.shape[0]
    
    # Adjust k range based on data size
    k_max = min(k_max, n_samples // 2)
    k_min = min(k_min, k_max)
    
    results = []
    k_values = np.linspace(k_min, k_max, 5, dtype=int)
    
    for k in k_values:
        try:
            mcl = MCLPipeline(inflation=2.0)
            mcl.fit(embeddings, k=k)
            summary = mcl.get_cluster_summary()
            
            results.append({
                "k": k,
                "n_clusters": summary["n_clusters"],
                "largest_cluster": summary["largest_cluster"],
                "smallest_cluster": summary["smallest_cluster"]
            })
        except Exception as e:
            print(f"Warning: k={k} failed: {e}")
            continue
    
    if not results:
        return {"error": "All k values failed", "estimated_clusters": n_samples // 10}
    
    # Simple heuristic: choose k that gives reasonable cluster count
    target_clusters = max(2, n_samples // 10)  # Rough target
    
    best_result = min(results, key=lambda x: abs(x["n_clusters"] - target_clusters))
    
    return {
        "estimated_clusters": best_result["n_clusters"],
        "recommended_k": best_result["k"],
        "all_results": results,
        "target_clusters": target_clusters
    }


def auto_train_mcl(embeddings: np.ndarray, search_iterations: int = 10, 
                  true_labels: np.ndarray = None) -> Dict:
    """Auto-train MCL with hyperparameter search.
    
    Args:
        embeddings: Input embeddings
        search_iterations: Number of parameter combinations to try
        true_labels: Ground truth labels for evaluation (optional)
        
    Returns:
        Dictionary with best parameters and results
    """
    n_samples = embeddings.shape[0]
    
    # Parameter search space
    inflation_values = np.linspace(1.2, 3.0, 5)
    k_values = np.linspace(20, min(100, n_samples // 2), 5, dtype=int)
    
    best_score = -1
    best_params = {}
    best_summary = {}
    best_evaluation = {}
    all_results = []
    
    iteration = 0
    for inflation in inflation_values:
        for k in k_values:
            if iteration >= search_iterations:
                break
                
            try:
                mcl = MCLPipeline(inflation=inflation)
                mcl.fit(embeddings, k=k)
                summary = mcl.get_cluster_summary()
                
                # Evaluate clustering quality
                evaluation_results = {}
                if true_labels is not None:
                    # Use true labels for evaluation with NMI/ARI
                    try:
                        evaluation_results = mcl_scoring_function(true_labels, mcl.cluster_labels)
                        # Use composite score (NMI + ARI) / 2 from metrics
                        metrics = evaluation_results.get('metrics', {})
                        nmi = metrics.get('nmi', 0)
                        ari = metrics.get('ari', 0)
                        total_score = (nmi + ari) / 2 if not np.isnan(nmi) and not np.isnan(ari) else -1
                    except Exception as e:
                        print(f"Warning: Evaluation failed for inflation={inflation}, k={k}: {e}")
                        total_score = -1
                        evaluation_results = {"error": str(e)}
                else:
                    # Use heuristic scoring when no true labels available
                    total_score = mcl_optimization_score(embeddings, mcl.cluster_labels)
                    evaluation_results = {
                        "heuristic_score": total_score,
                        "has_ground_truth": False
                    }
                
                result = {
                    "inflation": inflation,
                    "k": k,
                    "score": total_score,
                    "n_clusters": summary["n_clusters"],
                    "summary": summary,
                    "evaluation": evaluation_results
                }
                all_results.append(result)
                
                if total_score > best_score:
                    best_score = total_score
                    best_params = {"inflation": inflation, "k": k}
                    best_summary = summary
                    best_evaluation = evaluation_results
                
                iteration += 1
                
            except Exception as e:
                print(f"Warning: inflation={inflation}, k={k} failed: {e}")
                continue
    
    print(f"Auto-training completed: {iteration} parameter combinations tested")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    if best_evaluation:
        print(f"Best evaluation: {best_evaluation}")
    
    return {
        "best_parameters": best_params,
        "best_score": best_score,
        "best_summary": best_summary,
        "best_evaluation": best_evaluation,
        "all_results": all_results,
        "search_iterations": iteration
    }


def manual_train_mcl(embeddings: np.ndarray, inflation: float = 2.0, 
                    k: int = 50, max_iters: int = 100) -> Dict:
    """Train MCL with manual parameters.
    
    Args:
        embeddings: Input embeddings
        inflation: MCL inflation parameter
        k: Number of nearest neighbors
        max_iters: Maximum MCL iterations
        
    Returns:
        Dictionary with training results
    """
    try:
        mcl = MCLPipeline(inflation=inflation, max_iters=max_iters)
        mcl.fit(embeddings, k=k)
        summary = mcl.get_cluster_summary()
        
        return {
            "status": "success",
            "parameters": {
                "inflation": inflation,
                "k": k,
                "max_iters": max_iters
            },
            "cluster_labels": mcl.cluster_labels,
            "summary": summary
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "parameters": {
                "inflation": inflation,
                "k": k,
                "max_iters": max_iters
            }
        }