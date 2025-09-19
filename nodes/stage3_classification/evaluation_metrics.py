"""
Clustering evaluation metrics for Stage 3 MCL training.
Provides NMI and ARI scoring utilities.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from collections import Counter


def calculate_nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Calculate Normalized Mutual Information score.
    
    Args:
        labels_true: Ground truth cluster labels
        labels_pred: Predicted cluster labels
        
    Returns:
        NMI score between 0 and 1 (higher is better)
    """
    try:
        return normalized_mutual_info_score(labels_true, labels_pred)
    except Exception as e:
        print(f"Warning: NMI calculation failed: {e}")
        return 0.0


def calculate_ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Calculate Adjusted Rand Index score.
    
    Args:
        labels_true: Ground truth cluster labels
        labels_pred: Predicted cluster labels
        
    Returns:
        ARI score, typically between -1 and 1 (higher is better)
    """
    try:
        return adjusted_rand_score(labels_true, labels_pred)
    except Exception as e:
        print(f"Warning: ARI calculation failed: {e}")
        return 0.0


def calculate_clustering_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive clustering evaluation metrics.
    
    Args:
        labels_true: Ground truth cluster labels
        labels_pred: Predicted cluster labels
        
    Returns:
        Dictionary with NMI, ARI, and additional metrics
    """
    metrics = {
        'nmi': calculate_nmi(labels_true, labels_pred),
        'ari': calculate_ari(labels_true, labels_pred)
    }
    
    # Add basic clustering statistics
    metrics['n_clusters_true'] = len(np.unique(labels_true))
    metrics['n_clusters_pred'] = len(np.unique(labels_pred))
    metrics['n_samples'] = len(labels_true)
    
    # Add cluster size statistics
    true_counts = Counter(labels_true)
    pred_counts = Counter(labels_pred)
    
    metrics['avg_cluster_size_true'] = np.mean(list(true_counts.values()))
    metrics['avg_cluster_size_pred'] = np.mean(list(pred_counts.values()))
    metrics['max_cluster_size_true'] = max(true_counts.values())
    metrics['max_cluster_size_pred'] = max(pred_counts.values())
    metrics['min_cluster_size_true'] = min(true_counts.values())
    metrics['min_cluster_size_pred'] = min(pred_counts.values())
    
    return metrics


def evaluate_mcl_clustering(embeddings: np.ndarray, 
                          mcl_labels: np.ndarray,
                          ground_truth_labels: Optional[np.ndarray] = None,
                          metadata: Optional[Dict] = None) -> Dict[str, float]:
    """Evaluate MCL clustering results with multiple metrics.
    
    Args:
        embeddings: Original embeddings used for clustering
        mcl_labels: MCL cluster labels
        ground_truth_labels: True cluster labels (if available)
        metadata: Additional metadata about the clustering
        
    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        'n_samples': len(mcl_labels),
        'n_clusters': len(np.unique(mcl_labels)),
        'n_features': embeddings.shape[1] if embeddings is not None else 0
    }
    
    # Basic cluster statistics
    cluster_counts = Counter(mcl_labels)
    results['largest_cluster'] = max(cluster_counts.values())
    results['smallest_cluster'] = min(cluster_counts.values())
    results['avg_cluster_size'] = np.mean(list(cluster_counts.values()))
    results['cluster_size_std'] = np.std(list(cluster_counts.values()))
    
    # Calculate balance metric (how evenly distributed clusters are)
    n_clusters = len(cluster_counts)
    expected_size = len(mcl_labels) / n_clusters
    size_deviations = [abs(size - expected_size) for size in cluster_counts.values()]
    results['cluster_balance'] = 1.0 - (np.mean(size_deviations) / expected_size)
    
    # If ground truth is available, calculate supervised metrics
    if ground_truth_labels is not None:
        supervised_metrics = calculate_clustering_metrics(ground_truth_labels, mcl_labels)
        results.update(supervised_metrics)
        
        # Calculate overall score combining NMI and ARI
        results['combined_score'] = (supervised_metrics['nmi'] + supervised_metrics['ari']) / 2
    else:
        # Use unsupervised quality metrics
        results['silhouette_score'] = _calculate_silhouette_simple(embeddings, mcl_labels)
        results['combined_score'] = results['cluster_balance'] * 0.7 + results['silhouette_score'] * 0.3
    
    return results


def _calculate_silhouette_simple(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Simple silhouette score calculation for clustering quality.
    
    Args:
        embeddings: Data points
        labels: Cluster labels
        
    Returns:
        Simplified silhouette score
    """
    try:
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) > 1:
            return silhouette_score(embeddings, labels, metric='cosine')
        else:
            return 0.0
    except Exception as e:
        print(f"Warning: Silhouette calculation failed: {e}")
        return 0.0


def create_ground_truth_from_questions(metadata: Dict) -> Optional[np.ndarray]:
    """Create ground truth labels based on question IDs for evaluation.
    
    Args:
        metadata: Clustering metadata containing question mappings
        
    Returns:
        Ground truth labels or None if not possible
    """
    if 'original_dataframe' not in metadata:
        return None
    
    try:
        df = metadata['original_dataframe']
        if 'question_id' not in df.columns:
            return None
        
        # Create labels based on question IDs
        unique_questions = df['question_id'].unique()
        question_to_label = {qid: i for i, qid in enumerate(unique_questions)}
        
        # Map to long format if needed
        if metadata.get('format') == 'long':
            row_mapping = metadata.get('row_mapping', [])
            ground_truth = []
            
            for row_idx in row_mapping:
                question_id = df.iloc[row_idx]['question_id']
                ground_truth.append(question_to_label[question_id])
            
            return np.array(ground_truth)
        else:
            # Single format
            return np.array([question_to_label[qid] for qid in df['question_id']])
            
    except Exception as e:
        print(f"Warning: Could not create ground truth from questions: {e}")
        return None


def score_mcl_parameters(embeddings: np.ndarray, 
                        mcl_labels: np.ndarray,
                        parameters: Dict,
                        metadata: Optional[Dict] = None) -> float:
    """Score MCL clustering parameters for hyperparameter optimization.
    
    Args:
        embeddings: Original embeddings
        mcl_labels: MCL cluster labels
        parameters: MCL parameters used
        metadata: Clustering metadata
        
    Returns:
        Score between 0 and 1 (higher is better)
    """
    # Get ground truth if possible
    ground_truth = None
    if metadata:
        ground_truth = create_ground_truth_from_questions(metadata)
    
    # Evaluate clustering
    results = evaluate_mcl_clustering(embeddings, mcl_labels, ground_truth, metadata)
    
    # Return combined score
    return results.get('combined_score', 0.0)


def print_clustering_report(labels_true: Optional[np.ndarray], 
                          labels_pred: np.ndarray,
                          embeddings: Optional[np.ndarray] = None,
                          metadata: Optional[Dict] = None) -> None:
    """Print a comprehensive clustering evaluation report.
    
    Args:
        labels_true: Ground truth labels (optional)
        labels_pred: Predicted cluster labels  
        embeddings: Original embeddings (optional)
        metadata: Clustering metadata (optional)
    """
    print("\n📊 Clustering Evaluation Report")
    print("=" * 50)
    
    # Basic statistics
    n_samples = len(labels_pred)
    n_clusters = len(np.unique(labels_pred))
    
    print(f"📈 Basic Statistics:")
    print(f"   Samples: {n_samples}")
    print(f"   Clusters: {n_clusters}")
    
    if embeddings is not None:
        print(f"   Features: {embeddings.shape[1]}")
    
    # Cluster size distribution
    cluster_counts = Counter(labels_pred)
    print(f"\n🎯 Cluster Sizes:")
    print(f"   Largest: {max(cluster_counts.values())}")
    print(f"   Smallest: {min(cluster_counts.values())}")
    print(f"   Average: {np.mean(list(cluster_counts.values())):.1f}")
    print(f"   Std Dev: {np.std(list(cluster_counts.values())):.1f}")
    
    # Supervised metrics (if ground truth available)
    if labels_true is not None:
        nmi = calculate_nmi(labels_true, labels_pred)
        ari = calculate_ari(labels_true, labels_pred)
        
        print(f"\n🎯 Supervised Metrics:")
        print(f"   NMI: {nmi:.4f}")
        print(f"   ARI: {ari:.4f}")
        print(f"   Combined: {(nmi + ari)/2:.4f}")
        
        print(f"\n📋 Ground Truth vs Predicted:")
        print(f"   True clusters: {len(np.unique(labels_true))}")
        print(f"   Pred clusters: {n_clusters}")
    
    # Unsupervised metrics
    if embeddings is not None:
        silhouette = _calculate_silhouette_simple(embeddings, labels_pred)
        print(f"\n📊 Unsupervised Metrics:")
        print(f"   Silhouette: {silhouette:.4f}")
    
    print("=" * 50)