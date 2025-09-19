"""
Stage 3 Phase 1: Primary Labeling Node
Implements kNN → CSLS → MCL pipeline with singleton support.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import markov_clustering as mc

logger = logging.getLogger(__name__)


class Phase1PrimaryLabeling:
    """
    Phase 1 Primary Labeling: kNN → CSLS → MCL (with singletons allowed)
    
    Pipeline:
    1. Text normalization and embedding
    2. kNN graph construction with ANN (HNSW/FAISS)
    3. CSLS scoring and edge pruning 
    4. MCL clustering with singleton support
    5. Quality assessment and prototype extraction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Phase 1 labeling with configuration."""
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Phase 1."""
        return {
            # kNN parameters
            "knn_k": 50,  # k=30~80 후보 검색
            "knn_metric": "cosine",
            "mutual_knn": True,  # mutual-kNN for hub mitigation
            
            # CSLS parameters  
            "csls_neighborhood_size": 10,  # for local average similarity
            "csls_threshold": 0.1,  # τ_edge threshold for edge removal
            "top_m_edges": 20,  # top-m (15~25) edges to keep per node
            "prune_bottom_percentile": 30,  # remove bottom 30% edges
            
            # MCL parameters
            "mcl_inflation": 2.0,  # r ∈ {1.6,1.8,2.0,2.2}
            "mcl_expansion": 2,
            "mcl_pruning": 1e-3,
            "mcl_max_iters": 100,
            "mcl_convergence_check": True,
            
            # Singleton and small cluster handling
            "allow_singletons": True,
            "merge_small_clusters": True,
            "min_cluster_size": 2,
            "small_cluster_threshold": 3,
            
            # Quality assessment
            "compute_subset_score": True,
            "compute_cluster_quality": True
        }
    
    def process_embeddings(
        self, 
        embeddings: np.ndarray,
        texts: List[str] = None,
        ground_truth_labels: List[int] = None
    ) -> Dict[str, Any]:
        """
        Process embeddings through Phase 1 pipeline.
        
        Args:
            embeddings: L2-normalized embeddings (n_samples, n_features)
            texts: Original text data (optional, for prototype extraction)
            ground_truth_labels: Ground truth for evaluation (optional)
            
        Returns:
            Dictionary with Phase 1 results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting Phase 1 primary labeling for {embeddings.shape[0]} samples")
            
            # Step 1: Build kNN graph
            logger.info("Step 1: Building kNN graph...")
            knn_graph = self._build_knn_graph(embeddings)
            
            # Step 2: Compute CSLS scores
            logger.info("Step 2: Computing CSLS scores...")
            csls_graph = self._compute_csls_scores(embeddings, knn_graph)
            
            # Step 3: Prune and scale edges
            logger.info("Step 3: Pruning and scaling edges...")
            final_graph = self._prune_and_scale_edges(csls_graph)
            
            # Step 4: Run MCL clustering
            logger.info("Step 4: Running MCL clustering...")
            cluster_labels = self._run_mcl_clustering(final_graph)
            
            # Step 5: Post-process clusters
            logger.info("Step 5: Post-processing clusters...")
            processed_labels = self._post_process_clusters(
                cluster_labels, embeddings, final_graph
            )
            
            # Step 6: Extract prototypes and metadata
            logger.info("Step 6: Extracting prototypes and metadata...")
            prototypes = self._extract_prototypes(
                processed_labels, embeddings, texts
            )
            metadata = self._compute_metadata(
                processed_labels, embeddings, texts
            )
            
            # Step 7: Quality assessment
            logger.info("Step 7: Computing quality statistics...")
            quality_stats = self._compute_quality_stats(
                processed_labels, embeddings, final_graph, ground_truth_labels
            )
            
            processing_time = time.time() - start_time
            
            # Prepare results
            results = {
                "status": "completed",
                "cluster_labels": processed_labels.tolist(),
                "n_clusters": len(np.unique(processed_labels[processed_labels >= 0])),
                "n_singletons": np.sum(processed_labels == -1),
                "n_samples": len(embeddings),
                "prototypes": prototypes,
                "metadata": metadata,
                "quality_stats": quality_stats,
                "processing_time": processing_time,
                "config": self.config.copy()
            }
            
            logger.info(f"✅ Phase 1 completed: {results['n_clusters']} clusters, "
                       f"{results['n_singletons']} singletons in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Phase 1 processing failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "status": "failed",
                "error": error_msg,
                "processing_time": processing_time
            }
    
    def _build_knn_graph(self, embeddings: np.ndarray) -> np.ndarray:
        """Build kNN graph with mutual-kNN option."""
        n_samples = embeddings.shape[0]
        k = min(self.config["knn_k"], n_samples - 1)
        
        # Use sklearn NearestNeighbors for ANN
        nn = NearestNeighbors(
            n_neighbors=k + 1,  # +1 because it includes self
            metric=self.config["knn_metric"],
            algorithm='auto'
        )
        nn.fit(embeddings)
        
        # Get distances and indices
        distances, indices = nn.kneighbors(embeddings)
        
        # Remove self (first neighbor)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        # Build adjacency matrix
        adjacency = np.zeros((n_samples, n_samples), dtype=np.float32)
        
        for i in range(n_samples):
            for j, neighbor_idx in enumerate(indices[i]):
                if neighbor_idx < n_samples:
                    # Convert distance to similarity (cosine)
                    similarity = 1.0 - distances[i, j]
                    adjacency[i, neighbor_idx] = similarity
        
        # Apply mutual-kNN if configured
        if self.config["mutual_knn"]:
            adjacency = (adjacency + adjacency.T) / 2
            # Keep only mutual connections
            adjacency = adjacency * (adjacency == adjacency.T)
            
        return adjacency
    
    def _compute_csls_scores(
        self, 
        embeddings: np.ndarray, 
        knn_graph: np.ndarray
    ) -> np.ndarray:
        """Compute CSLS scores for edge re-weighting."""
        n_samples = embeddings.shape[0]
        k_csls = self.config["csls_neighborhood_size"]
        
        # Compute local average similarities r_i for each point
        r_values = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Get top-k neighbors for point i
            neighbors = np.argsort(knn_graph[i])[-k_csls:]
            if len(neighbors) > 0:
                r_values[i] = np.mean(knn_graph[i, neighbors])
        
        # Compute CSLS scores: CSLS(i,j) = 2*cos(i,j) - r_i - r_j
        csls_graph = np.zeros_like(knn_graph)
        
        for i in range(n_samples):
            for j in range(n_samples):
                if knn_graph[i, j] > 0:  # Only for existing edges
                    cos_sim = knn_graph[i, j]
                    csls_score = 2 * cos_sim - r_values[i] - r_values[j]
                    # Remove negative scores
                    csls_graph[i, j] = max(0, csls_score)
        
        return csls_graph
    
    def _prune_and_scale_edges(self, csls_graph: np.ndarray) -> csr_matrix:
        """Prune and scale edges based on configuration."""
        n_samples = csls_graph.shape[0]
        top_m = self.config["top_m_edges"]
        threshold = self.config["csls_threshold"]
        prune_percentile = self.config["prune_bottom_percentile"]
        
        # Apply threshold filter
        filtered_graph = csls_graph.copy()
        filtered_graph[filtered_graph < threshold] = 0
        
        # Keep only top-m edges per node
        for i in range(n_samples):
            row = filtered_graph[i]
            if np.sum(row > 0) > top_m:
                # Keep only top-m values
                top_indices = np.argsort(row)[-top_m:]
                new_row = np.zeros_like(row)
                new_row[top_indices] = row[top_indices]
                filtered_graph[i] = new_row
        
        # Remove bottom percentile globally
        if prune_percentile > 0:
            nonzero_values = filtered_graph[filtered_graph > 0]
            if len(nonzero_values) > 0:
                cutoff = np.percentile(nonzero_values, prune_percentile)
                filtered_graph[filtered_graph < cutoff] = 0
        
        # Convert to sparse matrix for MCL
        return csr_matrix(filtered_graph)
    
    def _run_mcl_clustering(self, graph: csr_matrix) -> np.ndarray:
        """Run MCL clustering on the graph."""
        if graph.nnz == 0:
            # No edges - return all singletons
            return np.full(graph.shape[0], -1, dtype=int)
        
        # Run MCL with correct parameters
        result = mc.run_mcl(
            graph, 
            inflation=self.config["mcl_inflation"],
            expansion=self.config["mcl_expansion"],
            iterations=self.config["mcl_max_iters"]
        )
        
        # Get clusters
        clusters = mc.get_clusters(result)
        
        # Convert to labels array
        labels = np.full(graph.shape[0], -1, dtype=int)
        for cluster_id, cluster_nodes in enumerate(clusters):
            for node in cluster_nodes:
                if node < graph.shape[0]:
                    labels[node] = cluster_id
        
        return labels
    
    def _post_process_clusters(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray, 
        graph: csr_matrix
    ) -> np.ndarray:
        """Post-process clusters based on configuration."""
        processed_labels = labels.copy()
        
        if not self.config["merge_small_clusters"]:
            return processed_labels
        
        # Handle small clusters
        unique_labels, counts = np.unique(
            processed_labels[processed_labels >= 0], 
            return_counts=True
        )
        
        small_clusters = unique_labels[counts <= self.config["small_cluster_threshold"]]
        
        for small_cluster_id in small_clusters:
            cluster_nodes = np.where(processed_labels == small_cluster_id)[0]
            
            if len(cluster_nodes) >= self.config["min_cluster_size"]:
                continue  # Keep as is
            
            # Try to merge with nearest cluster
            best_target = self._find_nearest_cluster(
                cluster_nodes, processed_labels, embeddings, graph
            )
            
            if best_target is not None:
                processed_labels[cluster_nodes] = best_target
            elif not self.config["allow_singletons"]:
                # Force merge if singletons not allowed
                processed_labels[cluster_nodes] = self._force_merge_cluster(
                    cluster_nodes, processed_labels, embeddings
                )
        
        return processed_labels
    
    def _find_nearest_cluster(
        self,
        cluster_nodes: np.ndarray,
        labels: np.ndarray,
        embeddings: np.ndarray,
        graph: csr_matrix
    ) -> Optional[int]:
        """Find the nearest cluster for merging."""
        # Calculate average embedding of the small cluster
        cluster_center = np.mean(embeddings[cluster_nodes], axis=0)
        
        # Find connections to other clusters through the graph
        connected_clusters = {}
        for node in cluster_nodes:
            # Get neighbors from graph
            neighbors = graph[node].nonzero()[1]
            for neighbor in neighbors:
                neighbor_label = labels[neighbor]
                if neighbor_label >= 0 and neighbor_label != labels[node]:
                    if neighbor_label not in connected_clusters:
                        connected_clusters[neighbor_label] = []
                    connected_clusters[neighbor_label].append(graph[node, neighbor])
        
        if not connected_clusters:
            return None
        
        # Choose cluster with strongest connection
        best_cluster = max(
            connected_clusters.keys(),
            key=lambda x: np.mean(connected_clusters[x])
        )
        
        return best_cluster
    
    def _force_merge_cluster(
        self,
        cluster_nodes: np.ndarray,
        labels: np.ndarray,
        embeddings: np.ndarray
    ) -> int:
        """Force merge with the most similar cluster."""
        cluster_center = np.mean(embeddings[cluster_nodes], axis=0)
        
        # Calculate distances to all other cluster centers
        unique_labels = np.unique(labels[labels >= 0])
        best_distance = float('inf')
        best_cluster = 0
        
        for target_label in unique_labels:
            target_nodes = np.where(labels == target_label)[0]
            target_center = np.mean(embeddings[target_nodes], axis=0)
            
            # Cosine distance
            distance = 1.0 - np.dot(cluster_center, target_center) / (
                np.linalg.norm(cluster_center) * np.linalg.norm(target_center)
            )
            
            if distance < best_distance:
                best_distance = distance
                best_cluster = target_label
        
        return best_cluster
    
    def _extract_prototypes(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray,
        texts: List[str] = None
    ) -> Dict[str, Any]:
        """Extract representative sentences (medoids) for each cluster."""
        prototypes = {}
        unique_labels = np.unique(labels[labels >= 0])
        
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_embeddings = embeddings[cluster_indices]
            
            # Find medoid (most central point)
            if len(cluster_embeddings) == 1:
                medoid_idx = 0
            else:
                # Calculate pairwise distances within cluster
                distances = np.sum(
                    (cluster_embeddings[:, None] - cluster_embeddings[None, :]) ** 2,
                    axis=2
                )
                # Find point with minimum average distance to others
                medoid_idx = np.argmin(np.mean(distances, axis=1))
            
            global_medoid_idx = cluster_indices[medoid_idx]
            
            prototype_info = {
                "medoid_index": int(global_medoid_idx),
                "cluster_size": len(cluster_indices),
                "embedding": embeddings[global_medoid_idx].tolist()
            }
            
            if texts is not None:
                prototype_info["text"] = texts[global_medoid_idx]
            
            prototypes[f"cluster_{label}"] = prototype_info
        
        # Handle singletons
        singleton_indices = np.where(labels == -1)[0]
        for i, singleton_idx in enumerate(singleton_indices):
            prototype_info = {
                "medoid_index": int(singleton_idx),
                "cluster_size": 1,
                "embedding": embeddings[singleton_idx].tolist()
            }
            
            if texts is not None:
                prototype_info["text"] = texts[singleton_idx]
            
            prototypes[f"singleton_{i}"] = prototype_info
        
        return prototypes
    
    def _compute_metadata(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray,
        texts: List[str] = None
    ) -> Dict[str, Any]:
        """Compute metadata for each group (size, keywords, co-context)."""
        metadata = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            
            group_info = {
                "group_id": int(label),
                "size": len(cluster_indices),
                "member_indices": cluster_indices.tolist(),
            }
            
            if texts is not None:
                cluster_texts = [texts[i] for i in cluster_indices]
                group_info["texts"] = cluster_texts
                
                # Simple keyword extraction (most common words)
                all_words = []
                for text in cluster_texts:
                    # Simple tokenization
                    words = text.lower().split()
                    all_words.extend(words)
                
                if all_words:
                    from collections import Counter
                    word_counts = Counter(all_words)
                    # Get top 5 most common words as keywords
                    keywords = [word for word, count in word_counts.most_common(5)]
                    group_info["keywords"] = keywords
            
            # Compute intra-cluster similarity statistics
            if len(cluster_indices) > 1:
                cluster_embeddings = embeddings[cluster_indices]
                # Pairwise cosine similarities
                similarities = []
                for i in range(len(cluster_embeddings)):
                    for j in range(i + 1, len(cluster_embeddings)):
                        sim = np.dot(cluster_embeddings[i], cluster_embeddings[j])
                        similarities.append(sim)
                
                group_info["avg_similarity"] = float(np.mean(similarities))
                group_info["min_similarity"] = float(np.min(similarities))
                group_info["std_similarity"] = float(np.std(similarities))
            else:
                group_info["avg_similarity"] = 1.0
                group_info["min_similarity"] = 1.0
                group_info["std_similarity"] = 0.0
            
            key = f"cluster_{label}" if label >= 0 else f"singleton_{label}"
            metadata[key] = group_info
        
        return metadata
    
    def _compute_quality_stats(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray,
        graph: csr_matrix,
        ground_truth_labels: List[int] = None
    ) -> Dict[str, Any]:
        """Compute quality statistics for Phase 1 results."""
        stats = {}
        
        # Basic statistics
        unique_labels = np.unique(labels[labels >= 0])
        stats["n_clusters"] = len(unique_labels)
        stats["n_singletons"] = np.sum(labels == -1)
        stats["n_samples"] = len(labels)
        
        # Cluster size distribution
        if len(unique_labels) > 0:
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            stats["cluster_sizes"] = {
                "mean": float(np.mean(cluster_sizes)),
                "std": float(np.std(cluster_sizes)),
                "min": int(np.min(cluster_sizes)),
                "max": int(np.max(cluster_sizes)),
                "distribution": cluster_sizes
            }
        
        # Graph connectivity statistics
        stats["graph_stats"] = {
            "n_edges": graph.nnz,
            "density": graph.nnz / (graph.shape[0] ** 2),
            "avg_degree": graph.nnz / graph.shape[0]
        }
        
        # Compute SubsetScore if ground truth is available
        if self.config["compute_subset_score"] and ground_truth_labels is not None:
            subset_score = self._compute_subset_score(labels, ground_truth_labels)
            stats["subset_score"] = subset_score
        
        # Intra-cluster quality
        if self.config["compute_cluster_quality"]:
            quality_metrics = self._compute_cluster_quality(labels, embeddings)
            stats["cluster_quality"] = quality_metrics
        
        return stats
    
    def _compute_subset_score(
        self, 
        predicted_labels: np.ndarray, 
        ground_truth_labels: np.ndarray
    ) -> float:
        """
        Compute SubsetScore: percentage of predicted clusters that are 
        subsets of ground truth clusters.
        """
        if len(predicted_labels) != len(ground_truth_labels):
            return 0.0
        
        # Convert to lists for easier handling
        pred_labels = predicted_labels.tolist()
        true_labels = ground_truth_labels
        
        # Group by predicted clusters
        pred_clusters = {}
        for i, label in enumerate(pred_labels):
            if label >= 0:  # Skip singletons
                if label not in pred_clusters:
                    pred_clusters[label] = []
                pred_clusters[label].append(i)
        
        if not pred_clusters:
            return 0.0
        
        # Check subset condition for each predicted cluster
        subset_count = 0
        
        for pred_cluster_id, pred_indices in pred_clusters.items():
            # Get ground truth labels for this predicted cluster
            true_labels_in_cluster = [true_labels[i] for i in pred_indices]
            
            # Check if all points have the same ground truth label
            if len(set(true_labels_in_cluster)) == 1:
                subset_count += 1
        
        return subset_count / len(pred_clusters)
    
    def _compute_cluster_quality(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray
    ) -> Dict[str, float]:
        """Compute intra-cluster vs inter-cluster quality metrics."""
        unique_labels = np.unique(labels[labels >= 0])
        
        if len(unique_labels) < 2:
            return {"intra_cluster_similarity": 1.0, "inter_cluster_similarity": 0.0}
        
        # Compute intra-cluster similarities
        intra_similarities = []
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) > 1:
                cluster_embeddings = embeddings[cluster_indices]
                for i in range(len(cluster_embeddings)):
                    for j in range(i + 1, len(cluster_embeddings)):
                        sim = np.dot(cluster_embeddings[i], cluster_embeddings[j])
                        intra_similarities.append(sim)
        
        # Compute inter-cluster similarities
        inter_similarities = []
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels[i + 1:], i + 1):
                indices1 = np.where(labels == label1)[0]
                indices2 = np.where(labels == label2)[0]
                
                # Sample pairs to avoid quadratic computation
                max_pairs = 100
                n_pairs = min(max_pairs, len(indices1) * len(indices2))
                
                if n_pairs > 0:
                    # Random sampling of pairs
                    for _ in range(min(10, n_pairs)):
                        idx1 = np.random.choice(indices1)
                        idx2 = np.random.choice(indices2)
                        sim = np.dot(embeddings[idx1], embeddings[idx2])
                        inter_similarities.append(sim)
        
        return {
            "intra_cluster_similarity": float(np.mean(intra_similarities)) if intra_similarities else 1.0,
            "inter_cluster_similarity": float(np.mean(inter_similarities)) if inter_similarities else 0.0
        }


def phase1_primary_labeling_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for Phase 1 Primary Labeling.
    
    Args:
        state: LangGraph state containing matched_questions with embeddings
        
    Returns:
        Updated state with phase1_* fields populated
    """
    logger.info("🚀 Starting Stage 3 Phase 1: Primary Labeling")
    
    try:
        # Set current phase
        state["stage3_current_phase"] = "phase1_labeling"
        
        # Extract embeddings from state
        matched_questions = state.get("matched_questions", {})
        if not matched_questions:
            return {
                **state,
                "stage3_phase1_status": "failed",
                "stage3_error": "No matched_questions found in state"
            }
        
        # Get configuration from state
        config = {
            # kNN parameters
            "knn_k": state.get("stage3_phase1_knn_k", 50),
            "knn_metric": "cosine",
            "mutual_knn": True,
            
            # CSLS parameters
            "csls_neighborhood_size": 10,
            "csls_threshold": state.get("stage3_phase1_csls_threshold", 0.1),
            "top_m_edges": state.get("stage3_phase1_top_m", 20),
            "prune_bottom_percentile": 30,
            
            # MCL parameters
            "mcl_inflation": state.get("stage3_phase1_mcl_inflation", 2.0),
            "mcl_expansion": state.get("stage3_phase1_mcl_expansion", 2),
            "mcl_pruning": state.get("stage3_phase1_mcl_pruning", 1e-3),
            "mcl_max_iters": 100,
            "mcl_convergence_check": True,
            
            # Cluster handling
            "allow_singletons": True,
            "merge_small_clusters": True,
            "min_cluster_size": 2,
            "small_cluster_threshold": 3,
            
            # Quality assessment
            "compute_subset_score": True,
            "compute_cluster_quality": True
        }
        
        # Initialize Phase 1 processor
        phase1_processor = Phase1PrimaryLabeling(config)
        
        # Process each question
        phase1_results = {}
        all_prototypes = {}
        all_metadata = {}
        overall_quality_stats = {}
        
        for question_id, question_data in matched_questions.items():
            logger.info(f"Processing question {question_id} for Phase 1")
            
            # Extract embeddings and texts
            embeddings = np.array(question_data.get("embeddings", []))
            texts = question_data.get("texts", [])
            ground_truth = question_data.get("original_labels", None)
            
            if embeddings.size == 0:
                logger.warning(f"No embeddings found for question {question_id}")
                continue
            
            # Process through Phase 1 pipeline
            result = phase1_processor.process_embeddings(
                embeddings, texts, ground_truth
            )
            
            phase1_results[question_id] = result
            
            if result["status"] == "completed":
                all_prototypes[question_id] = result["prototypes"]
                all_metadata[question_id] = result["metadata"]
                
                # Aggregate quality stats
                if question_id not in overall_quality_stats:
                    overall_quality_stats[question_id] = result["quality_stats"]
        
        # Update state with Phase 1 results
        updated_state = {
            **state,
            "stage3_phase1_status": "completed",
            "stage3_phase1_groups": phase1_results,
            "stage3_phase1_prototypes": all_prototypes,
            "stage3_phase1_metadata": all_metadata,
            "stage3_phase1_quality_stats": overall_quality_stats
        }
        
        # Log completion
        total_questions = len(phase1_results)
        successful_questions = sum(1 for r in phase1_results.values() if r["status"] == "completed")
        
        logger.info(f"✅ Phase 1 completed: {successful_questions}/{total_questions} questions processed")
        
        return updated_state
        
    except Exception as e:
        error_msg = f"Phase 1 primary labeling failed: {str(e)}"
        logger.error(error_msg)
        
        return {
            **state,
            "stage3_phase1_status": "failed",
            "stage3_error": error_msg
        }