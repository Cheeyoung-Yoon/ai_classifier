"""
Stage 3 Phase 2: Secondary Labeling Node
Implements graph-based community detection for topic/semantic integration.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logger.warning("python-louvain not available. Install with: pip install python-louvain")

try:
    import leidenalg
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    logger.warning("leidenalg not available. Install with: pip install leidenalg python-igraph")


class Phase2SecondaryLabeling:
    """
    Phase 2 Secondary Labeling: Topic/Semantic Integration
    
    Pipeline:
    1. Input: Phase 1 groups (prototypes + metadata)
    2. Graph construction with weighted edges (α·embedding + β·co-context + γ·keyword)
    3. Community detection (Louvain/Leiden)
    4. Labeling options (Human-in-the-loop / LLM assisted)
    5. Output: LabelV2 format with feedback loop
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Phase 2 labeling with configuration."""
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Phase 2."""
        return {
            # Graph construction
            "edge_weights": {
                "alpha": 0.5,  # embedding similarity weight
                "beta": 0.3,   # co-context weight  
                "gamma": 0.2   # keyword overlap weight
            },
            "top_m_edges": 10,  # Keep top-m edges
            "edge_threshold": 0.1,  # Remove edges below threshold
            
            # Community detection
            "algorithm": "louvain",  # louvain/leiden
            "resolution": 1.0,  # Resolution parameter for cluster count control
            "random_seed": 42,
            "n_iterations": 10,  # For leiden
            
            # Constraints
            "must_links": [],  # [(group_id1, group_id2), ...]
            "cannot_links": [],  # [(group_id1, group_id2), ...]
            
            # Labeling mode
            "labeling_mode": "llm_assisted",  # human_in_loop/llm_assisted
            
            # LLM configuration for label generation
            "llm_config": {
                "model": "gpt-3.5-turbo",
                "max_examples": 5,  # Max positive examples per label
                "generate_definitions": True,
                "generate_rules": True
            }
        }
    
    def process_phase1_groups(
        self,
        phase1_groups: Dict[str, Any],
        phase1_prototypes: Dict[str, Any],
        phase1_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process Phase 1 groups through Phase 2 pipeline.
        
        Args:
            phase1_groups: Phase 1 clustering results
            phase1_prototypes: Representative sentences/medoids
            phase1_metadata: Group metadata (size, keywords, etc.)
            
        Returns:
            Dictionary with Phase 2 results in LabelV2 format
        """
        start_time = time.time()
        
        try:
            logger.info("Starting Phase 2 secondary labeling")
            
            # Step 1: Prepare group data
            logger.info("Step 1: Preparing group data...")
            group_data = self._prepare_group_data(
                phase1_groups, phase1_prototypes, phase1_metadata
            )
            
            if len(group_data) < 2:
                logger.warning("Not enough groups for Phase 2 processing")
                return self._create_single_group_result(group_data)
            
            # Step 2: Build weighted graph
            logger.info("Step 2: Building weighted graph...")
            graph = self._build_weighted_graph(group_data)
            
            # Step 3: Apply constraints
            logger.info("Step 3: Applying constraints...")
            constrained_graph = self._apply_constraints(graph, group_data)
            
            # Step 4: Community detection
            logger.info("Step 4: Running community detection...")
            communities = self._detect_communities(constrained_graph)
            
            # Step 5: Generate labels
            logger.info("Step 5: Generating labels...")
            labels = self._generate_labels(communities, group_data)
            
            processing_time = time.time() - start_time
            
            # Prepare results
            results = {
                "status": "completed",
                "labeling_mode": self.config["labeling_mode"],
                "algorithm": self.config["algorithm"],
                "resolution": self.config["resolution"],
                "n_labels": len(labels),
                "labels": labels,
                "communities": communities,
                "processing_time": processing_time,
                "config": self.config.copy()
            }
            
            logger.info(f"✅ Phase 2 completed: {len(labels)} labels generated in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Phase 2 processing failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "status": "failed",
                "error": error_msg,
                "processing_time": processing_time
            }
    
    def _prepare_group_data(
        self,
        phase1_groups: Dict[str, Any],
        phase1_prototypes: Dict[str, Any], 
        phase1_metadata: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Prepare group data for Phase 2 processing."""
        group_data = {}
        
        for question_id in phase1_groups.keys():
            if question_id not in phase1_prototypes or question_id not in phase1_metadata:
                continue
                
            question_prototypes = phase1_prototypes[question_id]
            question_metadata = phase1_metadata[question_id]
            
            for group_key in question_prototypes.keys():
                # Create unique group ID across questions
                unique_group_id = f"{question_id}_{group_key}"
                
                prototype_info = question_prototypes[group_key]
                metadata_info = question_metadata.get(group_key, {})
                
                group_data[unique_group_id] = {
                    "question_id": question_id,
                    "original_group_key": group_key,
                    "prototype_text": prototype_info.get("text", ""),
                    "prototype_embedding": np.array(prototype_info.get("embedding", [])),
                    "cluster_size": prototype_info.get("cluster_size", 1),
                    "keywords": metadata_info.get("keywords", []),
                    "avg_similarity": metadata_info.get("avg_similarity", 1.0),
                    "member_indices": metadata_info.get("member_indices", []),
                    "texts": metadata_info.get("texts", [])
                }
        
        logger.info(f"Prepared {len(group_data)} groups for Phase 2")
        return group_data
    
    def _build_weighted_graph(self, group_data: Dict[str, Dict[str, Any]]) -> nx.Graph:
        """Build weighted graph between groups."""
        G = nx.Graph()
        
        # Add nodes
        for group_id, data in group_data.items():
            G.add_node(group_id, **data)
        
        # Add weighted edges
        group_ids = list(group_data.keys())
        weights = self.config["edge_weights"]
        
        for i, group_id1 in enumerate(group_ids):
            for j, group_id2 in enumerate(group_ids[i + 1:], i + 1):
                
                # Calculate edge weight components
                embedding_sim = self._compute_embedding_similarity(
                    group_data[group_id1], group_data[group_id2]
                )
                
                co_context_sim = self._compute_co_context_similarity(
                    group_data[group_id1], group_data[group_id2]
                )
                
                keyword_sim = self._compute_keyword_similarity(
                    group_data[group_id1], group_data[group_id2]
                )
                
                # Weighted combination
                total_weight = (
                    weights["alpha"] * embedding_sim + 
                    weights["beta"] * co_context_sim + 
                    weights["gamma"] * keyword_sim
                )
                
                # Add edge if above threshold
                if total_weight > self.config["edge_threshold"]:
                    G.add_edge(group_id1, group_id2, weight=total_weight)
        
        # Keep only top-m edges per node
        self._prune_graph_edges(G)
        
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _compute_embedding_similarity(
        self, 
        group1: Dict[str, Any], 
        group2: Dict[str, Any]
    ) -> float:
        """Compute embedding similarity between group prototypes."""
        emb1 = group1["prototype_embedding"]
        emb2 = group2["prototype_embedding"]
        
        if emb1.size == 0 or emb2.size == 0:
            return 0.0
        
        # Cosine similarity
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def _compute_co_context_similarity(
        self,
        group1: Dict[str, Any],
        group2: Dict[str, Any]
    ) -> float:
        """Compute co-context similarity (same question bonus)."""
        # Simple co-context: groups from same question have higher similarity
        if group1["question_id"] == group2["question_id"]:
            return 1.0
        else:
            return 0.0
    
    def _compute_keyword_similarity(
        self,
        group1: Dict[str, Any],
        group2: Dict[str, Any]
    ) -> float:
        """Compute keyword overlap similarity."""
        keywords1 = set(group1.get("keywords", []))
        keywords2 = set(group2.get("keywords", []))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    def _prune_graph_edges(self, G: nx.Graph) -> None:
        """Keep only top-m edges per node."""
        top_m = self.config["top_m_edges"]
        
        edges_to_remove = []
        
        for node in G.nodes():
            # Get edges for this node
            edges = [(node, neighbor, G[node][neighbor]["weight"]) 
                    for neighbor in G.neighbors(node)]
            
            if len(edges) > top_m:
                # Sort by weight and keep only top-m
                edges.sort(key=lambda x: x[2], reverse=True)
                edges_to_keep = set()
                
                for i in range(min(top_m, len(edges))):
                    edge = (edges[i][0], edges[i][1])
                    edges_to_keep.add(edge)
                    edges_to_keep.add((edge[1], edge[0]))  # Undirected
                
                # Mark other edges for removal
                for edge in edges[top_m:]:
                    edge_tuple = (edge[0], edge[1])
                    if edge_tuple not in edges_to_keep:
                        edges_to_remove.append(edge_tuple)
        
        # Remove edges
        for edge in edges_to_remove:
            if G.has_edge(edge[0], edge[1]):
                G.remove_edge(edge[0], edge[1])
    
    def _apply_constraints(
        self, 
        G: nx.Graph, 
        group_data: Dict[str, Dict[str, Any]]
    ) -> nx.Graph:
        """Apply must-link and cannot-link constraints."""
        constrained_G = G.copy()
        
        # Apply must-links (force connections)
        for link in self.config["must_links"]:
            if len(link) >= 2 and link[0] in G.nodes() and link[1] in G.nodes():
                # Add edge with maximum weight
                constrained_G.add_edge(link[0], link[1], weight=1.0)
        
        # Apply cannot-links (remove connections)
        for link in self.config["cannot_links"]:
            if len(link) >= 2 and constrained_G.has_edge(link[0], link[1]):
                constrained_G.remove_edge(link[0], link[1])
        
        return constrained_G
    
    def _detect_communities(self, G: nx.Graph) -> Dict[str, List[str]]:
        """Detect communities using configured algorithm."""
        algorithm = self.config["algorithm"]
        resolution = self.config["resolution"]
        
        if algorithm == "louvain" and LOUVAIN_AVAILABLE:
            return self._run_louvain(G, resolution)
        elif algorithm == "leiden" and LEIDEN_AVAILABLE:
            return self._run_leiden(G, resolution)
        else:
            # Fallback to simple connected components
            logger.warning(f"Algorithm {algorithm} not available, using connected components")
            return self._run_connected_components(G)
    
    def _run_louvain(self, G: nx.Graph, resolution: float) -> Dict[str, List[str]]:
        """Run Louvain community detection."""
        partition = community_louvain.best_partition(G, resolution=resolution, random_state=self.config["random_seed"])
        
        # Convert to community format
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[f"community_{community_id}"].append(node)
        
        return dict(communities)
    
    def _run_leiden(self, G: nx.Graph, resolution: float) -> Dict[str, List[str]]:
        """Run Leiden community detection."""
        # Convert to igraph
        ig_graph = ig.Graph.from_networkx(G)
        
        # Run Leiden
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            n_iterations=self.config["n_iterations"],
            seed=self.config["random_seed"]
        )
        
        # Convert back to community format
        communities = defaultdict(list)
        for i, community_id in enumerate(partition.membership):
            node_name = G.nodes()[i]  # Get original node name
            communities[f"community_{community_id}"].append(node_name)
        
        return dict(communities)
    
    def _run_connected_components(self, G: nx.Graph) -> Dict[str, List[str]]:
        """Fallback: use connected components."""
        communities = {}
        for i, component in enumerate(nx.connected_components(G)):
            communities[f"component_{i}"] = list(component)
        
        return communities
    
    def _generate_labels(
        self,
        communities: Dict[str, List[str]], 
        group_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate labels based on labeling mode."""
        mode = self.config["labeling_mode"]
        
        if mode == "llm_assisted":
            return self._generate_llm_labels(communities, group_data)
        elif mode == "human_in_loop":
            return self._generate_human_labels(communities, group_data)
        else:
            return self._generate_basic_labels(communities, group_data)
    
    def _generate_basic_labels(
        self,
        communities: Dict[str, List[str]],
        group_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate basic labels without LLM assistance."""
        labels = {}
        
        for community_id, group_ids in communities.items():
            # Collect all keywords from community groups
            all_keywords = []
            all_texts = []
            total_size = 0
            
            for group_id in group_ids:
                if group_id in group_data:
                    data = group_data[group_id]
                    all_keywords.extend(data.get("keywords", []))
                    all_texts.extend(data.get("texts", []))
                    total_size += data.get("cluster_size", 1)
            
            # Generate label name from most common keywords
            if all_keywords:
                from collections import Counter
                keyword_counts = Counter(all_keywords)
                top_keywords = [word for word, count in keyword_counts.most_common(3)]
                label_name = "_".join(top_keywords)
            else:
                label_name = f"label_{community_id}"
            
            # Create basic label definition
            label_definition = f"Cluster containing {len(group_ids)} groups with {total_size} total samples"
            
            # Get positive examples (sample texts)
            positive_examples = all_texts[:min(10, len(all_texts))]
            
            labels[community_id] = {
                "name": label_name,
                "definition": label_definition,
                "positive_examples": positive_examples,
                "negative_examples": [],  # Would need to be populated separately
                "group_ids": group_ids,
                "size": total_size,
                "keywords": list(set(all_keywords)),
                "decision_rules": {
                    "include_keywords": top_keywords[:3] if all_keywords else [],
                    "exclude_keywords": [],
                    "regex_patterns": []
                },
                "metadata": {
                    "creation_method": "basic_generation",
                    "confidence": 0.5
                }
            }
        
        return labels
    
    def _generate_llm_labels(
        self,
        communities: Dict[str, List[str]],
        group_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate labels using LLM assistance."""
        # This would integrate with an LLM service
        # For now, return enhanced basic labels with placeholder for LLM integration
        
        labels = self._generate_basic_labels(communities, group_data)
        
        # Enhance with LLM-like features
        for community_id, label_data in labels.items():
            # Placeholder for LLM enhancement
            label_data["llm_suggested_name"] = f"LLM_enhanced_{label_data['name']}"
            label_data["llm_definition"] = f"LLM-enhanced definition: {label_data['definition']}"
            label_data["metadata"]["creation_method"] = "llm_assisted"
            label_data["metadata"]["llm_model"] = self.config["llm_config"]["model"]
            label_data["metadata"]["confidence"] = 0.8
        
        return labels
    
    def _generate_human_labels(
        self,
        communities: Dict[str, List[str]],
        group_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate labels for human-in-the-loop mode."""
        labels = self._generate_basic_labels(communities, group_data)
        
        # Add human interaction placeholders
        for community_id, label_data in labels.items():
            label_data["human_review_needed"] = True
            label_data["suggested_actions"] = ["review_name", "review_definition", "review_examples"]
            label_data["metadata"]["creation_method"] = "human_in_loop"
            label_data["metadata"]["review_status"] = "pending"
        
        return labels
    
    def _create_single_group_result(self, group_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Handle case with only one group."""
        if not group_data:
            return {
                "status": "failed",
                "error": "No groups available for Phase 2 processing"
            }
        
        # Create single label for the only group
        group_id = list(group_data.keys())[0]
        data = group_data[group_id]
        
        label = {
            "single_group": {
                "name": "single_cluster",
                "definition": "Single cluster from Phase 1",
                "positive_examples": data.get("texts", [])[:5],
                "negative_examples": [],
                "group_ids": [group_id],
                "size": data.get("cluster_size", 1),
                "keywords": data.get("keywords", []),
                "decision_rules": {
                    "include_keywords": data.get("keywords", [])[:3],
                    "exclude_keywords": [],
                    "regex_patterns": []
                },
                "metadata": {
                    "creation_method": "single_group",
                    "confidence": 1.0
                }
            }
        }
        
        return {
            "status": "completed",
            "labeling_mode": "single_group",
            "n_labels": 1,
            "labels": label,
            "processing_time": 0.0
        }


def phase2_secondary_labeling_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for Phase 2 Secondary Labeling.
    
    Args:
        state: LangGraph state containing Phase 1 results
        
    Returns:
        Updated state with phase2_* fields populated
    """
    logger.info("🚀 Starting Stage 3 Phase 2: Secondary Labeling")
    
    try:
        # Set current phase
        state["stage3_current_phase"] = "phase2_labeling"
        
        # Check Phase 1 completion
        if state.get("stage3_phase1_status") != "completed":
            return {
                **state,
                "stage3_phase2_status": "failed",
                "stage3_error": "Phase 1 not completed successfully"
            }
        
        # Extract Phase 1 results
        phase1_groups = state.get("stage3_phase1_groups", {})
        phase1_prototypes = state.get("stage3_phase1_prototypes", {})
        phase1_metadata = state.get("stage3_phase1_metadata", {})
        
        if not phase1_groups or not phase1_prototypes or not phase1_metadata:
            return {
                **state,
                "stage3_phase2_status": "failed",
                "stage3_error": "Phase 1 results incomplete"
            }
        
        # Get configuration from state with complete defaults
        config = {
            "algorithm": state.get("stage3_phase2_algorithm", "louvain"),
            "resolution": state.get("stage3_phase2_resolution", 1.0),
            "labeling_mode": state.get("stage3_phase2_mode", "basic"),
            "edge_weights": state.get("stage3_phase2_edge_weights", {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}),
            "edge_threshold": state.get("stage3_phase2_edge_threshold", 0.1),
            "top_m_edges": state.get("stage3_phase2_top_m_edges", 10),
            "must_links": state.get("stage3_phase2_must_links", []),
            "cannot_links": state.get("stage3_phase2_cannot_links", []),
            "random_seed": state.get("stage3_phase2_random_seed", 42),
            "n_iterations": state.get("stage3_phase2_n_iterations", 10),
            "llm_config": {"model": "gpt-3.5-turbo"}
        }
        
        # Initialize Phase 2 processor
        phase2_processor = Phase2SecondaryLabeling(config)
        
        # Process Phase 1 groups through Phase 2
        result = phase2_processor.process_phase1_groups(
            phase1_groups, phase1_prototypes, phase1_metadata
        )
        
        # Update state with Phase 2 results
        updated_state = {
            **state,
            "stage3_phase2_status": result["status"],
            "stage3_phase2_labels": result.get("labels", {}),
            "stage3_phase2_feedback": {},  # Initialize empty feedback
            "stage3_phase2_llm_suggestions": result.get("labels", {}) if config["labeling_mode"] == "llm_assisted" else {}
        }
        
        if result["status"] == "failed":
            updated_state["stage3_error"] = result.get("error", "Phase 2 processing failed")
        
        # Update overall stage 3 status
        if result["status"] == "completed":
            updated_state["stage3_status"] = "completed"
            logger.info(f"✅ Stage 3 completed: {result.get('n_labels', 0)} labels generated")
        else:
            updated_state["stage3_status"] = "failed"
        
        return updated_state
        
    except Exception as e:
        error_msg = f"Phase 2 secondary labeling failed: {str(e)}"
        logger.error(error_msg)
        
        return {
            **state,
            "stage3_phase2_status": "failed",
            "stage3_status": "failed",
            "stage3_error": error_msg
        }