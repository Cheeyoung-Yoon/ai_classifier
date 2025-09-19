"""
Stage 3 Quality Assessment Tools
Implements SubsetScore, label consistency, and feedback loop mechanisms.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
import time

logger = logging.getLogger(__name__)


class QualityAssessmentTools:
    """
    Quality Assessment Tools for Stage 3 Two-Phase Labeling
    
    Includes:
    1. SubsetScore: Check mixing/purity of clusters
    2. Label consistency: Intra vs inter-cluster CSLS comparison  
    3. User modification tracking: merge/split/rename count
    4. Feedback loop management: must-link/cannot-link storage
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize quality assessment tools."""
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for quality assessment."""
        return {
            # SubsetScore parameters
            "subset_score_threshold": 0.7,  # Minimum acceptable subset score
            "allow_singletons_in_subset": True,
            
            # Label consistency parameters
            "consistency_sample_size": 100,  # Max samples for consistency check
            "intra_cluster_threshold": 0.5,  # Minimum intra-cluster similarity
            "inter_cluster_threshold": 0.3,  # Maximum inter-cluster similarity
            
            # Feedback tracking
            "track_user_modifications": True,
            "modification_penalty_weight": 0.1,  # Weight for modification count in quality score
            
            # Overall quality thresholds
            "quality_thresholds": {
                "excellent": 0.9,
                "good": 0.7,
                "acceptable": 0.5,
                "poor": 0.3
            }
        }
    
    def assess_phase1_quality(
        self,
        cluster_labels: np.ndarray,
        embeddings: np.ndarray,
        ground_truth_labels: Optional[np.ndarray] = None,
        texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Assess quality of Phase 1 primary labeling results.
        
        Args:
            cluster_labels: Predicted cluster labels from Phase 1
            embeddings: Original embeddings
            ground_truth_labels: Ground truth labels (optional)
            texts: Original text data (optional)
            
        Returns:
            Dictionary with Phase 1 quality metrics
        """
        start_time = time.time()
        
        try:
            logger.info("Assessing Phase 1 quality...")
            
            quality_metrics = {}
            
            # Basic cluster statistics
            quality_metrics["basic_stats"] = self._compute_basic_stats(cluster_labels)
            
            # SubsetScore if ground truth available
            if ground_truth_labels is not None:
                quality_metrics["subset_score"] = self._compute_subset_score(
                    cluster_labels, ground_truth_labels
                )
            
            # Intra-cluster consistency
            quality_metrics["cluster_consistency"] = self._compute_cluster_consistency(
                cluster_labels, embeddings
            )
            
            # Silhouette-like scores
            quality_metrics["separation_scores"] = self._compute_separation_scores(
                cluster_labels, embeddings
            )
            
            # Overall quality assessment
            quality_metrics["overall_quality"] = self._assess_overall_quality(quality_metrics)
            
            quality_metrics["assessment_time"] = time.time() - start_time
            
            logger.info(f"✅ Phase 1 quality assessment completed in {quality_metrics['assessment_time']:.2f}s")
            
            return quality_metrics
            
        except Exception as e:
            error_msg = f"Phase 1 quality assessment failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "assessment_time": time.time() - start_time}
    
    def assess_phase2_quality(
        self,
        phase2_labels: Dict[str, Any],
        phase1_groups: Dict[str, Any],
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess quality of Phase 2 secondary labeling results.
        
        Args:
            phase2_labels: LabelV2 format labels from Phase 2
            phase1_groups: Phase 1 group results
            user_feedback: User modifications and feedback (optional)
            
        Returns:
            Dictionary with Phase 2 quality metrics
        """
        start_time = time.time()
        
        try:
            logger.info("Assessing Phase 2 quality...")
            
            quality_metrics = {}
            
            # Label quality assessment
            quality_metrics["label_quality"] = self._assess_label_quality(phase2_labels)
            
            # Coverage assessment (how well labels cover Phase 1 groups)
            quality_metrics["coverage"] = self._assess_coverage(phase2_labels, phase1_groups)
            
            # User modification tracking
            if user_feedback is not None:
                quality_metrics["user_modifications"] = self._track_user_modifications(user_feedback)
            
            # Label consistency across groups
            quality_metrics["label_consistency"] = self._assess_label_consistency(phase2_labels)
            
            # Overall Phase 2 quality
            quality_metrics["overall_quality"] = self._assess_phase2_overall_quality(quality_metrics)
            
            quality_metrics["assessment_time"] = time.time() - start_time
            
            logger.info(f"✅ Phase 2 quality assessment completed in {quality_metrics['assessment_time']:.2f}s")
            
            return quality_metrics
            
        except Exception as e:
            error_msg = f"Phase 2 quality assessment failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "assessment_time": time.time() - start_time}
    
    def _compute_basic_stats(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Compute basic clustering statistics."""
        unique_labels = np.unique(cluster_labels)
        non_singleton_labels = unique_labels[unique_labels >= 0]
        
        stats = {
            "n_samples": len(cluster_labels),
            "n_clusters": len(non_singleton_labels),
            "n_singletons": np.sum(cluster_labels == -1),
            "singleton_ratio": np.sum(cluster_labels == -1) / len(cluster_labels)
        }
        
        if len(non_singleton_labels) > 0:
            cluster_sizes = [np.sum(cluster_labels == label) for label in non_singleton_labels]
            stats["cluster_sizes"] = {
                "mean": float(np.mean(cluster_sizes)),
                "std": float(np.std(cluster_sizes)),
                "min": int(np.min(cluster_sizes)),
                "max": int(np.max(cluster_sizes)),
                "median": float(np.median(cluster_sizes))
            }
        else:
            stats["cluster_sizes"] = {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
        
        return stats
    
    def _compute_subset_score(
        self, 
        predicted_labels: np.ndarray, 
        ground_truth_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute SubsetScore: percentage of predicted clusters that are 
        subsets of ground truth clusters.
        """
        if len(predicted_labels) != len(ground_truth_labels):
            return {"score": 0.0, "error": "Label arrays have different lengths"}
        
        # Group by predicted clusters
        pred_clusters = defaultdict(list)
        for i, label in enumerate(predicted_labels):
            if label >= 0 or self.config["allow_singletons_in_subset"]:
                pred_clusters[label].append(i)
        
        if not pred_clusters:
            return {"score": 0.0, "n_clusters": 0, "subset_clusters": 0}
        
        # Check subset condition for each predicted cluster
        subset_count = 0
        cluster_details = {}
        
        for pred_cluster_id, pred_indices in pred_clusters.items():
            # Get ground truth labels for this predicted cluster
            true_labels_in_cluster = [ground_truth_labels[i] for i in pred_indices]
            
            # Check if all points have the same ground truth label
            unique_true_labels = set(true_labels_in_cluster)
            is_subset = len(unique_true_labels) == 1
            
            if is_subset:
                subset_count += 1
            
            cluster_details[str(pred_cluster_id)] = {
                "size": len(pred_indices),
                "true_labels": list(unique_true_labels),
                "is_subset": is_subset,
                "purity": max(Counter(true_labels_in_cluster).values()) / len(true_labels_in_cluster)
            }
        
        subset_score = subset_count / len(pred_clusters)
        
        return {
            "score": subset_score,
            "n_clusters": len(pred_clusters),
            "subset_clusters": subset_count,
            "cluster_details": cluster_details,
            "interpretation": self._interpret_subset_score(subset_score)
        }
    
    def _interpret_subset_score(self, score: float) -> str:
        """Interpret SubsetScore value."""
        if score >= 0.9:
            return "Excellent - Very high purity clusters"
        elif score >= 0.7:
            return "Good - Most clusters are pure"
        elif score >= 0.5:
            return "Acceptable - Moderate cluster purity"
        elif score >= 0.3:
            return "Poor - Low cluster purity"
        else:
            return "Very Poor - Highly mixed clusters"
    
    def _compute_cluster_consistency(
        self, 
        cluster_labels: np.ndarray, 
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """Compute intra-cluster vs inter-cluster consistency."""
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        
        if len(unique_labels) < 2:
            return {"intra_similarity": 1.0, "inter_similarity": 0.0, "consistency_ratio": float('inf')}
        
        # Sample for efficiency
        max_samples = self.config["consistency_sample_size"]
        if len(embeddings) > max_samples:
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            sample_embeddings = embeddings[indices]
            sample_labels = cluster_labels[indices]
        else:
            sample_embeddings = embeddings
            sample_labels = cluster_labels
        
        # Compute intra-cluster similarities
        intra_similarities = []
        for label in unique_labels:
            cluster_indices = np.where(sample_labels == label)[0]
            if len(cluster_indices) > 1:
                cluster_embeddings = sample_embeddings[cluster_indices]
                for i in range(len(cluster_embeddings)):
                    for j in range(i + 1, len(cluster_embeddings)):
                        sim = np.dot(cluster_embeddings[i], cluster_embeddings[j])
                        intra_similarities.append(sim)
        
        # Compute inter-cluster similarities
        inter_similarities = []
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i + 1:]:
                indices1 = np.where(sample_labels == label1)[0]
                indices2 = np.where(sample_labels == label2)[0]
                
                # Sample pairs to avoid quadratic computation
                n_pairs = min(10, len(indices1) * len(indices2))
                for _ in range(n_pairs):
                    idx1 = np.random.choice(indices1)
                    idx2 = np.random.choice(indices2)
                    sim = np.dot(sample_embeddings[idx1], sample_embeddings[idx2])
                    inter_similarities.append(sim)
        
        # Calculate metrics
        avg_intra = np.mean(intra_similarities) if intra_similarities else 1.0
        avg_inter = np.mean(inter_similarities) if inter_similarities else 0.0
        consistency_ratio = avg_intra / (avg_inter + 1e-8)  # Avoid division by zero
        
        return {
            "intra_similarity": float(avg_intra),
            "inter_similarity": float(avg_inter), 
            "consistency_ratio": float(consistency_ratio),
            "n_intra_pairs": len(intra_similarities),
            "n_inter_pairs": len(inter_similarities),
            "interpretation": self._interpret_consistency(avg_intra, avg_inter, consistency_ratio)
        }
    
    def _interpret_consistency(self, intra: float, inter: float, ratio: float) -> str:
        """Interpret consistency metrics."""
        if ratio > 3.0 and intra > self.config["intra_cluster_threshold"]:
            return "Excellent - High intra-cluster similarity, low inter-cluster similarity"
        elif ratio > 2.0:
            return "Good - Clear separation between clusters"
        elif ratio > 1.5:
            return "Acceptable - Moderate cluster separation"
        else:
            return "Poor - Clusters not well separated"
    
    def _compute_separation_scores(
        self, 
        cluster_labels: np.ndarray, 
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """Compute silhouette-like separation scores."""
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        
        if len(unique_labels) < 2:
            return {"avg_silhouette": 1.0, "per_cluster_scores": {}}
        
        # Simplified silhouette computation
        silhouette_scores = []
        per_cluster_scores = {}
        
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            if len(cluster_indices) < 2:
                continue
            
            cluster_scores = []
            for idx in cluster_indices:
                # Intra-cluster distance (average distance to same cluster)
                same_cluster_indices = cluster_indices[cluster_indices != idx]
                if len(same_cluster_indices) > 0:
                    intra_dist = np.mean([
                        1 - np.dot(embeddings[idx], embeddings[other_idx])
                        for other_idx in same_cluster_indices[:10]  # Sample for efficiency
                    ])
                else:
                    intra_dist = 0
                
                # Inter-cluster distance (average distance to nearest other cluster)
                min_inter_dist = float('inf')
                for other_label in unique_labels:
                    if other_label == label:
                        continue
                    other_indices = np.where(cluster_labels == other_label)[0]
                    if len(other_indices) > 0:
                        inter_dist = np.mean([
                            1 - np.dot(embeddings[idx], embeddings[other_idx])
                            for other_idx in other_indices[:5]  # Sample for efficiency
                        ])
                        min_inter_dist = min(min_inter_dist, inter_dist)
                
                # Silhouette score
                if min_inter_dist != float('inf'):
                    silhouette = (min_inter_dist - intra_dist) / max(min_inter_dist, intra_dist)
                    cluster_scores.append(silhouette)
                    silhouette_scores.append(silhouette)
            
            if cluster_scores:
                per_cluster_scores[str(label)] = {
                    "avg_score": float(np.mean(cluster_scores)),
                    "std_score": float(np.std(cluster_scores)),
                    "size": len(cluster_indices)
                }
        
        avg_silhouette = np.mean(silhouette_scores) if silhouette_scores else 0.0
        
        return {
            "avg_silhouette": float(avg_silhouette),
            "per_cluster_scores": per_cluster_scores,
            "interpretation": self._interpret_silhouette(avg_silhouette)
        }
    
    def _interpret_silhouette(self, score: float) -> str:
        """Interpret silhouette score."""
        if score > 0.7:
            return "Excellent - Strong cluster structure"
        elif score > 0.5:
            return "Good - Reasonable cluster structure"
        elif score > 0.25:
            return "Acceptable - Weak cluster structure"
        else:
            return "Poor - No clear cluster structure"
    
    def _assess_overall_quality(self, quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall Phase 1 quality."""
        scores = []
        
        # Basic stats contribution
        basic_stats = quality_metrics.get("basic_stats", {})
        singleton_ratio = basic_stats.get("singleton_ratio", 1.0)
        # Lower singleton ratio is better (but some singletons are OK)
        singleton_score = max(0, 1 - min(singleton_ratio * 2, 1))  # Penalize >50% singletons
        scores.append(("singleton_ratio", singleton_score, 0.2))
        
        # SubsetScore contribution (if available)
        if "subset_score" in quality_metrics:
            subset_score = quality_metrics["subset_score"].get("score", 0)
            scores.append(("subset_score", subset_score, 0.4))
        
        # Consistency contribution
        consistency = quality_metrics.get("cluster_consistency", {})
        consistency_ratio = consistency.get("consistency_ratio", 1)
        consistency_score = min(consistency_ratio / 3.0, 1.0)  # Normalize to [0, 1]
        scores.append(("consistency", consistency_score, 0.3))
        
        # Separation contribution
        separation = quality_metrics.get("separation_scores", {})
        silhouette = separation.get("avg_silhouette", 0)
        silhouette_score = max(0, (silhouette + 1) / 2)  # Convert from [-1, 1] to [0, 1]
        scores.append(("separation", silhouette_score, 0.1))
        
        # Weighted average
        total_weight = sum(weight for _, _, weight in scores)
        if total_weight > 0:
            weighted_score = sum(score * weight for _, score, weight in scores) / total_weight
        else:
            weighted_score = 0.5
        
        # Determine quality level
        thresholds = self.config["quality_thresholds"]
        if weighted_score >= thresholds["excellent"]:
            quality_level = "excellent"
        elif weighted_score >= thresholds["good"]:
            quality_level = "good"
        elif weighted_score >= thresholds["acceptable"]:
            quality_level = "acceptable"
        else:
            quality_level = "poor"
        
        return {
            "weighted_score": weighted_score,
            "quality_level": quality_level,
            "component_scores": {name: score for name, score, _ in scores},
            "recommendations": self._generate_quality_recommendations(quality_metrics, weighted_score)
        }
    
    def _generate_quality_recommendations(
        self, 
        quality_metrics: Dict[str, Any], 
        overall_score: float
    ) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []
        
        # Check singleton ratio
        basic_stats = quality_metrics.get("basic_stats", {})
        singleton_ratio = basic_stats.get("singleton_ratio", 0)
        if singleton_ratio > 0.5:
            recommendations.append("High singleton ratio detected. Consider decreasing MCL inflation parameter.")
        
        # Check subset score
        if "subset_score" in quality_metrics:
            subset_score = quality_metrics["subset_score"].get("score", 0)
            if subset_score < 0.5:
                recommendations.append("Low subset score indicates mixed clusters. Consider adjusting kNN k parameter or CSLS threshold.")
        
        # Check consistency
        consistency = quality_metrics.get("cluster_consistency", {})
        consistency_ratio = consistency.get("consistency_ratio", 1)
        if consistency_ratio < 1.5:
            recommendations.append("Low consistency ratio. Clusters may not be well separated. Consider edge pruning adjustments.")
        
        # Check separation
        separation = quality_metrics.get("separation_scores", {})
        silhouette = separation.get("avg_silhouette", 0)
        if silhouette < 0.25:
            recommendations.append("Low silhouette score indicates poor cluster structure. Consider different clustering parameters.")
        
        # Overall recommendations
        if overall_score < 0.5:
            recommendations.append("Overall quality is low. Consider re-running with different parameters or more data preprocessing.")
        elif overall_score > 0.8:
            recommendations.append("High quality clustering achieved. Proceed to Phase 2.")
        
        return recommendations
    
    def _assess_label_quality(self, phase2_labels: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of Phase 2 label definitions."""
        label_quality = {}
        
        for label_id, label_data in phase2_labels.items():
            quality_score = 0.0
            quality_factors = []
            
            # Check label name quality
            name = label_data.get("name", "")
            if name and len(name) > 2:
                quality_score += 0.2
                quality_factors.append("valid_name")
            
            # Check definition quality
            definition = label_data.get("definition", "")
            if definition and len(definition) > 10:
                quality_score += 0.3
                quality_factors.append("valid_definition")
            
            # Check examples
            positive_examples = label_data.get("positive_examples", [])
            if len(positive_examples) >= 3:
                quality_score += 0.2
                quality_factors.append("sufficient_examples")
            
            # Check keywords
            keywords = label_data.get("keywords", [])
            if len(keywords) >= 2:
                quality_score += 0.1
                quality_factors.append("has_keywords")
            
            # Check decision rules
            decision_rules = label_data.get("decision_rules", {})
            if decision_rules.get("include_keywords") or decision_rules.get("regex_patterns"):
                quality_score += 0.2
                quality_factors.append("has_decision_rules")
            
            label_quality[label_id] = {
                "quality_score": quality_score,
                "quality_factors": quality_factors,
                "size": label_data.get("size", 0)
            }
        
        # Overall label quality
        if label_quality:
            avg_quality = np.mean([lq["quality_score"] for lq in label_quality.values()])
        else:
            avg_quality = 0.0
        
        return {
            "per_label_quality": label_quality,
            "average_quality": avg_quality,
            "n_labels": len(phase2_labels),
            "quality_distribution": self._compute_quality_distribution(label_quality)
        }
    
    def _compute_quality_distribution(self, label_quality: Dict[str, Any]) -> Dict[str, int]:
        """Compute distribution of label quality scores."""
        distribution = {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0}
        
        for quality_data in label_quality.values():
            score = quality_data["quality_score"]
            if score >= 0.8:
                distribution["excellent"] += 1
            elif score >= 0.6:
                distribution["good"] += 1
            elif score >= 0.4:
                distribution["acceptable"] += 1
            else:
                distribution["poor"] += 1
        
        return distribution
    
    def _assess_coverage(
        self, 
        phase2_labels: Dict[str, Any], 
        phase1_groups: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess how well Phase 2 labels cover Phase 1 groups."""
        # Count total Phase 1 groups
        total_groups = 0
        for question_results in phase1_groups.values():
            if question_results.get("status") == "completed":
                total_groups += question_results.get("n_clusters", 0)
        
        # Count covered groups
        covered_groups = 0
        for label_data in phase2_labels.values():
            group_ids = label_data.get("group_ids", [])
            covered_groups += len(group_ids)
        
        coverage_ratio = covered_groups / max(total_groups, 1)
        
        return {
            "total_phase1_groups": total_groups,
            "covered_groups": covered_groups,
            "coverage_ratio": coverage_ratio,
            "interpretation": "Complete coverage" if coverage_ratio >= 1.0 else 
                           "Good coverage" if coverage_ratio >= 0.8 else
                           "Partial coverage" if coverage_ratio >= 0.5 else
                           "Poor coverage"
        }
    
    def _track_user_modifications(self, user_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Track user modifications for feedback loop."""
        modifications = {
            "merge_count": 0,
            "split_count": 0,
            "rename_count": 0,
            "must_links_added": 0,
            "cannot_links_added": 0,
            "total_modifications": 0
        }
        
        # Count different types of modifications
        for action_type, actions in user_feedback.items():
            if isinstance(actions, list):
                count = len(actions)
                if action_type == "merges":
                    modifications["merge_count"] = count
                elif action_type == "splits":
                    modifications["split_count"] = count
                elif action_type == "renames":
                    modifications["rename_count"] = count
                elif action_type == "must_links":
                    modifications["must_links_added"] = count
                elif action_type == "cannot_links":
                    modifications["cannot_links_added"] = count
        
        modifications["total_modifications"] = sum([
            modifications["merge_count"],
            modifications["split_count"], 
            modifications["rename_count"]
        ])
        
        # Calculate modification quality penalty
        penalty = modifications["total_modifications"] * self.config["modification_penalty_weight"]
        modifications["quality_penalty"] = min(penalty, 0.5)  # Cap at 50% penalty
        
        return modifications
    
    def _assess_label_consistency(self, phase2_labels: Dict[str, Any]) -> Dict[str, Any]:
        """Assess consistency across Phase 2 labels."""
        consistency_metrics = {}
        
        # Check for overlapping keywords
        all_keywords = []
        label_keywords = {}
        
        for label_id, label_data in phase2_labels.items():
            keywords = label_data.get("keywords", [])
            label_keywords[label_id] = set(keywords)
            all_keywords.extend(keywords)
        
        # Find overlapping keywords between labels
        overlaps = []
        label_ids = list(label_keywords.keys())
        for i, label_id1 in enumerate(label_ids):
            for label_id2 in label_ids[i + 1:]:
                overlap = len(label_keywords[label_id1] & label_keywords[label_id2])
                if overlap > 0:
                    overlaps.append({
                        "label1": label_id1,
                        "label2": label_id2,
                        "overlap_count": overlap,
                        "overlap_keywords": list(label_keywords[label_id1] & label_keywords[label_id2])
                    })
        
        # Check name similarity (simple)
        name_similarities = []
        for i, label_id1 in enumerate(label_ids):
            name1 = phase2_labels[label_id1].get("name", "").lower()
            for label_id2 in label_ids[i + 1:]:
                name2 = phase2_labels[label_id2].get("name", "").lower()
                if name1 and name2:
                    # Simple word overlap
                    words1 = set(name1.split())
                    words2 = set(name2.split())
                    similarity = len(words1 & words2) / max(len(words1 | words2), 1)
                    if similarity > 0.3:  # Arbitrary threshold
                        name_similarities.append({
                            "label1": label_id1,
                            "label2": label_id2,
                            "similarity": similarity
                        })
        
        consistency_metrics = {
            "keyword_overlaps": overlaps,
            "name_similarities": name_similarities,
            "potential_conflicts": len(overlaps) + len(name_similarities),
            "consistency_score": max(0, 1 - (len(overlaps) + len(name_similarities)) * 0.1)
        }
        
        return consistency_metrics
    
    def _assess_phase2_overall_quality(self, quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall Phase 2 quality."""
        scores = []
        
        # Label quality contribution
        label_quality = quality_metrics.get("label_quality", {})
        avg_label_quality = label_quality.get("average_quality", 0)
        scores.append(("label_quality", avg_label_quality, 0.4))
        
        # Coverage contribution
        coverage = quality_metrics.get("coverage", {})
        coverage_ratio = coverage.get("coverage_ratio", 0)
        scores.append(("coverage", coverage_ratio, 0.3))
        
        # Consistency contribution
        consistency = quality_metrics.get("label_consistency", {})
        consistency_score = consistency.get("consistency_score", 0)
        scores.append(("consistency", consistency_score, 0.2))
        
        # User modification penalty (if available)
        if "user_modifications" in quality_metrics:
            modification_penalty = quality_metrics["user_modifications"].get("quality_penalty", 0)
            modification_score = 1 - modification_penalty
            scores.append(("user_satisfaction", modification_score, 0.1))
        
        # Weighted average
        total_weight = sum(weight for _, _, weight in scores)
        if total_weight > 0:
            weighted_score = sum(score * weight for _, score, weight in scores) / total_weight
        else:
            weighted_score = 0.5
        
        # Determine quality level
        thresholds = self.config["quality_thresholds"]
        if weighted_score >= thresholds["excellent"]:
            quality_level = "excellent"
        elif weighted_score >= thresholds["good"]:
            quality_level = "good" 
        elif weighted_score >= thresholds["acceptable"]:
            quality_level = "acceptable"
        else:
            quality_level = "poor"
        
        return {
            "weighted_score": weighted_score,
            "quality_level": quality_level,
            "component_scores": {name: score for name, score, _ in scores},
            "recommendations": self._generate_phase2_recommendations(quality_metrics, weighted_score)
        }
    
    def _generate_phase2_recommendations(
        self, 
        quality_metrics: Dict[str, Any], 
        overall_score: float
    ) -> List[str]:
        """Generate recommendations for Phase 2 improvement."""
        recommendations = []
        
        # Check label quality
        label_quality = quality_metrics.get("label_quality", {})
        avg_quality = label_quality.get("average_quality", 0)
        if avg_quality < 0.6:
            recommendations.append("Label quality is low. Consider adding more examples and improving definitions.")
        
        # Check coverage
        coverage = quality_metrics.get("coverage", {})
        coverage_ratio = coverage.get("coverage_ratio", 0)
        if coverage_ratio < 0.8:
            recommendations.append("Some Phase 1 groups are not covered by labels. Review clustering parameters.")
        
        # Check consistency
        consistency = quality_metrics.get("label_consistency", {})
        potential_conflicts = consistency.get("potential_conflicts", 0)
        if potential_conflicts > 2:
            recommendations.append("Multiple label conflicts detected. Review label definitions and keywords.")
        
        # Check user modifications
        if "user_modifications" in quality_metrics:
            total_mods = quality_metrics["user_modifications"].get("total_modifications", 0)
            if total_mods > 5:
                recommendations.append("High number of user modifications suggests algorithm parameters need tuning.")
        
        # Overall recommendations
        if overall_score < 0.5:
            recommendations.append("Overall Phase 2 quality is low. Consider re-running with different parameters.")
        elif overall_score > 0.8:
            recommendations.append("High quality labeling achieved. Labels are ready for use.")
        
        return recommendations


def quality_assessment_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for comprehensive quality assessment.
    
    Args:
        state: LangGraph state containing Phase 1 and Phase 2 results
        
    Returns:
        Updated state with quality assessment results
    """
    logger.info("🔍 Starting comprehensive quality assessment")
    
    try:
        # Initialize quality assessment tools
        quality_tools = QualityAssessmentTools()
        
        # Assess Phase 1 quality if available
        phase1_quality = {}
        if state.get("stage3_phase1_status") == "completed":
            logger.info("Assessing Phase 1 quality...")
            
            # Extract Phase 1 data for assessment
            phase1_groups = state.get("stage3_phase1_groups", {})
            matched_questions = state.get("matched_questions", {})
            
            for question_id, question_data in matched_questions.items():
                if question_id in phase1_groups:
                    result = phase1_groups[question_id]
                    if result.get("status") == "completed":
                        embeddings = np.array(question_data.get("embeddings", []))
                        cluster_labels = np.array(result.get("cluster_labels", []))
                        ground_truth = question_data.get("original_labels", None)
                        texts = question_data.get("texts", [])
                        
                        if embeddings.size > 0 and cluster_labels.size > 0:
                            assessment = quality_tools.assess_phase1_quality(
                                cluster_labels, embeddings, ground_truth, texts
                            )
                            phase1_quality[question_id] = assessment
        
        # Assess Phase 2 quality if available
        phase2_quality = {}
        if state.get("stage3_phase2_status") == "completed":
            logger.info("Assessing Phase 2 quality...")
            
            phase2_labels = state.get("stage3_phase2_labels", {})
            phase1_groups = state.get("stage3_phase1_groups", {})
            user_feedback = state.get("stage3_phase2_feedback", {})
            
            if phase2_labels:
                phase2_quality = quality_tools.assess_phase2_quality(
                    phase2_labels, phase1_groups, user_feedback
                )
        
        # Update state with quality assessment results
        updated_state = {
            **state,
            "stage3_phase1_quality": phase1_quality,
            "stage3_phase2_quality": phase2_quality
        }
        
        # Log quality summary
        if phase1_quality:
            avg_phase1_score = np.mean([
                q.get("overall_quality", {}).get("weighted_score", 0) 
                for q in phase1_quality.values() if "overall_quality" in q
            ])
            logger.info(f"Phase 1 average quality score: {avg_phase1_score:.3f}")
        
        if phase2_quality:
            phase2_score = phase2_quality.get("overall_quality", {}).get("weighted_score", 0)
            logger.info(f"Phase 2 quality score: {phase2_score:.3f}")
        
        logger.info("✅ Quality assessment completed")
        
        return updated_state
        
    except Exception as e:
        error_msg = f"Quality assessment failed: {str(e)}"
        logger.error(error_msg)
        
        return {
            **state,
            "stage3_error": error_msg
        }