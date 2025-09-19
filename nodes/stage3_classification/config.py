"""
Configuration for Stage 3 Optimized Clustering.
Updated to use sentence embedding optimized algorithms instead of MCL.
"""
from typing import Dict, Any


class Stage3Config:
    """Configuration for Stage 3 Optimized Clustering."""
    
    # Default optimized clustering parameters
    DEFAULT_ALGORITHM = "adaptive"  # adaptive, kmeans, dbscan, hierarchical
    DEFAULT_KMEANS_K_RANGE = [3, 4, 5, 6, 7, 8]
    DEFAULT_DBSCAN_EPS_RANGE = [0.1, 0.2, 0.3, 0.4, 0.5]
    DEFAULT_DBSCAN_MIN_SAMPLES_RANGE = [3, 5, 7, 10]
    DEFAULT_HIERARCHICAL_K_RANGE = [3, 4, 5, 6, 7, 8]
    DEFAULT_HIERARCHICAL_LINKAGE = ['ward', 'complete', 'average']
    DEFAULT_MAX_CLUSTERS = 15
    DEFAULT_SELECTION_CRITERIA = "silhouette"
    
    # Data processing defaults
    DEFAULT_USE_LONG_FORMAT = True
    DEFAULT_EMBEDDING_COLUMNS = ["embed", "embedding", "vector"]
    DEFAULT_MAX_SAMPLES = 1000  # Performance limit
    
    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """Get default configuration dictionary.
        
        Returns:
            Default configuration values
        """
        return {
            "algorithm": cls.DEFAULT_ALGORITHM,
            "kmeans_k_range": cls.DEFAULT_KMEANS_K_RANGE,
            "dbscan_eps_range": cls.DEFAULT_DBSCAN_EPS_RANGE,
            "dbscan_min_samples_range": cls.DEFAULT_DBSCAN_MIN_SAMPLES_RANGE,
            "hierarchical_k_range": cls.DEFAULT_HIERARCHICAL_K_RANGE,
            "hierarchical_linkage": cls.DEFAULT_HIERARCHICAL_LINKAGE,
            "max_clusters": cls.DEFAULT_MAX_CLUSTERS,
            "selection_criteria": cls.DEFAULT_SELECTION_CRITERIA,
            "use_long_format": cls.DEFAULT_USE_LONG_FORMAT,
            "embedding_columns": cls.DEFAULT_EMBEDDING_COLUMNS,
            "max_samples": cls.DEFAULT_MAX_SAMPLES
        }
    
    @classmethod
    def merge_with_state(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        """Merge state parameters with defaults.
        
        Args:
            state: LangGraph state potentially containing custom parameters
            
        Returns:
            Merged configuration
        """
        config = cls.get_defaults()
        
        # Override with state values if present
        if "stage3_algorithm" in state:
            config["algorithm"] = state["stage3_algorithm"]
        
        if "stage3_kmeans_k_range" in state:
            config["kmeans_k_range"] = state["stage3_kmeans_k_range"]
            
        if "stage3_dbscan_eps_range" in state:
            config["dbscan_eps_range"] = state["stage3_dbscan_eps_range"]
            
        if "stage3_max_clusters" in state:
            config["max_clusters"] = state["stage3_max_clusters"]
            
        if "stage3_use_long_format" in state:
            config["use_long_format"] = state["stage3_use_long_format"]
            
        if "stage3_max_samples" in state:
            config["max_samples"] = state["stage3_max_samples"]
        
        return config
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validated and corrected configuration
        """
        validated = config.copy()
        
        # Validate algorithm choice
        valid_algorithms = ["adaptive", "kmeans", "dbscan", "hierarchical"]
        if validated["algorithm"] not in valid_algorithms:
            validated["algorithm"] = cls.DEFAULT_ALGORITHM
        
        # Validate numeric ranges
        validated["max_clusters"] = max(2, min(50, validated["max_clusters"]))
        validated["max_samples"] = max(50, min(5000, validated["max_samples"]))
        
        # Validate k ranges
        validated["kmeans_k_range"] = [k for k in validated["kmeans_k_range"] if 2 <= k <= 20]
        if not validated["kmeans_k_range"]:
            validated["kmeans_k_range"] = cls.DEFAULT_KMEANS_K_RANGE
        
        # Validate DBSCAN parameters
        validated["dbscan_eps_range"] = [eps for eps in validated["dbscan_eps_range"] if 0.01 <= eps <= 1.0]
        if not validated["dbscan_eps_range"]:
            validated["dbscan_eps_range"] = cls.DEFAULT_DBSCAN_EPS_RANGE
            
        validated["dbscan_min_samples_range"] = [ms for ms in validated["dbscan_min_samples_range"] if 2 <= ms <= 20]
        if not validated["dbscan_min_samples_range"]:
            validated["dbscan_min_samples_range"] = cls.DEFAULT_DBSCAN_MIN_SAMPLES_RANGE
        validated["search_iterations"] = max(1, min(50, validated["search_iterations"]))
        
        # Ensure boolean types
        validated["use_long_format"] = bool(validated["use_long_format"])
        
        # Ensure list type for embedding columns
        if not isinstance(validated["embedding_columns"], list):
            validated["embedding_columns"] = cls.DEFAULT_EMBEDDING_COLUMNS
        
        return validated


def get_stage3_config(state: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get validated Stage3 configuration.
    
    Args:
        state: Optional LangGraph state for parameter overrides
        
    Returns:
        Validated configuration dictionary
    """
    if state is None:
        state = {}
    
    config = Stage3Config.merge_with_state(state)
    return Stage3Config.validate_config(config)