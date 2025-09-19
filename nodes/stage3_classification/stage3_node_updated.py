"""
Stage3 Classification Node for LangGraph Pipeline
Updated to use optimized clustering algorithms instead of MCL
"""

from typing import Dict, Any
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from stage3_node_optimized import stage3_classification_node as _optimized_node


def stage3_classification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main Stage3 classification node for LangGraph pipeline.
    Now uses optimized clustering algorithms instead of MCL.
    
    Args:
        state: LangGraph state containing stage2 results
        
    Returns:
        Updated state with stage3 clustering results
    """
    return _optimized_node(state)


def run_stage3_classification(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Legacy entry point for backward compatibility.
    
    Args:
        state: LangGraph state containing stage2 results
        
    Returns:
        Updated state with stage3 clustering results
    """
    return stage3_classification_node(state)


# Export the main functions
__all__ = ['stage3_classification_node', 'run_stage3_classification']