"""
Prompt branch mapping for different question types
"""


def get_branch_by_type(question_type: str) -> str:
    """
    Get appropriate prompt branch based on question type
    
    Args:
        question_type: Type of question (depend_pos_neg, pos_neg, depend, sentence)
        
    Returns:
        Branch name for llm_router
    """
    branch_map = {
        "depend_pos_neg": "sentence_depend_pos_neg_split",
        "pos_neg": "sentence_pos_neg_split", 
        "depend": "sentence_depend_split",
        "sentence": "sentence_only"
    }
    
    return branch_map.get(question_type, "sentence_only")