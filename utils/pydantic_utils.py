"""
Pydantic to JSON conversion utilities
"""
from typing import Any, Dict, List, Union
from pydantic import BaseModel


def pydantic_to_dict(obj: Any) -> Union[Dict, List, Any]:
    """
    Convert Pydantic model to dictionary
    
    Args:
        obj: Pydantic model instance, RootModel, or any other object
        
    Returns:
        Dictionary representation or the original object if not a Pydantic model
    """
    if isinstance(obj, BaseModel):
        # Pydantic v2 방식
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        # Pydantic v1 fallback
        elif hasattr(obj, 'dict'):
            return obj.dict()
    
    # RootModel의 경우 root 속성에 실제 데이터가 있음
    if hasattr(obj, 'root'):
        return obj.root
    
    return obj


def pydantic_to_json(obj: Any) -> str:
    """
    Convert Pydantic model to JSON string
    
    Args:
        obj: Pydantic model instance or any other object
        
    Returns:
        JSON string representation
    """
    import json
    
    if isinstance(obj, BaseModel):
        # Pydantic v2 방식
        if hasattr(obj, 'model_dump_json'):
            return obj.model_dump_json()
        # Pydantic v1 fallback
        elif hasattr(obj, 'json'):
            return obj.json()
    
    # RootModel 또는 기타 객체
    data = pydantic_to_dict(obj)
    return json.dumps(data, ensure_ascii=False, indent=2)


def extract_llm_response_data(response: Any) -> Any:
    """
    Extract actual data from LLM response which might contain raw/parsed structure
    
    Args:
        response: LLM response (could be dict with 'parsed' key, Pydantic model, or raw data)
        
    Returns:
        Extracted data in dictionary format
    """
    # LLM client가 {'raw': ..., 'parsed': ..., 'parsing_error': ...} 형태로 반환하는 경우
    if isinstance(response, dict):
        if 'parsed' in response and response['parsed'] is not None:
            return pydantic_to_dict(response['parsed'])
        elif 'raw' in response:
            return response['raw']
    
    # 직접 Pydantic 모델인 경우
    return pydantic_to_dict(response)


def safe_get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    """
    Safely get field from Pydantic model or dict
    
    Args:
        obj: Pydantic model or dictionary
        field_name: Field name to extract
        default: Default value if field doesn't exist
        
    Returns:
        Field value or default
    """
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    elif hasattr(obj, field_name):
        return getattr(obj, field_name, default)
    else:
        # Try converting to dict first
        try:
            data = pydantic_to_dict(obj)
            return data.get(field_name, default) if isinstance(data, dict) else default
        except:
            return default