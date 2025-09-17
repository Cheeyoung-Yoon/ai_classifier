# utils/json_utils.py
import json
from typing import Any, Dict
from pydantic import BaseModel
import pandas as pd


def make_json_serializable(obj: Any) -> Any:
    """
    객체를 JSON 직렬화 가능한 형태로 변환
    """
    if isinstance(obj, BaseModel):
        # Pydantic 모델의 경우 dict로 변환
        return obj.model_dump()
    
    elif isinstance(obj, pd.DataFrame):
        # DataFrame의 경우 dict로 변환
        return obj.to_dict()
    
    elif hasattr(obj, '__dict__'):
        # 일반 객체의 경우 __dict__ 사용
        return {k: make_json_serializable(v) for k, v in obj.__dict__.items()}
    
    elif isinstance(obj, dict):
        # 딕셔너리의 경우 재귀적으로 변환
        return {k: make_json_serializable(v) for k, v in obj.items()}
    
    elif isinstance(obj, (list, tuple)):
        # 리스트/튜플의 경우 재귀적으로 변환
        return [make_json_serializable(item) for item in obj]
    
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # 기본 타입은 그대로 반환
        return obj
    
    else:
        # 기타 객체는 문자열로 변환
        return str(obj)


def save_debug_json(data: Any, filepath: str):
    """
    디버그용으로 안전하게 JSON 저장
    """
    try:
        serializable_data = make_json_serializable(data)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        print(f"Debug data saved to: {filepath}")
    except Exception as e:
        print(f"Failed to save debug data: {e}")
        # 대체 방법: 텍스트로 저장
        try:
            with open(filepath.replace('.json', '.txt'), "w", encoding="utf-8") as f:
                f.write(str(data))
            print(f"Data saved as text to: {filepath.replace('.json', '.txt')}")
        except Exception as e2:
            print(f"Failed to save as text: {e2}")


def extract_pydantic_data(pydantic_obj: BaseModel) -> Dict[str, Any]:
    """
    Pydantic 객체에서 안전하게 데이터 추출
    """
    try:
        return pydantic_obj.model_dump()
    except Exception:
        try:
            return pydantic_obj.dict()
        except Exception:
            return {"error": "Failed to extract pydantic data", "type": str(type(pydantic_obj))}
