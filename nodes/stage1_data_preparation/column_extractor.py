# nodes/stage1_data_preparation/column_extractor.py
# Open column extractor for Stage 1 - Data Preparation

from tools.file_preprocess.get_open_column import DataFrameParser
from graph.state import GraphState
from utils.data_utils import DataHelper

def get_open_column_original(state: dict, deps=None) -> dict:
    """기존 함수 - DataFrame 직접 사용시 (하위 호환성)"""
    df = state["raw_dataframe_path"]
    return {"open_columns": DataFrameParser.get_object_columns(df)}

def get_open_column_from_path(dataframe_path: str, meta_info: dict = None) -> dict:
    """경로에서 open columns 추출 - 메모리 효율적"""
    if meta_info and 'object_columns' in meta_info:
        # 이미 분석된 정보가 있으면 사용
        return {"open_columns": meta_info['object_columns']}
    
    # 메타 정보가 없으면 DataHelper 사용
    return {"open_columns": DataHelper.get_open_columns_from_path(dataframe_path, meta_info or {})}

def get_open_column_node(state: GraphState, deps=None) -> GraphState:
    """LangGraph용 노드 함수 - 경로 기반"""
    try:
        # dataframe_path와 meta 정보 사용
        if "raw_data_info" in state and "dataframe_path" in state["raw_data_info"]:
            dataframe_path = state["raw_data_info"]["dataframe_path"]
            meta_info = state["raw_data_info"].get("meta", {})
            result = get_open_column_from_path(dataframe_path, meta_info)
            state["open_columns"] = result["open_columns"]
            
            # Memory cleanup은 이제 flush 노드에서 처리
                
        elif "raw_dataframe_path" in state:
            # 하위 호환성: DataFrame 객체가 직접 전달된 경우
            open_columns = DataFrameParser.get_object_columns(state["raw_dataframe_path"])
            state["open_columns"] = open_columns
        else:
            raise KeyError("No dataframe_path or raw_dataframe_path found in state")
        
        return state
    except Exception as e:
        state["error"] = f"Open columns detection error: {str(e)}"
        return state