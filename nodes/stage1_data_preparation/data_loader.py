# nodes/stage1_data_preparation/data_loader.py
## Data file loader for Stage 1 - Data Preparation

import pandas as pd
import json
from pathlib import Path
from tools.file_preprocess.file_loader import FileLoader
from graph.state import GraphState

file_loader = FileLoader()

def load_data_file(path: str):
    """기존 함수 - 단독 사용시"""
    data_file = file_loader.smart_load_excel(path)
    return {'raw_data_info': data_file}

def load_data_node(state: GraphState) -> GraphState:
    """LangGraph용 노드 함수"""
    try:
        result = load_data_file(state["data_file_path"])
        state["raw_data_info"] = result["raw_data_info"]
        
        # 새로운 명시적 이름으로 설정
        dataframe_path = result["raw_data_info"]["dataframe_path"]
        state["raw_dataframe_path"] = dataframe_path
        
        return state
    except Exception as e:
        state["error"] = f"Data loading error: {str(e)}"
        return state