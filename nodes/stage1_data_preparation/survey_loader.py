# nodes/stage1_data_preparation/survey_loader.py
## Survey file loader for Stage 1 - Data Preparation

from typing import Dict, Any
import os
import json
from pathlib import Path
from graph.state import GraphState
from tools.file_preprocess.file_loader import FileLoader

file_loader = FileLoader()

def load_survey_file(path: str):
    """기존 함수 - 단독 사용시"""
    survey_file = file_loader.load_survey(path)
    return {'raw_survey_info': survey_file}

def load_survey_node(state: GraphState) -> GraphState:
    """LangGraph용 노드 함수"""
    try:
        result = load_survey_file(state["survey_file_path"])
        state["raw_survey_info"] = result["raw_survey_info"]
        return state
    except Exception as e:
        state["error"] = f"Survey loading error: {str(e)}"
        return state