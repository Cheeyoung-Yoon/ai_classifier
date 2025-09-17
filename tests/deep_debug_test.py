# tests/deep_debug_test.py
# %%
"""
Deep debugging script with complete state printing
"""

import sys
import os

# Add project root to path
project_root = "/home/cyyoon/test_area/ai_text_classification/2.langgraph"
sys.path.insert(0, project_root)

from graph.graph import graph
import pandas as pd
import json
import pprint
from graph.graph import create_workflow
from nodes.state_flush_node import stage1_memory_flush_node, memory_status_check_node, force_memory_cleanup

# 파일 상단 import 근처에 추가
import pickle
import pathlib
from datetime import datetime

try:
    import pandas as pd
except Exception:
    pd = None

def _to_jsonable(obj):
    """JSON 직렬화 가능 형태로 변환 (최대한 정보 보존, non-serializable은 repr)"""
    # 기본 타입
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    
    # dict
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    
    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    
    # pandas
    if pd is not None:
        if isinstance(obj, pd.DataFrame):
            # 너무 크면 전부 쓰지 말고 정보 + head만
            try:
                preview = obj.head(20).to_dict(orient="list")
            except Exception:
                preview = "DataFrame(head) 변환 실패"
            return {
                "__type__": "pandas.DataFrame",
                "shape": list(obj.shape),
                "columns": obj.columns.tolist(),
                "dtypes": {c: str(t) for c, t in obj.dtypes.items()},
                "head_preview": preview,
            }
        if isinstance(obj, pd.Series):
            return {
                "__type__": "pandas.Series",
                "name": obj.name,
                "shape": [len(obj)],
                "dtype": str(obj.dtype),
                "head_preview": obj.head(20).to_list(),
            }
    
    # numpy
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            # 큰 배열은 summary만
            return {
                "__type__": "numpy.ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "preview": obj.flatten()[:20].tolist()
            }
        # numpy scalar
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    except Exception:
        pass
    
    # pathlib
    if isinstance(obj, pathlib.Path):
        return str(obj)
    
    # bytes는 길이만
    if isinstance(obj, (bytes, bytearray)):
        return {"__type__": "bytes", "length": len(obj)}
    
    # 예외
    if isinstance(obj, BaseException):
        return {"__type__": "Exception", "repr": repr(obj), "str": str(obj)}
    
    # 기타: 형 정보 + repr
    return {"__type__": type(obj).__name__, "repr": repr(obj)}

def save_state_snapshot(state: dict, step_name: str, out_dir: str = "debug_states", prefix: str = "") -> dict:
    """
    state를 JSON + pickle로 저장.
    반환: 저장된 파일 경로 dict
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_step = step_name.replace(" ", "_").replace("/", "_")
    if prefix:
        prefix = prefix.rstrip("_") + "_"
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    base = out_path / f"{prefix}{ts}_{safe_step}"
    json_path = base.with_suffix(".json")
    pkl_path = base.with_suffix(".pkl")
    
    # JSON (사람 친화적)
    jsonable = _to_jsonable(state)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(jsonable, f, ensure_ascii=False, indent=2)
    
    # pickle (완전 복원용)
    with open(pkl_path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return {"json": str(json_path), "pickle": str(pkl_path)}



def deep_print_state(state, step_name):
    """Print complete state information"""
    print(f"\n{'='*80}")
    print(f"STEP: {step_name}")
    print(f"{'='*80}")
    
    for key, value in state.items():
        print(f"\n--- {key.upper()} ---")
        
        if value is None:
            print("None")
        elif isinstance(value, str):
            if len(value) > 500:
                print(f"String (length: {len(value)})")
                print(f"Preview: {value[:200]}...")
                print(f"...{value[-200:]}")
            else:
                print(value)
        elif isinstance(value, dict):
            if len(str(value)) > 1000:
                print(f"Dict with keys: {list(value.keys())}")
                for k, v in value.items():
                    print(f"  {k}: {type(v)} {f'(len: {len(v)})' if hasattr(v, '__len__') else ''}")
                    if isinstance(v, dict) and len(str(v)) < 500:
                        print(f"    Content: {v}")
                    elif isinstance(v, list) and len(v) < 10:
                        print(f"    Content: {v}")
            else:
                pprint.pprint(value, width=120, depth=3)
        elif isinstance(value, list):
            print(f"List (length: {len(value)})")
            if len(value) < 5:
                pprint.pprint(value, width=120, depth=2)
            else:
                print(f"First 3 items:")
                for i, item in enumerate(value[:3]):
                    print(f"  [{i}]: {type(item)} - {str(item)[:100]}")
                if len(value) > 3:
                    print(f"  ... and {len(value) - 3} more items")
        else:
            print(f"{type(value)}: {str(value)[:200]}")
    
    print(f"\n{'='*80}")
    print(f"END OF STEP: {step_name}")
    print(f"{'='*80}\n")
    save_state_snapshot(state, step_name, out_dir="debug_states", prefix="print")
    
    
def run_debug_workflow():
    """Run workflow with deep debugging"""
    print("Creating workflow...")
    app = create_workflow()
    
    # Initial state
    initial_state = {
        "project_name": "SUV_DEBUG",
        "survey_file_path": "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/test.txt",
        "data_file_path": "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/-SUV_776부.xlsx",
        "raw_survey_info": None,
        "raw_data_info": None,
        "dataframe": None,
        "parsed_survey": None,
        "open_columns": None,
        "data_sample": None,
        "question_data_match": None,
        "matched_questions": None,
        "matched_questions_meta": None,
        "llm_logs": [],
        "llm_meta": [],
        "error": None
    }
    
    deep_print_state(initial_state, "INITIAL STATE")
    
    # Custom execution with state tracking
    current_state = initial_state.copy()
    
    try:
        # Step 1: Load Survey
        print("\n" + ">"*50 + " EXECUTING: LOAD SURVEY " + "<"*50)
        from nodes.stage1_data_preparation.survey_loader import load_survey_node
        current_state = load_survey_node(current_state)
        deep_print_state(current_state, "AFTER LOAD SURVEY")
        
        if current_state.get("error"):
            print(f"ERROR in load_survey: {current_state['error']}")
            return current_state
        
        # Step 2: Load Data
        print("\n" + ">"*50 + " EXECUTING: LOAD DATA " + "<"*50)
        from nodes.stage1_data_preparation.data_loader import load_data_node
        current_state = load_data_node(current_state)
        deep_print_state(current_state, "AFTER LOAD DATA")
        
        if current_state.get("error"):
            print(f"ERROR in load_data: {current_state['error']}")
            return current_state
        
        # Step 3: Parse Survey
        print("\n" + ">"*50 + " EXECUTING: PARSE SURVEY " + "<"*50)
        from nodes.stage1_data_preparation.survey_parser import parse_survey_node
        current_state = parse_survey_node(current_state)
        deep_print_state(current_state, "AFTER PARSE SURVEY")
        
        if current_state.get("error"):
            print(f"ERROR in parse_survey: {current_state['error']}")
            return current_state
        
        # Step 4: Get Open Columns
        print("\n" + ">"*50 + " EXECUTING: GET OPEN COLUMNS " + "<"*50)
        from nodes.stage1_data_preparation.column_extractor import get_open_column_node
        current_state = get_open_column_node(current_state)
        deep_print_state(current_state, "AFTER GET OPEN COLUMNS")
        
        if current_state.get("error"):
            print(f"ERROR in get_open_column: {current_state['error']}")
            return current_state
        
        # Step 5: Question Data Matcher - THE PROBLEMATIC STEP
        print("\n" + ">"*50 + " EXECUTING: QUESTION DATA MATCHER " + "<"*50)
        print("STATE BEFORE MATCHER:")
        print("parsed_survey type:", type(current_state.get('parsed_survey')))
        print("parsed_survey keys:", current_state.get('parsed_survey', {}).keys() if isinstance(current_state.get('parsed_survey'), dict) else "Not a dict")
        
        if current_state.get('parsed_survey') and isinstance(current_state['parsed_survey'], dict):
            if 'parsed' in current_state['parsed_survey']:
                print("parsed_survey['parsed'] type:", type(current_state['parsed_survey']['parsed']))
                print("parsed_survey['parsed'] keys:", current_state['parsed_survey']['parsed'].keys() if isinstance(current_state['parsed_survey']['parsed'], dict) else "Not a dict")
                
                if 'questions' in current_state['parsed_survey']['parsed']:
                    questions = current_state['parsed_survey']['parsed']['questions']
                    print("questions type:", type(questions))
                    print("questions length:", len(questions) if hasattr(questions, '__len__') else "No length")
                    if isinstance(questions, list) and questions:
                        print("First question:", questions[0])
        
        from nodes.stage1_data_preparation.question_matcher import question_data_matcher_node
        try:
            current_state = question_data_matcher_node(current_state)
            deep_print_state(current_state, "AFTER QUESTION DATA MATCHER")
        except Exception as e:
            print(f"DETAILED ERROR in question_data_matcher: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return current_state
        
        if current_state.get("error"):
            print(f"ERROR in question_data_matcher: {current_state['error']}")
            return current_state
        # Step 6:  Survey Data Integration
        from nodes.survey_data_integrate import survey_data_integrate_node
        
        try:
            current_state = survey_data_integrate_node(current_state)
            deep_print_state(current_state, "AFTER Survey Data Integration")
        
        except Exception as e:
            print(f"DETAILED ERROR in survey_data_integrate: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return current_state
        
        if current_state.get("error"):
            print(f"ERROR in question_data_matcher: {current_state['error']}")
            return current_state
        # 🧹 MEMORY FLUSH TEST 추가!
        print("\n" + "🧹"*20 + " TESTING MEMORY FLUSH " + "🧹"*20)
        try:
            # Memory status before flush
            memory_status_check_node(current_state)
            
            # Stage 1 memory flush
            from nodes.stage1_data_preparation.memory_optimizer import stage1_memory_flush_node
            current_state = stage1_memory_flush_node(current_state)
            deep_print_state(current_state, "AFTER STAGE 1 MEMORY FLUSH")
            
            # 🗄️ FLUSH된 STATE를 JSON으로 저장
            print("\n💾 Saving flushed state to JSON...")
            try:
                saved_files = save_state_snapshot(current_state, "stage1_flushed", prefix="flushed_")
                print(f"✅ Flushed state saved:")
                for file_type, file_path in saved_files.items():
                    print(f"  {file_type}: {file_path}")
                    
                # JSON 파일 내용 미리보기
                import json
                json_path = saved_files.get('json_path')
                if json_path and os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        saved_data = json.load(f)
                    print(f"  📄 JSON contains {len(saved_data)} fields")
                    non_null_fields = [k for k, v in saved_data.items() if v is not None]
                    print(f"  📊 Non-null fields: {len(non_null_fields)} - {non_null_fields[:5]}{'...' if len(non_null_fields) > 5 else ''}")
                    
            except Exception as e:
                print(f"❌ Failed to save flushed state: {e}")
            
            # Memory status after flush
            memory_status_check_node(current_state)
            
            print("✅ Memory flush integration test completed successfully!")
            
        except Exception as e:
            print(f"❌ Memory flush integration test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "🎉"*20 + " WORKFLOW COMPLETED SUCCESSFULLY " + "🎉"*20)
        
    except Exception as e:
        print(f"WORKFLOW ERROR: {e}")
        import traceback
        traceback.print_exc()
        current_state["error"] = str(e)
    
    return current_state

if __name__ == "__main__":
    print("Running debug workflow with integrated memory flush...")
    
    # Run full workflow with integrated memory flush
    result = run_debug_workflow()
    print(f"\nFINAL RESULT:")
    print(f"Error: {result.get('error', 'None')}")
    print(f"Completed steps: {[k for k, v in result.items() if v is not None and k != 'error']}")

# %%
