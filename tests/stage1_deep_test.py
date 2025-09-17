# tests/stage1_deep_test.py
"""
Stage 1 Deep Test - 각 노드별 상세 로그 및 state 저장 테스트

이 테스트는 다음 기능을 제공합니다:
1. 각 노드 실행 전후 state 상세 로깅
2. 모든 state를 JSON/pickle 형태로 자동 저장
3. Stage tracking 및 history 관리 통합
4. LLM cost 및 runtime 추적
5. Memory optimization 과정 상세 기록
6. 에러 발생 시 완전한 디버깅 정보 제공
"""

import sys
import os
import json
import pickle
import pprint
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = "/home/cyyoon/test_area/ai_text_classification/2.langgraph"
sys.path.insert(0, project_root)

from graph.state import initialize_project_state
from utils.stage_history_manager import get_or_create_history_manager
from utils.cost_tracker import calculate_total_llm_cost, print_pipeline_status
from nodes.shared.stage_tracker import update_stage_tracking, print_pipeline_status

# Import all stage 1 nodes
from nodes.stage1_data_preparation.survey_loader import load_survey_node
from nodes.stage1_data_preparation.data_loader import load_data_node
from nodes.stage1_data_preparation.survey_parser import parse_survey_node
from nodes.stage1_data_preparation.survey_context import survey_context_node
from nodes.stage1_data_preparation.column_extractor import get_open_column_node
from nodes.stage1_data_preparation.question_matcher import question_data_matcher_node
from nodes.survey_data_integrate import survey_data_integrate_node
from nodes.stage1_data_preparation.memory_optimizer import stage1_memory_flush_node
from nodes.state_flush_node import memory_status_check_node

try:
    import pandas as pd
except ImportError:
    pd = None

class Stage1DeepTester:
    """Stage 1 노드별 상세 테스트 클래스"""
    
    def __init__(self, project_name: str = "stage1_deep_test", 
                 survey_file: str = "test.txt", 
                 data_file: str = "-SUV_776부.xlsx"):
        self.project_name = project_name
        self.survey_file = survey_file
        self.data_file = data_file
        self.test_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.output_dir = Path(f"tests/debug_states/stage1_deep_{self.test_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test results tracking
        self.node_results = []
        self.stage_snapshots = {}
        self.error_states = []
        
        print(f"🔬 Stage 1 Deep Test Initialized")
        print(f"  Test ID: {self.test_id}")
        print(f"  Output Directory: {self.output_dir}")
        print(f"  Project: {project_name}")
        print(f"  Survey: {survey_file}")
        print(f"  Data: {data_file}")
    
    def _to_jsonable(self, obj):
        """JSON 직렬화 가능 형태로 변환 (최대한 정보 보존)"""
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        
        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}
        
        if isinstance(obj, (list, tuple, set)):
            return [self._to_jsonable(v) for v in obj]
        
        # pandas handling
        if pd is not None:
            if isinstance(obj, pd.DataFrame):
                try:
                    preview = obj.head(10).to_dict(orient="records")
                except Exception:
                    preview = "DataFrame preview failed"
                return {
                    "__type__": "pandas.DataFrame",
                    "shape": list(obj.shape),
                    "columns": obj.columns.tolist(),
                    "dtypes": {c: str(t) for c, t in obj.dtypes.items()},
                    "memory_usage": obj.memory_usage(deep=True).sum() if hasattr(obj, 'memory_usage') else "unknown",
                    "head_preview": preview
                }
            if isinstance(obj, pd.Series):
                return {
                    "__type__": "pandas.Series",
                    "name": obj.name,
                    "shape": [len(obj)],
                    "dtype": str(obj.dtype),
                    "head_preview": obj.head(10).to_list()
                }
        
        # numpy handling
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return {
                    "__type__": "numpy.ndarray",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "size": obj.size,
                    "preview": obj.flatten()[:10].tolist()
                }
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
        except ImportError:
            pass
        
        # pathlib
        if isinstance(obj, Path):
            return str(obj)
        
        # bytes
        if isinstance(obj, (bytes, bytearray)):
            return {"__type__": "bytes", "length": len(obj)}
        
        # exceptions
        if isinstance(obj, BaseException):
            return {"__type__": "Exception", "repr": repr(obj), "str": str(obj)}
        
        # default: type info + repr
        try:
            return {"__type__": type(obj).__name__, "repr": repr(obj)[:500]}
        except:
            return {"__type__": type(obj).__name__, "repr": "repr_failed"}
    
    def save_state_snapshot(self, state: Dict[str, Any], step_name: str, 
                           node_name: str = "", execution_time: float = 0.0) -> Dict[str, str]:
        """
        State를 JSON과 pickle 형태로 저장하고 메타데이터 기록
        
        Returns:
            저장된 파일 경로들
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_step = step_name.replace(" ", "_").replace("/", "_")
        safe_node = node_name.replace(" ", "_").replace("/", "_") if node_name else ""
        
        if safe_node:
            base_name = f"{timestamp}_{safe_node}_{safe_step}"
        else:
            base_name = f"{timestamp}_{safe_step}"
        
        json_path = self.output_dir / f"{base_name}.json"
        pkl_path = self.output_dir / f"{base_name}.pkl"
        meta_path = self.output_dir / f"{base_name}_meta.json"
        
        # JSON 저장 (사람이 읽기 쉬운 형태)
        jsonable_state = self._to_jsonable(state)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(jsonable_state, f, ensure_ascii=False, indent=2)
        
        # pickle 저장 (완전 복원용)
        with open(pkl_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 메타데이터 저장
        metadata = {
            "test_id": self.test_id,
            "timestamp": timestamp,
            "step_name": step_name,
            "node_name": node_name,
            "execution_time_seconds": execution_time,
            "state_fields": list(state.keys()),
            "non_null_fields": [k for k, v in state.items() if v is not None],
            "null_fields": [k for k, v in state.items() if v is None],
            "pipeline_id": state.get('pipeline_id'),
            "current_stage": state.get('current_stage'),
            "total_llm_cost": state.get('total_llm_cost_usd', 0),
            "llm_calls_count": len(state.get('llm_logs') or []),
            "error": state.get('error'),
            "files": {
                "json": str(json_path),
                "pickle": str(pkl_path),
                "metadata": str(meta_path)
            }
        }
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 저장 결과를 내부 추적에 기록
        self.stage_snapshots[step_name] = metadata
        
        return {
            "json": str(json_path),
            "pickle": str(pkl_path),
            "metadata": str(meta_path)
        }
    
    def log_node_execution(self, node_name: str, state_before: Dict[str, Any], 
                          state_after: Dict[str, Any], execution_time: float, 
                          error: Exception = None):
        """노드 실행 결과를 상세히 로깅"""
        print(f"\n{'#' * 60}")
        print(f"NODE EXECUTION LOG: {node_name}")
        print(f"{'#' * 60}")
        print(f"  Execution Time: {execution_time:.4f}s")
        print(f"  State Fields Before: {len([k for k, v in state_before.items() if v is not None])}")
        print(f"  State Fields After: {len([k for k, v in state_after.items() if v is not None])}")
        
        # Changed fields analysis
        changed_fields = []
        new_fields = []
        deleted_fields = []
        
        for key in set(state_before.keys()) | set(state_after.keys()):
            before_val = state_before.get(key)
            after_val = state_after.get(key)
            
            if key not in state_before and after_val is not None:
                new_fields.append(key)
            elif key not in state_after:
                deleted_fields.append(key)
            elif before_val != after_val:
                changed_fields.append(key)
        
        if new_fields:
            print(f"  New Fields: {new_fields}")
        if changed_fields:
            print(f"  Changed Fields: {changed_fields}")
        if deleted_fields:
            print(f"  Deleted Fields: {deleted_fields}")
        
        # LLM usage tracking
        before_llm_count = len(state_before.get('llm_logs') or [])
        after_llm_count = len(state_after.get('llm_logs') or [])
        if after_llm_count > before_llm_count:
            new_llm_calls = after_llm_count - before_llm_count
            before_cost = state_before.get('total_llm_cost_usd', 0)
            after_cost = state_after.get('total_llm_cost_usd', 0)
            cost_increase = after_cost - before_cost
            print(f"  LLM Usage: +{new_llm_calls} calls, +${cost_increase:.6f}")
        
        # Error handling
        if error:
            print(f"  ERROR: {error}")
            print(f"  Error Type: {type(error).__name__}")
            self.error_states.append({
                "node_name": node_name,
                "error": str(error),
                "error_type": type(error).__name__,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            })
        elif state_after.get('error'):
            print(f"  State Error: {state_after['error']}")
        else:
            print(f"  Success")
        
        # 노드 실행 결과 기록
        result_record = {
            "node_name": node_name,
            "execution_time": execution_time,
            "success": error is None and not state_after.get('error'),
            "fields_before": len([k for k, v in state_before.items() if v is not None]),
            "fields_after": len([k for k, v in state_after.items() if v is not None]),
            "new_fields": new_fields,
            "changed_fields": changed_fields,
            "deleted_fields": deleted_fields,
            "llm_calls_added": max(0, after_llm_count - before_llm_count),
            "cost_increase": state_after.get('total_llm_cost_usd', 0) - state_before.get('total_llm_cost_usd', 0),
            "error": str(error) if error else state_after.get('error'),
            "timestamp": datetime.now().isoformat()
        }
        
        self.node_results.append(result_record)
        
        print(f"{'#' * 60}\n")
    
    def execute_node_with_logging(self, node_func, node_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """노드를 실행하고 상세한 로깅과 state 저장을 수행"""
        print(f"\n{'=' * 50}")
        print(f"EXECUTING NODE: {node_name}")
        print(f"{'=' * 50}")
        
        # 실제 graph 노드 이름과 매핑
        graph_node_names = {
            "LOAD_SURVEY": "load_survey",
            "LOAD_DATA": "load_data", 
            "PARSE_SURVEY": "parse_survey",
            "EXTRACT_SURVEY_CONTEXT": "extract_survey_context",
            "GET_OPEN_COLUMNS": "get_open_columns",
            "QUESTION_DATA_MATCHER": "match_questions",
            "SURVEY_DATA_INTEGRATE": "survey_data_integrate",
            "STAGE1_MEMORY_FLUSH": "stage1_memory_flush",
            "MEMORY_STATUS_CHECK": "memory_status_check"
        }
        
        # 실제 graph 노드 이름 사용
        actual_node_name = graph_node_names.get(node_name, node_name.lower())
        
        # 노드 실행 전 상태를 로깅용으로만 보관 (저장하지 않음 - 이전 노드 AFTER와 중복)
        state_before = state.copy()
        
        # 노드 실행
        start_time = datetime.now()
        error = None
        try:
            state_after = node_func(state)
        except Exception as e:
            error = e
            state_after = state.copy()
            state_after['error'] = str(e)
            print(f"ERROR: Node execution failed: {e}")
            import traceback
            traceback.print_exc()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Stage tracking update (노드 완료 후)
        if not error and 'update_stage_tracking' in globals():
            try:
                state_after = update_stage_tracking(state_after, actual_node_name)
            except Exception as e:
                print(f"WARNING: Stage tracking warning: {e}")
        
        # After state 저장
        self.save_state_snapshot(state_after, f"AFTER_{node_name}", node_name, execution_time)
        
        # 로깅
        self.log_node_execution(node_name, state_before, state_after, execution_time, error)
        
        return state_after
    
    def run_full_stage1_test(self):
        """Stage 1 전체 노드들을 순차적으로 실행하며 상세 테스트"""
        print(f"\n{'#' * 80}")
        print(f"STARTING STAGE 1 DEEP TEST")
        print(f"{'#' * 80}")
        
        # Initialize project state with stage tracking
        print("Setting up project state with stage tracking...")
        current_state = initialize_project_state(
            self.project_name, 
            self.survey_file, 
            self.data_file
        )
        
        # Save initial state
        self.save_state_snapshot(current_state, "INITIAL_STATE", "INITIALIZATION")
        print(f"Initial state saved with {len([k for k, v in current_state.items() if v is not None])} non-null fields")
        
        # Stage 1 nodes in execution order
        stage1_nodes = [
            (load_survey_node, "LOAD_SURVEY"),
            (load_data_node, "LOAD_DATA"),
            (parse_survey_node, "PARSE_SURVEY"),
            (survey_context_node, "EXTRACT_SURVEY_CONTEXT"),
            (get_open_column_node, "GET_OPEN_COLUMNS"),
            (question_data_matcher_node, "QUESTION_DATA_MATCHER"),
            (survey_data_integrate_node, "SURVEY_DATA_INTEGRATE"),
            (stage1_memory_flush_node, "STAGE1_MEMORY_FLUSH"),
            (memory_status_check_node, "MEMORY_STATUS_CHECK")
        ]
        
        total_start_time = datetime.now()
        
        # Execute each node with detailed logging
        for i, (node_func, node_name) in enumerate(stage1_nodes, 1):
            print(f"\n{'>' * 20} STEP {i}/{len(stage1_nodes)}: {node_name} {'<' * 20}")
            
            # 에러가 이미 발생했다면 중단
            if current_state.get('error'):
                print(f"STOPPING: Previous error: {current_state['error']}")
                break
            
            # 노드 실행
            current_state = self.execute_node_with_logging(node_func, node_name, current_state)
            
            # Pipeline status 출력
            try:
                print_pipeline_status(current_state)
            except Exception as e:
                print(f"WARNING: Pipeline status warning: {e}")
        
        total_end_time = datetime.now()
        total_execution_time = (total_end_time - total_start_time).total_seconds()
        
        # Final state 저장
        self.save_state_snapshot(current_state, "FINAL_STATE", "COMPLETION", total_execution_time)
        
        # Test summary 생성
        self.generate_test_summary(current_state, total_execution_time)
        
        return current_state
    
    def generate_test_summary(self, final_state: Dict[str, Any], total_time: float):
        """테스트 결과 요약 생성"""
        summary_path = self.output_dir / "test_summary.json"
        
        summary = {
            "test_metadata": {
                "test_id": self.test_id,
                "project_name": self.project_name,
                "survey_file": self.survey_file,
                "data_file": self.data_file,
                "total_execution_time": total_time,
                "timestamp": datetime.now().isoformat(),
                "output_directory": str(self.output_dir)
            },
            "execution_results": {
                "total_nodes": len(self.node_results),
                "successful_nodes": len([r for r in self.node_results if r['success']]),
                "failed_nodes": len([r for r in self.node_results if not r['success']]),
                "total_llm_calls": len(final_state.get('llm_logs') or []),
                "total_llm_cost": final_state.get('total_llm_cost_usd', 0),
                "pipeline_id": final_state.get('pipeline_id'),
                "final_stage": final_state.get('current_stage'),
                "final_error": final_state.get('error')
            },
            "node_results": self.node_results,
            "stage_snapshots": self.stage_snapshots,
            "error_states": self.error_states,
            "final_state_summary": {
                "total_fields": len(final_state),
                "non_null_fields": len([k for k, v in final_state.items() if v is not None]),
                "field_names": list(final_state.keys())
            }
        }
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 콘솔에 요약 출력
        print(f"\n{'=' * 80}")
        print(f"TEST SUMMARY")
        print(f"{'=' * 80}")
        print(f"🔬 Test ID: {self.test_id}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"📋 Nodes Executed: {len(self.node_results)}")
        print(f"✅ Successful: {len([r for r in self.node_results if r['success']])}")
        print(f"❌ Failed: {len([r for r in self.node_results if not r['success']])}")
        print(f"💰 Total LLM Cost: ${final_state.get('total_llm_cost_usd', 0):.6f}")
        print(f"📞 Total LLM Calls: {len(final_state.get('llm_logs') or [])}")
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"📄 Summary File: {summary_path}")
        
        # History file 정보
        if final_state.get('pipeline_id'):
            try:
                history_manager = get_or_create_history_manager(final_state['pipeline_id'])
                history_file = history_manager.get_history_file_path()
                print(f"📜 History File: {history_file}")
            except Exception as e:
                print(f"⚠️  History file info error: {e}")
        
        # 실패한 노드 정보
        failed_nodes = [r for r in self.node_results if not r['success']]
        if failed_nodes:
            print(f"\n❌ Failed Nodes Details:")
            for node in failed_nodes:
                print(f"  • {node['node_name']}: {node['error']}")
        
        print(f"{'=' * 80}")

def main():
    """메인 테스트 실행 함수"""
    print("Starting Stage 1 Deep Test with Enhanced Logging")
    
    # 실제 데이터로 테스트 (기존 test 프로젝트 사용)
    tester = Stage1DeepTester(
        project_name="test",  # 기존에 있는 test 프로젝트 사용
        survey_file="test.txt",
        data_file="-SUV_776부.xlsx"
    )
    
    try:
        final_state = tester.run_full_stage1_test()
        
        print("\n🎉 Stage 1 Deep Test Completed!")
        print(f"📁 All logs and states saved to: {tester.output_dir}")
        
        return 0 if not final_state.get('error') else 1
        
    except Exception as e:
        print(f"\nERROR: Deep test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())