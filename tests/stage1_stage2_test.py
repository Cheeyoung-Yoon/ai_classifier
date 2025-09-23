"""
Stage 1 & Stage 2 Integration Test
각 노드별 처리 과정을 시각적으로 보여주는 테스트 파일

Stage 1: 데이터 준비 단계 (LLM 없음)
- survey_loader: 설문 파일 로드
- data_loader: 데이터 파일 로드
- survey_parser: 설문 파싱
- survey_context: 설문 컨텍스트 생성
- column_extractor: 오픈 컬럼 추출
- question_matcher: 질문-데이터 매칭
- memory_optimizer: 메모리 최적화

Stage 2: 데이터 전처리 단계 (LLM 사용)
- stage2_main: 메인 라우터
- stage2_word_node: 단어 단위 처리
- stage2_sentence_node: 문장 단위 처리
- stage2_etc_node: 기타 데이터 처리
"""

import sys
import os
import json
import time
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# 기본 임포트
try:
    from graph.state import initialize_project_state
    from utils.cost_tracker import print_pipeline_status
    from utils.stage_history_manager import get_or_create_history_manager
    from utils.project_manager import get_project_manager
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Working Directory:", os.getcwd())
    print("Project Root:", project_root)
    sys.exit(1)

class Stage1Stage2Tester:
    """Stage 1과 Stage 2 통합 테스트 클래스"""
    
    def __init__(self, project_name: str = "test"):
        self.project_name = project_name
        self.survey_filename = "test.txt"
        self.data_filename = "-SUV_776부.xlsx"
        self.start_time = datetime.now()
        self.node_results = {}
        
        # 저장을 위한 디렉토리 설정 - graph의 프로젝트 구조에 맞춤
        self.project_data_dir = project_root / "data" / self.project_name
        self.output_dir = self.project_data_dir / "test_outputs"
        self.state_dir = self.output_dir / "states"
        self.data_dir = self.output_dir / "data"
        
        # 프로젝트 매니저 초기화 (graph와 동일한 구조 사용)
        self.project_manager = get_project_manager(self.project_name, str(project_root))
        
        self._setup_output_directories()
        
    def _setup_output_directories(self):
        """출력 디렉토리 생성 - graph의 프로젝트 구조에 맞춤"""
        # 프로젝트 데이터 디렉토리 먼저 생성
        self.project_data_dir.mkdir(exist_ok=True)
        
        # 테스트 출력 디렉토리 생성
        self.output_dir.mkdir(exist_ok=True)
        self.state_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        print(f"📁 프로젝트 디렉토리: {self.project_data_dir}")
        print(f"📁 테스트 출력 디렉토리: {self.output_dir}")
        
    def save_state(self, state: Dict[Any, Any], stage_name: str, timestamp: str = None):
        """State를 graph의 프로젝트 구조에 맞춰 저장"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # 1. graph의 프로젝트 매니저를 사용한 정식 state 저장
        try:
            # state에 현재 stage 정보 추가
            state_copy = dict(state)
            state_copy["current_stage"] = stage_name
            state_copy["test_timestamp"] = timestamp
            
            # 프로젝트 매니저의 save_state 사용 (정식 경로)
            official_state_path = self.project_manager.save_state(state_copy, {"save_state_log": True})
            print(f"💾 {stage_name} 정식 State 저장: {official_state_path}")
            
        except Exception as e:
            print(f"❌ 정식 State 저장 실패: {e}")
            official_state_path = None
            
        # 2. 테스트용 추가 저장 (JSON과 pickle)
        json_filename = f"{stage_name}_test_state_{timestamp}.json"
        json_path = self.state_dir / json_filename
        
        pickle_filename = f"{stage_name}_test_state_{timestamp}.pkl"
        pickle_path = self.state_dir / pickle_filename
        
        try:
            # JSON용 데이터 준비 (직렬화 가능한 데이터만)
            json_state = {}
            for key, value in state.items():
                try:
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        json_state[key] = value
                    elif hasattr(value, 'to_dict'):  # DataFrame 등
                        json_state[f"{key}_info"] = f"DataFrame with shape: {value.shape if hasattr(value, 'shape') else 'N/A'}"
                    else:
                        json_state[f"{key}_type"] = str(type(value))
                except:
                    json_state[f"{key}_error"] = "Cannot serialize"
            
            # 테스트용 JSON 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_state, f, indent=2, ensure_ascii=False)
            
            # 테스트용 Pickle 저장 (전체 state)
            with open(pickle_path, 'wb') as f:
                pickle.dump(state, f)
                
            print(f"💾 {stage_name} 테스트용 추가 저장:")
            print(f"   JSON: {json_path}")
            print(f"   Pickle: {pickle_path}")
            
            return official_state_path, json_path, pickle_path
            
        except Exception as e:
            print(f"❌ 테스트용 State 저장 실패: {e}")
            return official_state_path, None, None
            
    def save_generated_data(self, state: Dict[Any, Any], stage_name: str, timestamp: str = None):
        """생성된 데이터 파일들을 저장"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        saved_files = []
        
        try:
            # DataFrame 저장 (전체)
            if 'df' in state and state['df'] is not None:
                df_filename = f"{stage_name}_dataframe_{timestamp}.csv"
                df_path = self.data_dir / df_filename
                state['df'].to_csv(df_path, index=False, encoding='utf-8-sig')
                saved_files.append(df_path)
                print(f"📊 전체 DataFrame 저장: {df_path}")
            
            # raw_dataframe_path에서 DataFrame 로드하여 각 질문별 CSV 생성
            raw_df_path = state.get('raw_dataframe_path')
            question_mapping = state.get('question_data_match')
            
            if raw_df_path and question_mapping and os.path.exists(raw_df_path):
                try:
                    # DataFrame 로드
                    df = pd.read_csv(raw_df_path)
                    print(f"📊 원본 DataFrame 로드: {len(df)} 행, {len(df.columns)} 열")
                    
                    # question_mapping이 문자열인 경우 JSON으로 파싱
                    if isinstance(question_mapping, str):
                        question_mapping = json.loads(question_mapping)
                    
                    # 각 질문별로 CSV 저장
                    question_csv_dir = self.data_dir / f"{stage_name}_questions"
                    question_csv_dir.mkdir(exist_ok=True)
                    
                    # question_mapping이 list 형태인 경우 처리
                    if isinstance(question_mapping, list):
                        for question_item in question_mapping:
                            if isinstance(question_item, dict) and 'question_number' in question_item and 'columns' in question_item:
                                question_id = question_item['question_number']
                                columns = question_item['columns']
                                
                                if columns:  # 컬럼이 있는 질문만 처리
                                    # ID 컬럼과 해당 질문의 컬럼들만 추출
                                    available_columns = ['ID'] + [col for col in columns if col in df.columns]
                                    
                                    if len(available_columns) > 1:  # ID 외에 실제 데이터 컬럼이 있는 경우
                                        question_df = df[available_columns]
                                        
                                        question_csv_filename = f"{question_id}_{timestamp}.csv"
                                        question_csv_path = question_csv_dir / question_csv_filename
                                        
                                        question_df.to_csv(question_csv_path, index=False, encoding='utf-8-sig')
                                        saved_files.append(question_csv_path)
                                        print(f"📋 질문 {question_id} CSV 저장: {question_csv_path} ({len(question_df)} 행, {len(available_columns)} 열)")
                                    else:
                                        print(f"⚠️ 질문 {question_id}: 사용 가능한 컬럼이 없음 (매핑된 컬럼: {columns})")
                    
                    # question_mapping이 dict 형태인 경우 처리 (이전 버전 호환성)
                    elif isinstance(question_mapping, dict):
                        for question_id, columns in question_mapping.items():
                            if columns:  # 컬럼이 있는 질문만 처리
                                # ID 컬럼과 해당 질문의 컬럼들만 추출
                                available_columns = ['ID'] + [col for col in columns if col in df.columns]
                                
                                if len(available_columns) > 1:  # ID 외에 실제 데이터 컬럼이 있는 경우
                                    question_df = df[available_columns]
                                    
                                    question_csv_filename = f"{question_id}_{timestamp}.csv"
                                    question_csv_path = question_csv_dir / question_csv_filename
                                    
                                    question_df.to_csv(question_csv_path, index=False, encoding='utf-8-sig')
                                    saved_files.append(question_csv_path)
                                    print(f"📋 질문 {question_id} CSV 저장: {question_csv_path} ({len(question_df)} 행, {len(available_columns)} 열)")
                                else:
                                    print(f"⚠️ 질문 {question_id}: 사용 가능한 컬럼이 없음 (매핑된 컬럼: {columns})")
                    
                except Exception as e:
                    print(f"❌ 질문별 CSV 저장 실패: {e}")
                    import traceback
                    print(f"   상세 오류: {traceback.format_exc()}")
            
            # 설문 데이터 저장
            if 'raw_survey_info' in state and state['raw_survey_info']:
                survey_filename = f"{stage_name}_survey_data_{timestamp}.json"
                survey_path = self.data_dir / survey_filename
                with open(survey_path, 'w', encoding='utf-8') as f:
                    json.dump(state['raw_survey_info'], f, indent=2, ensure_ascii=False)
                saved_files.append(survey_path)
                print(f"📋 설문 데이터 저장: {survey_path}")
            
            # 파싱된 설문 저장
            if 'parsed_survey' in state and state['parsed_survey']:
                parsed_filename = f"{stage_name}_parsed_survey_{timestamp}.json"
                parsed_path = self.data_dir / parsed_filename
                with open(parsed_path, 'w', encoding='utf-8') as f:
                    json.dump(state['parsed_survey'], f, indent=2, ensure_ascii=False)
                saved_files.append(parsed_path)
                print(f"🔍 파싱된 설문 저장: {parsed_path}")
            
            # 질문-컬럼 매핑 저장
            if 'question_data_match' in state and state['question_data_match']:
                mapping_filename = f"{stage_name}_question_mapping_{timestamp}.json"
                mapping_path = self.data_dir / mapping_filename
                mapping_data = state['question_data_match']
                if isinstance(mapping_data, str):
                    try:
                        mapping_data = json.loads(mapping_data)
                    except:
                        pass
                with open(mapping_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping_data, f, indent=2, ensure_ascii=False)
                saved_files.append(mapping_path)
                print(f"🎯 질문 매핑 저장: {mapping_path}")
            
            # Stage2 처리 결과 저장
            if stage_name == "stage2" and 'stage2_processed_data' in state:
                stage2_filename = f"{stage_name}_processed_data_{timestamp}.json"
                stage2_path = self.data_dir / stage2_filename
                with open(stage2_path, 'w', encoding='utf-8') as f:
                    json.dump(state['stage2_processed_data'], f, indent=2, ensure_ascii=False)
                saved_files.append(stage2_path)
                print(f"⚙️ Stage2 처리 데이터 저장: {stage2_path}")
                
            return saved_files
            
        except Exception as e:
            print(f"❌ 데이터 저장 실패: {e}")
            import traceback
            print(f"   상세 오류: {traceback.format_exc()}")
            return []
        
    def print_separator(self, title: str, char: str = "="):
        """구분선과 제목 출력"""
        print(f"\n{char * 60}")
        print(f"🔥 {title}")
        print(f"{char * 60}")
        
    def print_node_header(self, node_name: str, stage: str):
        """노드 실행 헤더 출력"""
        print(f"\n{'▶' * 3} [{stage}] {node_name} 실행 중...")
        print(f"⏰ 시작 시간: {datetime.now().strftime('%H:%M:%S')}")
        
    def print_node_result(self, node_name: str, state: Dict[Any, Any], execution_time: float):
        """노드 실행 결과 출력"""
        print(f"✅ {node_name} 완료 (소요시간: {execution_time:.2f}초)")
        
        # 메모리 사용량 출력
        if hasattr(state, 'memory_usage'):
            print(f"💾 메모리 사용량: {state.get('memory_usage', 'N/A')}")
            
        # 주요 결과 데이터 출력
        self._print_key_results(node_name, state)
        print("-" * 50)
        
    def _print_key_results(self, node_name: str, state: Dict[Any, Any]):
        """노드별 주요 결과 출력"""
        if "survey_loader" in node_name:
            survey_data = state.get("survey_data", {})
            print(f"📋 설문 로드 결과: {len(survey_data)} 개 항목")
            
        elif "data_loader" in node_name:
            df = state.get("df")
            if df is not None:
                print(f"📊 데이터 로드 결과: {len(df)} 행, {len(df.columns)} 열")
            
        elif "survey_parser" in node_name:
            parsed_survey = state.get("parsed_survey", {})
            print(f"🔍 파싱된 질문 수: {len(parsed_survey)}")
            
        elif "survey_context" in node_name:
            survey_context = state.get("survey_context", "")
            print(f"📝 생성된 컨텍스트 길이: {len(survey_context)} 문자")
            
        elif "column_extractor" in node_name:
            open_columns = state.get("open_columns", [])
            print(f"🔓 추출된 오픈 컬럼: {len(open_columns)} 개")
            
        elif "question_matcher" in node_name:
            matched_questions = state.get("matched_questions", {})
            print(f"🎯 매칭된 질문: {len(matched_questions)} 개")
            
        elif "stage2" in node_name:
            processed_data = state.get("stage2_processed_data", {})
            print(f"⚙️ Stage2 처리 결과: {len(processed_data)} 개 항목")
            
    def run_stage1_tests(self):
        """Stage 1 노드들을 순차적으로 테스트"""
        self.print_separator("STAGE 1: 데이터 준비 단계 테스트", "=")
        
        # 상태 초기화
        try:
            state = initialize_project_state(
                self.project_name, 
                self.survey_filename, 
                self.data_filename
            )
            print("✅ 초기 상태 생성 완료")
        except Exception as e:
            print(f"❌ 상태 초기화 실패: {e}")
            return None
            
        # Stage 1 노드들 순차 실행
        stage1_nodes = [
            ("survey_loader", "설문 파일 로더"),
            ("data_loader", "데이터 파일 로더"),
            ("survey_parser", "설문 파서"),
            ("survey_context", "설문 컨텍스트 생성기"),
            ("column_extractor", "오픈 컬럼 추출기"),
            ("question_matcher", "질문-데이터 매처"),
            ("memory_optimizer", "메모리 최적화기")
        ]
        
        for node_id, node_desc in stage1_nodes:
            if self._run_single_stage1_node(state, node_id, node_desc):
                print(f"✅ {node_desc} 성공")
            else:
                print(f"❌ {node_desc} 실패")
                break
        
        # Stage 1 완료 후 데이터 저장 및 Stage2를 위한 데이터 준비
        if state:
            self.print_separator("STAGE 1 데이터 저장 및 Stage2 준비", "-")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Stage2를 위한 질문 매핑 준비
            question_mapping = state.get('question_data_match')
            print(f"🔍 question_data_match: {question_mapping}")
            
            if question_mapping:
                try:
                    # question_mapping을 matched_questions 형태로 변환
                    if isinstance(question_mapping, str):
                        question_mapping = json.loads(question_mapping)
                    
                    print(f"🔍 파싱된 question_mapping 타입: {type(question_mapping)}")
                    print(f"🔍 파싱된 question_mapping: {question_mapping}")
                    
                    matched_questions = {}
                    
                    # question_mapping이 dict 형태인 경우 (새로운 형태)
                    if isinstance(question_mapping, dict):
                        # 'question_column_mapping' 키를 찾아서 처리
                        if 'question_column_mapping' in question_mapping:
                            column_mapping_str = question_mapping['question_column_mapping']
                            if isinstance(column_mapping_str, str):
                                column_mapping = json.loads(column_mapping_str)
                            else:
                                column_mapping = column_mapping_str
                                
                            print(f"🔍 column_mapping: {column_mapping}")
                            
                            # dict 형태의 매핑 처리
                            if isinstance(column_mapping, dict):
                                for question_id, columns in column_mapping.items():
                                    if columns:  # 컬럼이 있는 질문만 처리
                                        # 해당 질문의 type 찾기
                                        parsed_survey = state.get('parsed_survey', {})
                                        question_type = "ETC"  # 기본값
                                        
                                        if parsed_survey and 'parsed' in parsed_survey:
                                            questions = parsed_survey['parsed'].get('questions', [])
                                            for q in questions:
                                                if q.get('open_question_number') == question_id:
                                                    question_type = q.get('question_type', 'ETC').upper()
                                                    break
                                        
                                        matched_questions[question_id] = {
                                            'question_info': {
                                                'question_type': question_type,
                                                'open_question_number': question_id
                                            },
                                            'columns': columns
                                        }
                                        print(f"📝 질문 {question_id} 준비 완료 (타입: {question_type}, 컬럼: {len(columns)}개)")
                        else:
                            # question_mapping이 직접 매핑 정보인 경우
                            for question_id, columns in question_mapping.items():
                                if columns:  # 컬럼이 있는 질문만 처리
                                    parsed_survey = state.get('parsed_survey', {})
                                    question_type = "ETC"  # 기본값
                                    
                                    if parsed_survey and 'parsed' in parsed_survey:
                                        questions = parsed_survey['parsed'].get('questions', [])
                                        for q in questions:
                                            if q.get('open_question_number') == question_id:
                                                question_type = q.get('question_type', 'ETC').upper()
                                                break
                                    
                                    matched_questions[question_id] = {
                                        'question_info': {
                                            'question_type': question_type,
                                            'open_question_number': question_id
                                        },
                                        'columns': columns
                                    }
                                    print(f"📝 질문 {question_id} 준비 완료 (타입: {question_type}, 컬럼: {len(columns)}개)")
                    
                    # 기존 list 형태 처리
                    elif isinstance(question_mapping, list):
                        for item in question_mapping:
                            if isinstance(item, dict) and 'question_number' in item and 'columns' in item:
                                question_id = item['question_number']
                                columns = item['columns']
                                
                                if columns:  # 컬럼이 있는 질문만 처리
                                    # 해당 질문의 type 찾기
                                    parsed_survey = state.get('parsed_survey', {})
                                    question_type = "ETC"  # 기본값
                                    
                                    if parsed_survey and 'parsed' in parsed_survey:
                                        questions = parsed_survey['parsed'].get('questions', [])
                                        for q in questions:
                                            if q.get('open_question_number') == question_id:
                                                question_type = q.get('question_type', 'ETC').upper()
                                                break
                                    
                                    matched_questions[question_id] = {
                                        'question_info': {
                                            'question_type': question_type,
                                            'open_question_number': question_id
                                        },
                                        'columns': columns
                                    }
                                    print(f"📝 질문 {question_id} 준비 완료 (타입: {question_type}, 컬럼: {len(columns)}개)")
                    
                    # matched_questions를 state에 추가
                    state['matched_questions'] = matched_questions
                    print(f"✅ Stage2용 질문 매핑 완료: {len(matched_questions)}개 질문")
                    
                except Exception as e:
                    print(f"❌ Stage2 질문 준비 실패: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("❌ question_data_match가 없습니다.")
            
            # State 저장 (정식 + 테스트용)
            official_path, json_path, pickle_path = self.save_state(state, "stage1", timestamp)
            
            # 생성된 데이터 저장
            saved_files = self.save_generated_data(state, "stage1", timestamp)
            
            # 저장 결과를 node_results에 기록
            self.node_results["stage1_saved_files"] = {
                "official_state": str(official_path) if official_path else None,
                "test_state_json": str(json_path) if json_path else None,
                "test_state_pickle": str(pickle_path) if pickle_path else None,
                "data_files": [str(f) for f in saved_files]
            }
                
        return state
        
    def _run_single_stage1_node(self, state: Dict[Any, Any], node_id: str, node_desc: str) -> bool:
        """개별 Stage 1 노드 실행"""
        self.print_node_header(node_desc, "STAGE 1")
        start_time = time.time()
        
        try:
            # 절대 경로로 동적 임포트
            project_root_str = str(project_root)
            if project_root_str not in sys.path:
                sys.path.insert(0, project_root_str)
            
            if node_id == "survey_loader":
                # 직접 모듈 경로 지정
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "survey_loader", 
                    project_root / "nodes" / "stage1_data_preparation" / "survey_loader.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                state = module.load_survey_node(state)
                
            elif node_id == "data_loader":
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "data_loader", 
                    project_root / "nodes" / "stage1_data_preparation" / "data_loader.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                state = module.load_data_node(state)
                
            elif node_id == "survey_parser":
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "survey_parser", 
                    project_root / "nodes" / "stage1_data_preparation" / "survey_parser.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                state = module.parse_survey_node(state)
                
            elif node_id == "survey_context":
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "survey_context", 
                    project_root / "nodes" / "stage1_data_preparation" / "survey_context.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                state = module.survey_context_node(state)
                
            elif node_id == "column_extractor":
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "column_extractor", 
                    project_root / "nodes" / "stage1_data_preparation" / "column_extractor.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                state = module.get_open_column_node(state)
                
            elif node_id == "question_matcher":
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "question_matcher", 
                    project_root / "nodes" / "stage1_data_preparation" / "question_matcher.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                state = module.question_data_matcher_node(state)
                
            elif node_id == "memory_optimizer":
                print(f"⚠️  {node_desc} 스킵 (메모리 최적화는 선택적 실행)")
                execution_time = time.time() - start_time
                self.print_node_result(node_desc, state, execution_time)
                return True
                
            execution_time = time.time() - start_time
            self.print_node_result(node_desc, state, execution_time)
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ {node_desc} 실행 실패 (소요시간: {execution_time:.2f}초)")
            print(f"   오류: {str(e)}")
            import traceback
            print(f"   상세 오류: {traceback.format_exc()}")
            return False
            
    def run_stage2_tests(self, state: Dict[Any, Any]):
        """Stage 2 노드들을 테스트하고 각 질문별 처리를 진행합니다."""
        if state is None:
            print("❌ Stage 1 실행 결과가 없어 Stage 2를 실행할 수 없습니다.")
            return None
            
        print(f"\n{'='*60}")
        print("🎯 STAGE 2 TESTING 시작")
        print(f"{'='*60}")
        
        matched_questions = state.get('matched_questions', {})
        if not matched_questions:
            print("❌ matched_questions가 없어서 Stage2를 진행할 수 없습니다.")
            return state
        
        print(f"📊 처리할 질문 수: {len(matched_questions)}개")
        
        # 각 질문별로 Stage2 처리
        for question_id, question_data in matched_questions.items():
            self.print_separator(f"질문 {question_id} 처리", "=")
            
            # 현재 질문 정보를 state에 설정 (딥카피 문제 해결)
            current_state = state.copy()
            current_state['current_question_id'] = question_id
            current_state['current_question'] = question_data['question_info']
            current_state['current_columns'] = question_data['columns']
            
            # DataFrame이 없으면 raw_dataframe_path에서 로드
            if 'df' not in current_state and 'raw_dataframe_path' in current_state:
                try:
                    import pandas as pd
                    df_path = current_state['raw_dataframe_path']
                    df = pd.read_csv(df_path)
                    current_state['df'] = df
                    print(f"📊 DataFrame 로드 완료: {df.shape}")
                except Exception as e:
                    print(f"❌ DataFrame 로드 실패: {e}")
                    continue
            
            print(f"📝 질문 ID: {question_id}")
            print(f"📝 질문 타입: {question_data['question_info'].get('question_type', 'ETC')}")
            print(f"📝 처리할 컬럼 수: {len(question_data['columns'])}")
            print(f"🔍 DataFrame 존재 여부: {'df' in current_state}")
            if 'df' in current_state:
                df = current_state['df']
                print(f"🔍 DataFrame 타입: {type(df)}")
                if hasattr(df, 'shape'):
                    print(f"🔍 DataFrame 크기: {df.shape}")
            else:
                print("❌ state에 'df' 키가 없습니다.")
            
            # Stage2 Main 노드 실행
            self.print_separator("Stage2 Main 노드 실행", "-")
            try:
                from nodes.stage2_data_preprocessing.stage2_main import stage2_data_preprocessing_node
                stage2_state = stage2_data_preprocessing_node(current_state)
                if stage2_state:
                    print("✅ Stage2 Main 노드 실행 완료")
                    current_state.update(stage2_state)
                else:
                    print("❌ Stage2 Main 노드 실행 실패")
                    continue
            except Exception as e:
                print(f"❌ Stage2 Main 노드 실행 중 오류: {e}")
                continue
            
            # 질문 타입에 따라 적절한 Stage2 처리 노드 실행
            question_type = question_data['question_info'].get('question_type', 'ETC').upper()
            
            if question_type == 'WORD':
                self.print_separator("Stage2 Word 노드 실행", "-")
                try:
                    from nodes.stage2_data_preprocessing.stage2_word_node import stage2_word_node
                    processing_state = stage2_word_node(current_state)
                    if processing_state:
                        print("✅ Stage2 Word 노드 실행 완료")
                        current_state.update(processing_state)
                        self.save_question_results(current_state, question_id)
                    else:
                        print("❌ Stage2 Word 노드 실행 실패")
                        continue
                except Exception as e:
                    print(f"❌ Stage2 Word 노드 실행 중 오류: {e}")
                    continue
                    
            elif question_type == 'SENTENCE':
                self.print_separator("Stage2 Sentence 노드 실행", "-")
                try:
                    from nodes.stage2_data_preprocessing.stage2_sentence_node import stage2_sentence_node
                    processing_state = stage2_sentence_node(current_state)
                    if processing_state:
                        print("✅ Stage2 Sentence 노드 실행 완료")
                        current_state.update(processing_state)
                        self.save_question_results(current_state, question_id)
                    else:
                        print("❌ Stage2 Sentence 노드 실행 실패")
                        continue
                except Exception as e:
                    print(f"❌ Stage2 Sentence 노드 실행 중 오류: {e}")
                    continue
                    
            else:  # ETC or other types
                self.print_separator("Stage2 ETC 노드 실행", "-")
                try:
                    from nodes.stage2_data_preprocessing.stage2_etc_node import stage2_etc_node
                    processing_state = stage2_etc_node(current_state)
                    if processing_state:
                        print("✅ Stage2 ETC 노드 실행 완료")
                        current_state.update(processing_state)
                        self.save_question_results(current_state, question_id)
                    else:
                        print("❌ Stage2 ETC 노드 실행 실패")
                        continue
                except Exception as e:
                    print(f"❌ Stage2 ETC 노드 실행 중 오류: {e}")
                    continue
            
            print(f"🎯 질문 {question_id} 처리 완료\n")
        
        # 모든 질문 처리 완료 후 최종 state 업데이트
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.print_separator("STAGE 2 완료 후 데이터 저장", "-")
        # State 저장 (정식 + 테스트용)
        official_path, json_path, pickle_path = self.save_state(current_state, "stage2", timestamp)
        
        # 저장 결과를 node_results에 기록
        self.node_results["stage2_saved_files"] = {
            "official_state": str(official_path) if official_path else None,
            "test_state_json": str(json_path) if json_path else None,
            "test_state_pickle": str(pickle_path) if pickle_path else None,
            "data_files": []  # Stage2에서는 일반적으로 별도 데이터 파일을 생성하지 않음
        }
        
        return current_state
    
    def save_question_results(self, state, question_id):
        """질문별 처리 결과를 CSV로 저장합니다."""
        try:
            # processed_data가 있는지 확인
            processed_data = state.get('processed_data')
            if processed_data is None:
                print(f"❌ 질문 {question_id}의 processed_data가 없습니다.")
                return
            
            # DataFrame으로 변환
            if isinstance(processed_data, str):
                # JSON 문자열인 경우 파싱
                try:
                    import json
                    processed_data = json.loads(processed_data)
                except:
                    print(f"❌ 질문 {question_id}의 processed_data JSON 파싱 실패")
                    return
            
            # DataFrame 생성
            if isinstance(processed_data, list):
                df = pd.DataFrame(processed_data)
            elif isinstance(processed_data, dict):
                df = pd.DataFrame([processed_data])
            else:
                print(f"❌ 질문 {question_id}의 processed_data 형식이 올바르지 않습니다.")
                return
            
            if df.empty:
                print(f"❌ 질문 {question_id}의 processed_data가 비어있습니다.")
                return
            
            # 파일 경로 설정
            question_dir = self.project_data_dir / "processed_questions"
            question_dir.mkdir(exist_ok=True)
            
            csv_path = question_dir / f"question_{question_id}_processed.csv"
            
            # UTF-8-sig 인코딩으로 CSV 저장
            df.to_csv(csv_path, encoding='utf-8-sig', index=False)
            print(f"✅ 질문 {question_id} 처리 결과 저장: {csv_path}")
            print(f"📊 데이터 행 수: {len(df)}, 컬럼 수: {len(df.columns)}")
            
            if len(df.columns) > 0:
                print(f"📋 컬럼: {list(df.columns)}")
                
        except Exception as e:
            print(f"❌ 질문 {question_id} 결과 저장 실패: {e}")
            import traceback
            traceback.print_exc()
        
    def print_final_summary(self, final_state: Dict[Any, Any]):
        """최종 요약 출력"""
        self.print_separator("최종 테스트 요약", "=")
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        print(f"⏱️  총 실행 시간: {total_time:.2f}초")
        
        if final_state:
            print(f"✅ 최종 상태: 정상 완료")
            
            # 최종 데이터 요약
            df = final_state.get("df")
            if df is not None:
                print(f"📊 최종 데이터프레임: {len(df)} 행, {len(df.columns)} 열")
                
            survey_data = final_state.get("survey_data", {})
            print(f"📋 설문 데이터: {len(survey_data)} 개 항목")
            
            processed_data = final_state.get("stage2_processed_data", {})
            print(f"⚙️ Stage 2 처리 데이터: {len(processed_data)} 개 항목")
            
        else:
            print(f"❌ 최종 상태: 실행 실패")
        
        # 저장된 파일들 정보 출력
        self.print_separator("저장된 파일들", "-")
        print(f"📁 프로젝트 디렉토리: {self.project_data_dir}")
        print(f"📁 테스트 출력 디렉토리: {self.output_dir}")
        
        if "stage1_saved_files" in self.node_results:
            stage1_files = self.node_results["stage1_saved_files"]
            print(f"\n🔸 Stage 1 저장 파일들:")
            if stage1_files["official_state"]:
                print(f"   📄 정식 State: {Path(stage1_files['official_state']).name}")
            if stage1_files["test_state_json"]:
                print(f"   📄 테스트 State JSON: {Path(stage1_files['test_state_json']).name}")
            if stage1_files["test_state_pickle"]:
                print(f"   📦 테스트 State Pickle: {Path(stage1_files['test_state_pickle']).name}")
            for data_file in stage1_files["data_files"]:
                print(f"   📊 Data: {Path(data_file).name}")
        
        if "stage2_saved_files" in self.node_results:
            stage2_files = self.node_results["stage2_saved_files"]
            print(f"\n🔸 Stage 2 저장 파일들:")
            if stage2_files["official_state"]:
                print(f"   📄 정식 State: {Path(stage2_files['official_state']).name}")
            if stage2_files["test_state_json"]:
                print(f"   📄 테스트 State JSON: {Path(stage2_files['test_state_json']).name}")
            if stage2_files["test_state_pickle"]:
                print(f"   📦 테스트 State Pickle: {Path(stage2_files['test_state_pickle']).name}")
            for data_file in stage2_files["data_files"]:
                print(f"   📊 Data: {Path(data_file).name}")
        
        # 저장된 총 파일 수
        total_files = 0
        for stage_key in ["stage1_saved_files", "stage2_saved_files"]:
            if stage_key in self.node_results:
                stage_files = self.node_results[stage_key]
                # 정식 + 테스트용 state 파일들 카운트
                total_files += len([f for f in [
                    stage_files["official_state"], 
                    stage_files["test_state_json"], 
                    stage_files["test_state_pickle"]
                ] if f])
                total_files += len(stage_files["data_files"])
        
        print(f"\n💾 총 저장된 파일 수: {total_files}개")
        print(f"📍 주요 경로:")
        print(f"   - 정식 state.json: {self.project_manager.state_file_path}")
        print(f"   - state history: {Path(self.project_manager.project_dir) / 'state_history'}")
        print(f"{'=' * 60}")

def main():
    """메인 실행 함수"""
    print("🚀 Stage 1 & Stage 2 통합 테스트 시작")
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 테스터 인스턴스 생성
    tester = Stage1Stage2Tester("test")
    
    try:
        # Stage 1 실행
        stage1_result = tester.run_stage1_tests()
        
        # Stage 2 실행 (Stage 1이 성공한 경우만)
        if stage1_result:
            final_result = tester.run_stage2_tests(stage1_result)
        else:
            final_result = None
            
        # 최종 요약
        tester.print_final_summary(final_result)
        
    except KeyboardInterrupt:
        print("\n⚠️  사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()