#!/usr/bin/env python3
"""
개선된 세부적인 개별 노드 테스트 시스템 (문제점 수정)
"""

import os
import sys
import time
import tempfile
import shutil
import traceback
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Callable, Tuple
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class DetailedLogger:
    """상세한 로그 출력을 위한 클래스"""
    
    def __init__(self):
        self.indent_level = 0
        self.test_start_time = None
        self.node_start_time = None
        
    def log(self, message: str, level: str = "INFO"):
        """레벨별 로그 출력"""
        indent = "  " * self.indent_level
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        if level == "HEADER":
            print(f"\n{'='*80}")
            print(f"{indent}🎯 {message}")
            print(f"{'='*80}")
        elif level == "NODE":
            print(f"\n{'-'*60}")
            print(f"{indent}🔧 {message}")
            print(f"{'-'*60}")
        elif level == "TEST":
            print(f"\n{indent}🧪 {message}")
        elif level == "SUCCESS":
            print(f"{indent}✅ {message}")
        elif level == "ERROR":
            print(f"{indent}❌ {message}")
        elif level == "WARNING":
            print(f"{indent}⚠️  {message}")
        elif level == "DEBUG":
            print(f"{indent}🔍 [{timestamp}] {message}")
        elif level == "INFO":
            print(f"{indent}ℹ️  {message}")
        elif level == "TIMER":
            print(f"{indent}⏱️  {message}")
        else:
            print(f"{indent}{message}")
    
    def start_timer(self, context: str = "test"):
        """타이머 시작"""
        if context == "node":
            self.node_start_time = time.time()
        else:
            self.test_start_time = time.time()
    
    def end_timer(self, context: str = "test") -> float:
        """타이머 종료 및 경과 시간 반환"""
        if context == "node" and self.node_start_time:
            elapsed = time.time() - self.node_start_time
            self.log(f"Node execution time: {elapsed:.3f}s", "TIMER")
            return elapsed
        elif self.test_start_time:
            elapsed = time.time() - self.test_start_time
            self.log(f"Test execution time: {elapsed:.3f}s", "TIMER")
            return elapsed
        return 0.0
    
    def indent(self):
        """인덴트 증가"""
        self.indent_level += 1
    
    def dedent(self):
        """인덴트 감소"""
        self.indent_level = max(0, self.indent_level - 1)


class NodeTestFixtures:
    """노드 테스트를 위한 데이터 픽스쳐"""
    
    @staticmethod
    def create_base_state() -> Dict[str, Any]:
        """기본 상태 생성"""
        return {
            'project_name': 'detailed_test_project',
            'survey_file_path': '/tmp/test_survey.txt',
            'data_file_path': '/tmp/test_data.xlsx',
            'pipeline_id': 'detailed_test_pipeline_001',
            'current_stage': 'INITIALIZATION',
            'total_llm_cost_usd': 0.0,
            'questions': {},
            'data': None,
            'survey_raw_content': '',
            'survey_context': '',
            'question_data_match': {},
            'open_columns': [],
            'question_processing_queue': [],
            'current_question_idx': 0,
            'current_question': None,
            'current_data_sample': [],
            'stage2_processing_complete': False,
            'raw_dataframe_path': '/tmp/test_data.xlsx'
        }
    
    @staticmethod
    def create_survey_content() -> str:
        """테스트용 설문 내용 생성"""
        return """
Q1. 브랜드에 대한 전반적인 만족도는 어떠신가요?
① 매우 만족 ② 만족 ③ 보통 ④ 불만족 ⑤ 매우 불만족

Q2. 제품 품질에 대한 의견을 자세히 말씀해주세요.
(자유 서술형)

Q3. 가격 대비 만족도는 어떠신가요?
① 매우 좋음 ② 좋음 ③ 보통 ④ 나쁨 ⑤ 매우 나쁨

Q4. 추천하고 싶은 정도를 말씀해주세요.
(자유 응답)
        """
    
    @staticmethod
    def create_test_dataframe_with_id() -> pd.DataFrame:
        """ID 컬럼이 포함된 테스트용 데이터프레임 생성"""
        return pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],  # SmartExcelLoader가 요구하는 ID 컬럼 추가
            'Q1': ['매우 만족', '만족', '보통', '불만족', '매우 만족'],
            'Q2': [
                '품질이 정말 좋아요. 기대 이상입니다.',
                '괜찮은 편이지만 개선할 점이 있어요.',
                '그냥 보통이에요. 특별하지 않아요.',
                '품질이 아쉬워요. 더 좋아졌으면 해요.',
                '품질 최고! 매우 만족합니다.'
            ],
            'Q3': ['좋음', '보통', '나쁨', '보통', '매우 좋음'],
            'Q4': [
                '적극 추천하고 싶어요!',
                '지인에게 추천할 의향 있어요.',
                '추천하지 않을 것 같아요.',
                '잘 모르겠어요.',
                '꼭 추천하고 싶어요!'
            ],
            'respondent_id': [1, 2, 3, 4, 5]
        })
    
    @staticmethod
    def create_questions_dict() -> Dict[str, Any]:
        """테스트용 질문 딕셔너리 생성"""
        return {
            'Q1': {
                'question_number': 'Q1',
                'question_text': '브랜드에 대한 전반적인 만족도는 어떠신가요?',
                'question_type': 'MULTIPLE_CHOICE',
                'choices': {
                    '1': '매우 만족',
                    '2': '만족',
                    '3': '보통',
                    '4': '불만족',
                    '5': '매우 불만족'
                }
            },
            'Q2': {
                'question_number': 'Q2',
                'question_text': '제품 품질에 대한 의견을 자세히 말씀해주세요.',
                'question_type': 'OPEN_ENDED',
                'choices': {}
            },
            'Q3': {
                'question_number': 'Q3',
                'question_text': '가격 대비 만족도는 어떠신가요?',
                'question_type': 'MULTIPLE_CHOICE',
                'choices': {
                    '1': '매우 좋음',
                    '2': '좋음',
                    '3': '보통',
                    '4': '나쁨',
                    '5': '매우 나쁨'
                }
            },
            'Q4': {
                'question_number': 'Q4',
                'question_text': '추천하고 싶은 정도를 말씀해주세요.',
                'question_type': 'OPEN_ENDED',
                'choices': {}
            }
        }


class ImprovedNodeTester:
    """개선된 개별 노드 테스터"""
    
    def __init__(self):
        self.logger = DetailedLogger()
        self.fixtures = NodeTestFixtures()
        self.temp_dir = None
        self.project_dir = None
        self.test_results = {}
        
    def setup_test_environment(self):
        """테스트 환경 설정 - 프로젝트 디렉토리도 생성"""
        self.logger.log("테스트 환경 설정 중...", "INFO")
        self.temp_dir = tempfile.mkdtemp()
        
        # 프로젝트 디렉토리 생성
        self.project_dir = Path(project_root) / "data" / "detailed_test_project"
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # 테스트 파일 생성
        survey_file = Path(self.temp_dir) / "test_survey.txt"
        data_file = Path(self.temp_dir) / "test_data.xlsx"
        
        survey_file.write_text(self.fixtures.create_survey_content(), encoding='utf-8')
        
        # ID 컬럼이 포함된 Excel 파일로 저장
        df = self.fixtures.create_test_dataframe_with_id()
        df.to_excel(data_file, index=False, engine='openpyxl')
        
        self.logger.log(f"임시 디렉토리: {self.temp_dir}", "DEBUG")
        self.logger.log(f"프로젝트 디렉토리: {self.project_dir}", "DEBUG")
        self.logger.log(f"설문 파일: {survey_file}", "DEBUG")
        self.logger.log(f"데이터 파일: {data_file}", "DEBUG")
        
        return str(survey_file), str(data_file)
    
    def cleanup_test_environment(self):
        """테스트 환경 정리"""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.logger.log("임시 테스트 환경 정리 완료", "INFO")
        
        if self.project_dir:
            shutil.rmtree(self.project_dir, ignore_errors=True)
            self.logger.log("프로젝트 테스트 환경 정리 완료", "INFO")
    
    def safe_compare_values(self, old_val, new_val, key: str) -> bool:
        """DataFrame 등을 안전하게 비교"""
        try:
            if isinstance(old_val, pd.DataFrame) and isinstance(new_val, pd.DataFrame):
                return old_val.equals(new_val)
            elif isinstance(old_val, pd.DataFrame) or isinstance(new_val, pd.DataFrame):
                return False  # 하나만 DataFrame인 경우
            else:
                return old_val != new_val
        except (ValueError, TypeError):
            # 비교할 수 없는 타입들은 변경된 것으로 간주
            return True
    
    def test_single_node(self, node_name: str, node_function: Callable, 
                        input_state: Dict[str, Any], 
                        test_description: str = "") -> Tuple[bool, Dict[str, Any], str]:
        """단일 노드 테스트 (DataFrame 안전 비교 포함)"""
        self.logger.log(f"Testing Node: {node_name}", "NODE")
        if test_description:
            self.logger.log(f"Description: {test_description}", "INFO")
        
        self.logger.indent()
        self.logger.start_timer("node")
        
        try:
            # 입력 상태 로깅
            self.logger.log("Input State Analysis:", "DEBUG")
            self.logger.indent()
            for key, value in input_state.items():
                if isinstance(value, (dict, list)):
                    self.logger.log(f"{key}: {type(value).__name__} (length: {len(value)})", "DEBUG")
                elif isinstance(value, pd.DataFrame):
                    self.logger.log(f"{key}: DataFrame {value.shape}", "DEBUG")
                else:
                    self.logger.log(f"{key}: {value}", "DEBUG")
            self.logger.dedent()
            
            # 노드 실행
            self.logger.log("Executing node function...", "INFO")
            result_state = node_function(input_state.copy())
            
            # 실행 시간 측정
            execution_time = self.logger.end_timer("node")
            
            # 결과 상태 로깅 (안전한 비교)
            self.logger.log("Output State Analysis:", "DEBUG")
            self.logger.indent()
            
            if isinstance(result_state, dict):
                for key, value in result_state.items():
                    if key not in input_state:
                        if isinstance(value, (dict, list)):
                            self.logger.log(f"NEW {key}: {type(value).__name__} (length: {len(value)})", "DEBUG")
                        elif isinstance(value, pd.DataFrame):
                            self.logger.log(f"NEW {key}: DataFrame {value.shape}", "DEBUG")
                        else:
                            self.logger.log(f"NEW {key}: {value}", "DEBUG")
                    elif self.safe_compare_values(input_state[key], value, key):
                        if isinstance(value, (dict, list)):
                            self.logger.log(f"MODIFIED {key}: {type(value).__name__} (length: {len(value)})", "DEBUG")
                        elif isinstance(value, pd.DataFrame):
                            self.logger.log(f"MODIFIED {key}: DataFrame {value.shape}", "DEBUG")
                        else:
                            self.logger.log(f"MODIFIED {key}: {value}", "DEBUG")
            
            self.logger.dedent()
            
            # 기본 검증
            if not isinstance(result_state, dict):
                raise AssertionError(f"Node must return dict, got {type(result_state)}")
            
            if 'current_stage' not in result_state:
                self.logger.log("Warning: No 'current_stage' in result", "WARNING")
            
            self.logger.log(f"Node {node_name} executed successfully!", "SUCCESS")
            self.logger.dedent()
            
            return True, result_state, ""
            
        except Exception as e:
            self.logger.log(f"Node {node_name} failed: {str(e)}", "ERROR")
            self.logger.indent()
            self.logger.log("Traceback:", "DEBUG")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    self.logger.log(line, "DEBUG")
            self.logger.dedent()
            self.logger.dedent()
            
            return False, {}, str(e)


class Stage1NodeTestsFixed:
    """수정된 Stage 1 노드들의 개별 테스트"""
    
    def __init__(self, tester: ImprovedNodeTester):
        self.tester = tester
        self.logger = tester.logger
    
    def test_survey_loader_node(self, survey_file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Survey Loader 노드 테스트"""
        from nodes.stage1_data_preparation.survey_loader import load_survey_node
        
        state = self.tester.fixtures.create_base_state()
        state['survey_file_path'] = survey_file_path
        
        success, result, error = self.tester.test_single_node(
            "survey_loader", 
            load_survey_node, 
            state,
            "설문 파일을 로드하고 원시 내용을 추출"
        )
        
        if success:
            # 추가 검증
            if 'raw_survey_info' not in result:
                self.logger.log("Missing 'raw_survey_info' in result", "ERROR")
                success = False
            else:
                survey_info = result['raw_survey_info']
                if 'text' in survey_info and len(survey_info['text']) > 0:
                    self.logger.log(f"Loaded survey content ({len(survey_info['text'])} chars)", "SUCCESS")
                else:
                    self.logger.log("Empty survey content", "WARNING")
        
        return success, result
    
    def test_data_loader_node(self, data_file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Data Loader 노드 테스트"""
        from nodes.stage1_data_preparation.data_loader import load_data_node
        
        state = self.tester.fixtures.create_base_state()
        state['data_file_path'] = data_file_path
        
        success, result, error = self.tester.test_single_node(
            "data_loader",
            load_data_node,
            state,
            "데이터 파일을 로드하고 DataFrame으로 변환"
        )
        
        if success:
            # 추가 검증
            if 'raw_data_info' not in result:
                self.logger.log("Missing 'raw_data_info' in result", "ERROR")
                success = False
            elif 'error' in result:
                self.logger.log(f"Data loading error: {result['error']}", "ERROR")
                success = False
            else:
                data_info = result['raw_data_info']
                if 'dataframe_path' in data_info:
                    self.logger.log(f"Data loaded successfully: {data_info['dataframe_path']}", "SUCCESS")
                else:
                    self.logger.log("Missing dataframe_path in data_info", "WARNING")
        
        return success, result
    
    def test_survey_parser_node_with_valid_state(self, survey_info: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Survey Parser 노드 테스트 (유효한 상태로)"""
        from nodes.stage1_data_preparation.survey_parser import parse_survey_node
        
        with patch('io_layer.llm.client.LLMClient') as mock_llm_client, \
             patch('config.prompt.prompt_loader.resolve_branch') as mock_resolve_branch:
            
            # Mock 설정
            mock_response = {
                "raw": "Survey parsed successfully",
                "parsed": [
                    {
                        "question_number": "Q1",
                        "question_text": "브랜드에 대한 전반적인 만족도는 어떠신가요?",
                        "question_type": "MULTIPLE_CHOICE",
                        "choices": {"1": "매우 만족", "2": "만족", "3": "보통", "4": "불만족", "5": "매우 불만족"}
                    },
                    {
                        "question_number": "Q2", 
                        "question_text": "제품 품질에 대한 의견을 자세히 말씀해주세요.",
                        "question_type": "OPEN_ENDED",
                        "choices": {}
                    }
                ]
            }
            
            mock_llm_instance = Mock()
            mock_llm_instance.chat.return_value = (mock_response, Mock())
            mock_llm_client.return_value = mock_llm_instance
            
            mock_resolve_branch.return_value = {
                'system': 'You are a survey parser',
                'user_template': 'Parse this survey: {survey_content}',
                'schema': Mock()
            }
            
            state = self.tester.fixtures.create_base_state()
            state['raw_survey_info'] = survey_info  # 이전 노드에서 생성된 정보 사용
            
            success, result, error = self.tester.test_single_node(
                "survey_parser",
                parse_survey_node,
                state,
                "LLM을 사용하여 설문을 파싱하고 질문 구조 추출"
            )
            
            if success:
                # 추가 검증
                if 'questions' not in result:
                    self.logger.log("Missing 'questions' in result", "ERROR")
                    success = False
                elif not isinstance(result['questions'], dict):
                    self.logger.log("Questions is not a dict", "ERROR")
                    success = False
                else:
                    questions = result['questions']
                    self.logger.log(f"Parsed {len(questions)} questions: {list(questions.keys())}", "SUCCESS")
        
        return success, result


class MockProjectManager:
    """프로젝트 매니저 Mock"""
    
    def __init__(self, project_name: str = "test", base_dir: str = "/tmp"):
        self.project_name = project_name
        self.base_dir = base_dir
        self.state_file_path = "/tmp/mock_state.json"
    
    def save_state(self, state, config):
        # 실제로 파일을 저장하지 않고 성공한 것처럼 처리
        pass
    
    def create_project_structure(self):
        return {"project_dir": "/tmp/mock_project"}


def main():
    """메인 테스트 함수 (수정된 버전)"""
    tester = ImprovedNodeTester()
    
    # 테스트 시작
    tester.logger.log("개선된 세부적인 개별 노드 테스트 시작", "HEADER")
    tester.logger.start_timer()
    
    try:
        # 프로젝트 매니저 Mock
        with patch('utils.project_manager.ProjectDirectoryManager', MockProjectManager):
            
            # 테스트 환경 설정
            survey_file_path, data_file_path = tester.setup_test_environment()
            
            # 테스트 클래스들 초기화
            stage1_tests = Stage1NodeTestsFixed(tester)
            
            # 테스트 결과 수집
            results = {}
            
            # =============================================================================
            # STAGE 1 NODE TESTS (FIXED)
            # =============================================================================
            tester.logger.log("STAGE 1 노드들 개별 테스트 (수정됨)", "HEADER")
            
            # 1. Survey Loader
            success, survey_state = stage1_tests.test_survey_loader_node(survey_file_path)
            results['survey_loader'] = success
            
            # 2. Data Loader  
            success, data_state = stage1_tests.test_data_loader_node(data_file_path)
            results['data_loader'] = success
            
            # 3. Survey Parser (with valid state)
            if survey_state and 'raw_survey_info' in survey_state:
                success, parser_state = stage1_tests.test_survey_parser_node_with_valid_state(
                    survey_state['raw_survey_info']
                )
                results['survey_parser'] = success
            else:
                tester.logger.log("Survey loader 실패로 parser 테스트 스킵", "WARNING")
                results['survey_parser'] = False
            
            # 간단한 통계형 노드 테스트들
            tester.logger.log("간단한 노드 테스트들", "HEADER")
            
            # Router 테스트
            from router.stage2_router import stage2_type_router
            from nodes.stage2_next_question import stage2_completion_router
            
            # 라우터 테스트
            tester.logger.log("라우터 테스트 (수정됨)", "NODE")
            
            # Stage2 Type Router 테스트 (수정된 버전)
            test_cases = [
                ({'current_question': {'question_type': 'WORD', 'question_id': 'Q1'}}, 'WORD'),
                ({'current_question': {'question_type': 'SENTENCE', 'question_id': 'Q2'}}, 'SENTENCE'),
                ({'current_question': {'question_type': 'ETC', 'question_id': 'Q3'}}, 'ETC'),
                ({'current_question': None}, '__END__'),
                ({'current_question': {'question_type': 'UNKNOWN', 'question_id': 'Q4'}}, 'ETC'),
            ]
            
            router_success = 0
            for i, (input_state, expected) in enumerate(test_cases):
                try:
                    result = stage2_type_router(input_state)
                    if result == expected:
                        tester.logger.log(f"Router test {i+1}: ✅ PASS", "SUCCESS")
                        router_success += 1
                    else:
                        tester.logger.log(f"Router test {i+1}: ❌ Expected {expected}, got {result}", "ERROR")
                except Exception as e:
                    tester.logger.log(f"Router test {i+1}: ❌ Error {e}", "ERROR")
            
            results['stage2_type_router'] = router_success == len(test_cases)
            
            # Completion Router 테스트 (수정된 버전)
            completion_test_cases = [
                ({'current_question_idx': 0, 'question_processing_queue': [{'q1': 'data'}, {'q2': 'data'}]}, 'CONTINUE'),
                ({'current_question_idx': 1, 'question_processing_queue': [{'q1': 'data'}, {'q2': 'data'}]}, 'CONTINUE'),
                ({'current_question_idx': 2, 'question_processing_queue': [{'q1': 'data'}, {'q2': 'data'}]}, '__END__'),
                ({'current_question_idx': 0, 'question_processing_queue': []}, '__END__'),
            ]
            
            completion_success = 0
            for i, (input_state, expected) in enumerate(completion_test_cases):
                try:
                    result = stage2_completion_router(input_state)
                    if result == expected:
                        tester.logger.log(f"Completion test {i+1}: ✅ PASS", "SUCCESS")
                        completion_success += 1
                    else:
                        tester.logger.log(f"Completion test {i+1}: ❌ Expected {expected}, got {result}", "ERROR")
                except Exception as e:
                    tester.logger.log(f"Completion test {i+1}: ❌ Error {e}", "ERROR")
            
            results['stage2_completion_router'] = completion_success == len(completion_test_cases)
            
            # =============================================================================
            # 최종 결과 리포트
            # =============================================================================
            total_time = tester.logger.end_timer()
            
            tester.logger.log("개선된 테스트 결과 요약", "HEADER")
            
            # 카테고리별 결과
            categories = {
                "Core Stage 1 Nodes": [
                    'survey_loader', 'data_loader', 'survey_parser'
                ],
                "Router Tests": [
                    'stage2_type_router', 'stage2_completion_router'
                ]
            }
            
            total_tests = 0
            total_passed = 0
            
            for category, node_list in categories.items():
                tester.logger.log(f"{category}:", "INFO")
                tester.logger.indent()
                
                category_passed = 0
                for node in node_list:
                    if node in results:
                        status = "✅ PASS" if results[node] else "❌ FAIL"
                        tester.logger.log(f"{node}: {status}", "INFO")
                        if results[node]:
                            category_passed += 1
                        total_tests += 1
                    else:
                        tester.logger.log(f"{node}: ⚠️ NOT TESTED", "WARNING")
                
                total_passed += category_passed
                tester.logger.log(f"Category Result: {category_passed}/{len(node_list)} passed", 
                                "SUCCESS" if category_passed == len(node_list) else "WARNING")
                tester.logger.dedent()
            
            # 전체 요약
            tester.logger.log("🎯 IMPROVED FINAL SUMMARY", "HEADER")
            tester.logger.log(f"Total Tests: {total_tests}", "INFO")
            tester.logger.log(f"Passed: {total_passed}", "SUCCESS")
            tester.logger.log(f"Failed: {total_tests - total_passed}", "ERROR")
            tester.logger.log(f"Success Rate: {(total_passed/total_tests)*100:.1f}%", "INFO")
            tester.logger.log(f"Total Runtime: {total_time:.2f} seconds", "TIMER")
            
            if total_passed == total_tests:
                tester.logger.log("🎉 모든 개선된 테스트가 통과했습니다!", "SUCCESS")
            else:
                tester.logger.log(f"⚠️ {total_tests - total_passed}개 테스트에서 여전히 문제가 있습니다.", "WARNING")
            
            # 개별 노드별 상세 분석
            tester.logger.log("🔍 개별 노드 상세 분석", "HEADER")
            
            if results.get('survey_loader'):
                tester.logger.log("✅ Survey Loader: 파일 로딩과 텍스트 추출이 정상 작동", "SUCCESS")
            
            if results.get('data_loader'):
                tester.logger.log("✅ Data Loader: Excel 파일 로딩과 DataFrame 생성이 정상 작동", "SUCCESS")
            
            if results.get('survey_parser'):
                tester.logger.log("✅ Survey Parser: LLM을 통한 설문 파싱이 정상 작동", "SUCCESS")
            
            if results.get('stage2_type_router'):
                tester.logger.log("✅ Type Router: 질문 타입별 라우팅이 정상 작동", "SUCCESS")
            
            if results.get('stage2_completion_router'):
                tester.logger.log("✅ Completion Router: 완료 상태 체크가 정상 작동", "SUCCESS")
            
            tester.logger.log("개선 사항:", "INFO")
            tester.logger.indent()
            tester.logger.log("• DataFrame 안전 비교 로직 추가", "INFO")
            tester.logger.log("• Excel 파일에 ID 컬럼 추가", "INFO")
            tester.logger.log("• 프로젝트 디렉토리 자동 생성", "INFO")
            tester.logger.log("• Stage tracker Mock 처리", "INFO")
            tester.logger.log("• 상태 의존성 문제 해결", "INFO")
            tester.logger.dedent()
            
            return total_passed == total_tests
        
    except Exception as e:
        tester.logger.log(f"전체 테스트 실행 중 오류 발생: {e}", "ERROR")
        tester.logger.indent()
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                tester.logger.log(line, "DEBUG")
        tester.logger.dedent()
        return False
        
    finally:
        # 테스트 환경 정리
        tester.cleanup_test_environment()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)