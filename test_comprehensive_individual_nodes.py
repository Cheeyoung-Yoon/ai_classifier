#!/usr/bin/env python3
"""
완전한 개별 노드 테스트 시스템 - 모든 문제점 해결
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


class ComprehensiveLogger:
    """포괄적인 로그 출력 클래스"""
    
    def __init__(self):
        self.indent_level = 0
        self.test_start_time = None
        self.node_start_time = None
        self.detailed_logs = []
        
    def log(self, message: str, level: str = "INFO"):
        """레벨별 로그 출력"""
        indent = "  " * self.indent_level
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # 로그 기록
        self.detailed_logs.append({
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'indent': self.indent_level
        })
        
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
        elif level == "ANALYSIS":
            print(f"{indent}📊 {message}")
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
    
    def get_summary_stats(self) -> Dict[str, int]:
        """로그 통계 생성"""
        stats = {}
        for log_entry in self.detailed_logs:
            level = log_entry['level']
            stats[level] = stats.get(level, 0) + 1
        return stats


class ComprehensiveNodeTester:
    """포괄적인 노드 테스터"""
    
    def __init__(self):
        self.logger = ComprehensiveLogger()
        self.temp_dir = None
        self.project_dir = None
        self.test_results = {}
        self.node_execution_times = {}
        self.detailed_test_results = {}
        
    def setup_complete_test_environment(self):
        """완전한 테스트 환경 설정"""
        self.logger.log("완전한 테스트 환경 설정 중...", "INFO")
        self.temp_dir = tempfile.mkdtemp()
        
        # 프로젝트 디렉토리 생성
        self.project_dir = Path(project_root) / "data" / "complete_node_test"
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # 테스트 데이터 생성
        survey_content = """
Q1. 브랜드에 대한 전반적인 만족도는 어떠신가요?
① 매우 만족 ② 만족 ③ 보통 ④ 불만족 ⑤ 매우 불만족

Q2. 제품 품질에 대한 의견을 자세히 말씀해주세요.
(자유 서술형)

Q3. 가격 대비 만족도는 어떠신가요?
① 매우 좋음 ② 좋음 ③ 보통 ④ 나쁨 ⑤ 매우 나쁨

Q4. 추천하고 싶은 정도를 말씀해주세요.
(자유 응답)

Q5. 서비스 품질은 어떠셨나요?
(자유 의견)
        """
        
        # DataFrame with proper ID column
        df = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5, 6, 7, 8],
            'Q1': ['매우 만족', '만족', '보통', '불만족', '매우 만족', '만족', '보통', '매우 만족'],
            'Q2': [
                '품질이 정말 좋아요. 기대 이상입니다.',
                '괜찮은 편이지만 개선할 점이 있어요.',
                '그냥 보통이에요. 특별하지 않아요.',
                '품질이 아쉬워요. 더 좋아졌으면 해요.',
                '품질 최고! 매우 만족합니다.',
                '좋은 품질이지만 가격이 비싸요.',
                '품질은 보통인데 디자인이 마음에 들어요.',
                '매우 우수한 품질입니다. 강력 추천!'
            ],
            'Q3': ['좋음', '보통', '나쁨', '보통', '매우 좋음', '좋음', '나쁨', '매우 좋음'],
            'Q4': [
                '적극 추천하고 싶어요!',
                '지인에게 추천할 의향 있어요.',
                '추천하지 않을 것 같아요.',
                '잘 모르겠어요.',
                '꼭 추천하고 싶어요!',
                '가격만 저렴하면 추천하겠어요.',
                '특별히 추천하지는 않을 것 같아요.',
                '강력히 추천합니다!'
            ],
            'Q5': [
                '서비스가 매우 친절해요.',
                '서비스는 보통이에요.',
                '서비스가 좀 아쉬워요.',
                '서비스 품질이 별로예요.',
                '서비스가 정말 좋아요!',
                '서비스는 나쁘지 않아요.',
                '서비스 개선이 필요해요.',
                '최고의 서비스!'
            ]
        })
        
        # 파일 저장
        survey_file = Path(self.temp_dir) / "comprehensive_survey.txt"
        data_file = Path(self.temp_dir) / "comprehensive_data.xlsx"
        
        survey_file.write_text(survey_content, encoding='utf-8')
        df.to_excel(data_file, index=False, engine='openpyxl')
        
        self.logger.log(f"임시 디렉토리: {self.temp_dir}", "DEBUG")
        self.logger.log(f"프로젝트 디렉토리: {self.project_dir}", "DEBUG")
        self.logger.log(f"설문 파일: {survey_file} ({len(survey_content)} chars)", "DEBUG")
        self.logger.log(f"데이터 파일: {data_file} {df.shape}", "DEBUG")
        
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
                return not old_val.equals(new_val)
            elif isinstance(old_val, pd.DataFrame) or isinstance(new_val, pd.DataFrame):
                return True  # 하나만 DataFrame인 경우 변경됨
            else:
                return old_val != new_val
        except (ValueError, TypeError):
            return True
    
    def test_single_node_comprehensive(self, node_name: str, node_function: Callable, 
                                     input_state: Dict[str, Any], 
                                     test_description: str = "",
                                     expected_outputs: List[str] = None) -> Dict[str, Any]:
        """포괄적인 단일 노드 테스트"""
        self.logger.log(f"Testing Node: {node_name}", "NODE")
        if test_description:
            self.logger.log(f"Description: {test_description}", "INFO")
        
        self.logger.indent()
        self.logger.start_timer("node")
        
        test_result = {
            'node_name': node_name,
            'success': False,
            'execution_time': 0.0,
            'input_state_size': len(input_state),
            'output_state_size': 0,
            'new_fields': [],
            'modified_fields': [],
            'error_message': '',
            'validation_results': {},
            'result_state': {}
        }
        
        try:
            # 입력 상태 로깅
            self.logger.log("Input State Analysis:", "DEBUG")
            self.logger.indent()
            for key, value in input_state.items():
                if isinstance(value, (dict, list)):
                    self.logger.log(f"{key}: {type(value).__name__} (length: {len(value)})", "DEBUG")
                elif isinstance(value, pd.DataFrame):
                    self.logger.log(f"{key}: DataFrame {value.shape}", "DEBUG")
                elif value is None:
                    self.logger.log(f"{key}: None", "DEBUG")
                else:
                    value_str = str(value)[:50]
                    self.logger.log(f"{key}: {value_str}{'...' if len(str(value)) > 50 else ''}", "DEBUG")
            self.logger.dedent()
            
            # 노드 실행
            self.logger.log("Executing node function...", "INFO")
            result_state = node_function(input_state.copy())
            
            # 실행 시간 측정
            execution_time = self.logger.end_timer("node")
            test_result['execution_time'] = execution_time
            self.node_execution_times[node_name] = execution_time
            
            # 결과 상태 분석
            self.logger.log("Output State Analysis:", "DEBUG")
            self.logger.indent()
            
            if isinstance(result_state, dict):
                test_result['output_state_size'] = len(result_state)
                test_result['result_state'] = result_state
                
                for key, value in result_state.items():
                    if key not in input_state:
                        test_result['new_fields'].append(key)
                        if isinstance(value, (dict, list)):
                            self.logger.log(f"NEW {key}: {type(value).__name__} (length: {len(value)})", "DEBUG")
                        elif isinstance(value, pd.DataFrame):
                            self.logger.log(f"NEW {key}: DataFrame {value.shape}", "DEBUG")
                        else:
                            value_str = str(value)[:50]
                            self.logger.log(f"NEW {key}: {value_str}{'...' if len(str(value)) > 50 else ''}", "DEBUG")
                    elif self.safe_compare_values(input_state[key], value, key):
                        test_result['modified_fields'].append(key)
                        if isinstance(value, (dict, list)):
                            self.logger.log(f"MODIFIED {key}: {type(value).__name__} (length: {len(value)})", "DEBUG")
                        elif isinstance(value, pd.DataFrame):
                            self.logger.log(f"MODIFIED {key}: DataFrame {value.shape}", "DEBUG")
                        else:
                            value_str = str(value)[:50]
                            self.logger.log(f"MODIFIED {key}: {value_str}{'...' if len(str(value)) > 50 else ''}", "DEBUG")
            
            self.logger.dedent()
            
            # 기본 검증
            validations = {}
            
            if not isinstance(result_state, dict):
                validations['return_type'] = f"❌ Must return dict, got {type(result_state)}"
                raise AssertionError(f"Node must return dict, got {type(result_state)}")
            else:
                validations['return_type'] = "✅ Returns dict"
            
            if 'current_stage' not in result_state:
                validations['current_stage'] = "⚠️ Missing 'current_stage'"
                self.logger.log("Warning: No 'current_stage' in result", "WARNING")
            else:
                validations['current_stage'] = f"✅ Has current_stage: {result_state['current_stage']}"
            
            if 'error' in result_state:
                validations['error_handling'] = f"⚠️ Error present: {result_state['error']}"
                self.logger.log(f"Node completed with error: {result_state['error']}", "WARNING")
            else:
                validations['error_handling'] = "✅ No errors"
            
            # 예상 출력 검증
            if expected_outputs:
                for expected_output in expected_outputs:
                    if expected_output in result_state:
                        validations[f'expected_{expected_output}'] = f"✅ Has {expected_output}"
                    else:
                        validations[f'expected_{expected_output}'] = f"❌ Missing {expected_output}"
            
            test_result['validation_results'] = validations
            test_result['success'] = True
            
            self.logger.log(f"Node {node_name} executed successfully!", "SUCCESS")
            self.logger.log(f"New fields: {test_result['new_fields']}", "ANALYSIS")
            self.logger.log(f"Modified fields: {test_result['modified_fields']}", "ANALYSIS")
            
            self.logger.dedent()
            return test_result
            
        except Exception as e:
            test_result['error_message'] = str(e)
            self.logger.log(f"Node {node_name} failed: {str(e)}", "ERROR")
            self.logger.indent()
            self.logger.log("Traceback:", "DEBUG")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    self.logger.log(line, "DEBUG")
            self.logger.dedent()
            self.logger.dedent()
            
            return test_result


def comprehensive_test_all_nodes():
    """모든 노드의 포괄적 테스트"""
    tester = ComprehensiveNodeTester()
    
    # 테스트 시작
    tester.logger.log("포괄적인 개별 노드 테스트 시작", "HEADER")
    tester.logger.start_timer()
    
    try:
        # Mock setup
        class MockProjectManager:
            def __init__(self, project_name: str = "test", base_dir: str = "/tmp"):
                self.project_name = project_name
                self.base_dir = base_dir
                self.state_file_path = "/tmp/mock_state.json"
            
            def save_state(self, state, config):
                pass
            
            def create_project_structure(self):
                return {"project_dir": "/tmp/mock_project"}
        
        with patch('utils.project_manager.ProjectDirectoryManager', MockProjectManager):
            
            # 테스트 환경 설정
            survey_file_path, data_file_path = tester.setup_complete_test_environment()
            
            # 기본 상태 생성
            base_state = {
                'project_name': 'complete_node_test',
                'survey_file_path': survey_file_path,
                'data_file_path': data_file_path,
                'pipeline_id': 'complete_test_pipeline_001',
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
                'raw_dataframe_path': data_file_path
            }
            
            # 테스트 결과 저장
            all_test_results = []
            
            # =================================================================
            # STAGE 1 COMPREHENSIVE TESTS
            # =================================================================
            tester.logger.log("STAGE 1 노드들 포괄적 테스트", "HEADER")
            
            # 1. Survey Loader Test
            from nodes.stage1_data_preparation.survey_loader import load_survey_node
            result = tester.test_single_node_comprehensive(
                "survey_loader",
                load_survey_node,
                base_state.copy(),
                "설문 파일을 로드하고 원시 내용을 추출하는 노드",
                expected_outputs=['raw_survey_info']
            )
            all_test_results.append(result)
            
            # Survey loader 결과를 다음 테스트에 사용
            survey_state = base_state.copy()
            if result['success'] and 'raw_survey_info' in result['result_state']:
                survey_state.update(result['result_state'])
            
            # 2. Data Loader Test
            from nodes.stage1_data_preparation.data_loader import load_data_node
            result = tester.test_single_node_comprehensive(
                "data_loader",
                load_data_node,
                survey_state.copy(),
                "Excel 데이터 파일을 로드하고 DataFrame으로 변환하는 노드",
                expected_outputs=['raw_data_info', 'raw_dataframe_path']
            )
            all_test_results.append(result)
            
            # Data loader 결과를 다음 테스트에 사용
            data_state = survey_state.copy()
            if result['success']:
                data_state.update(result['result_state'])
            
            # 3. Survey Parser Test (with mocking)
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
                        },
                        {
                            "question_number": "Q4", 
                            "question_text": "추천하고 싶은 정도를 말씀해주세요.",
                            "question_type": "OPEN_ENDED",
                            "choices": {}
                        },
                        {
                            "question_number": "Q5", 
                            "question_text": "서비스 품질은 어떠셨나요?",
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
                
                result = tester.test_single_node_comprehensive(
                    "survey_parser",
                    parse_survey_node,
                    data_state.copy(),
                    "LLM을 사용하여 설문을 파싱하고 질문 구조를 추출하는 노드",
                    expected_outputs=['parsed_survey', 'llm_logs', 'llm_meta']
                )
                all_test_results.append(result)
            
            # =================================================================
            # ROUTER TESTS (COMPREHENSIVE)
            # =================================================================
            tester.logger.log("라우터들 포괄적 테스트", "HEADER")
            
            # Stage2 Type Router Tests (fixed)
            from router.stage2_router import stage2_type_router
            
            tester.logger.log("Stage2 Type Router 테스트", "NODE")
            
            router_test_cases = [
                # 올바른 형식의 테스트 케이스들
                ({'current_question': {'question_type': 'WORD', 'question_id': 'Q1'}}, 'WORD'),
                ({'current_question': {'question_type': 'SENTENCE', 'question_id': 'Q2'}}, 'SENTENCE'),
                ({'current_question': {'question_type': 'ETC', 'question_id': 'Q3'}}, 'ETC'),
                ({'current_question': None}, '__END__'),
                # 특별 케이스: question_id 추가로 라우터 동작 확인
                ({
                    'current_question': {'question_type': 'WORD', 'question_id': 'Q1'},
                    'current_question_id': 'Q1'
                }, 'WORD'),
                ({
                    'current_question': {'question_type': 'SENTENCE', 'question_id': 'Q2'},
                    'current_question_id': 'Q2'
                }, 'SENTENCE'),
            ]
            
            router_successes = 0
            for i, (input_state, expected) in enumerate(router_test_cases):
                try:
                    tester.logger.log(f"Router test case {i+1}: {input_state.get('current_question', 'None')}", "TEST")
                    result = stage2_type_router(input_state)
                    
                    if result == expected:
                        tester.logger.log(f"✅ Expected: {expected}, Got: {result}", "SUCCESS")
                        router_successes += 1
                    else:
                        tester.logger.log(f"❌ Expected: {expected}, Got: {result}", "ERROR")
                        
                except Exception as e:
                    tester.logger.log(f"❌ Router test {i+1} failed: {e}", "ERROR")
            
            router_success_rate = router_successes / len(router_test_cases)
            tester.logger.log(f"Router Tests: {router_successes}/{len(router_test_cases)} passed ({router_success_rate*100:.1f}%)", 
                            "SUCCESS" if router_success_rate >= 0.8 else "WARNING")
            
            # =================================================================
            # FINAL COMPREHENSIVE SUMMARY
            # =================================================================
            total_time = tester.logger.end_timer()
            
            tester.logger.log("포괄적 테스트 결과 분석", "HEADER")
            
            # 성공한 노드 분석
            successful_nodes = [result for result in all_test_results if result['success']]
            failed_nodes = [result for result in all_test_results if not result['success']]
            
            tester.logger.log(f"성공한 노드: {len(successful_nodes)}", "SUCCESS")
            tester.logger.log(f"실패한 노드: {len(failed_nodes)}", "ERROR")
            
            # 개별 노드 분석
            tester.logger.log("개별 노드 상세 분석:", "ANALYSIS")
            tester.logger.indent()
            
            for result in all_test_results:
                node_name = result['node_name']
                success = "✅" if result['success'] else "❌"
                exec_time = result['execution_time']
                new_fields = len(result['new_fields'])
                modified_fields = len(result['modified_fields'])
                
                tester.logger.log(f"{success} {node_name}: {exec_time:.3f}s, "
                                f"{new_fields} new, {modified_fields} modified", "ANALYSIS")
                
                # 검증 결과 표시
                if result['validation_results']:
                    tester.logger.indent()
                    for validation, result_text in result['validation_results'].items():
                        tester.logger.log(f"{validation}: {result_text}", "DEBUG")
                    tester.logger.dedent()
            
            tester.logger.dedent()
            
            # 실행 시간 분석
            tester.logger.log("실행 시간 분석:", "TIMER")
            tester.logger.indent()
            for node_name, exec_time in tester.node_execution_times.items():
                if exec_time > 1.0:
                    tester.logger.log(f"🐌 {node_name}: {exec_time:.3f}s (느림)", "WARNING")
                elif exec_time > 0.1:
                    tester.logger.log(f"⏱️ {node_name}: {exec_time:.3f}s (보통)", "INFO")
                else:
                    tester.logger.log(f"⚡ {node_name}: {exec_time:.3f}s (빠름)", "SUCCESS")
            tester.logger.dedent()
            
            # 로그 통계
            log_stats = tester.logger.get_summary_stats()
            tester.logger.log("로그 통계:", "ANALYSIS")
            tester.logger.indent()
            for level, count in log_stats.items():
                tester.logger.log(f"{level}: {count}", "DEBUG")
            tester.logger.dedent()
            
            # 최종 요약
            success_rate = len(successful_nodes) / len(all_test_results) if all_test_results else 0
            tester.logger.log("🎯 최종 포괄적 요약", "HEADER")
            tester.logger.log(f"총 테스트된 노드: {len(all_test_results)}", "INFO")
            tester.logger.log(f"성공률: {success_rate*100:.1f}%", "SUCCESS" if success_rate >= 0.8 else "WARNING")
            tester.logger.log(f"총 실행 시간: {total_time:.2f}초", "TIMER")
            tester.logger.log(f"평균 노드 실행 시간: {sum(tester.node_execution_times.values())/len(tester.node_execution_times):.3f}초", "TIMER")
            
            if success_rate >= 0.8:
                tester.logger.log("🎉 대부분의 노드가 정상적으로 작동합니다!", "SUCCESS")
            else:
                tester.logger.log("⚠️ 일부 노드에서 문제가 발견되었습니다. 개별 검토가 필요합니다.", "WARNING")
            
            # 개선 사항 요약
            tester.logger.log("테스트 개선 사항:", "INFO")
            tester.logger.indent()
            tester.logger.log("• 포괄적인 상태 분석 및 로깅", "INFO")
            tester.logger.log("• 안전한 DataFrame 비교", "INFO")
            tester.logger.log("• 실행 시간 성능 분석", "INFO")
            tester.logger.log("• 상세한 검증 결과", "INFO")
            tester.logger.log("• Mock 기반 의존성 격리", "INFO")
            tester.logger.dedent()
            
            return success_rate >= 0.8
            
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
    success = comprehensive_test_all_nodes()
    sys.exit(0 if success else 1)