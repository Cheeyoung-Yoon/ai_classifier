#!/usr/bin/env python3
"""
세부적인 개별 노드 테스트 시스템
모든 노드를 하나씩 개별적으로 테스트하고 상세한 로그를 출력
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
            'data_file_path': '/tmp/test_data.xlsx',  # Excel로 변경
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
            'raw_dataframe_path': '/tmp/test_data.xlsx'  # Excel로 변경
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
    def create_test_dataframe() -> pd.DataFrame:
        """테스트용 데이터프레임 생성"""
        return pd.DataFrame({
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


class IndividualNodeTester:
    """개별 노드 테스터"""
    
    def __init__(self):
        self.logger = DetailedLogger()
        self.fixtures = NodeTestFixtures()
        self.temp_dir = None
        self.test_results = {}
        
    def setup_test_environment(self):
        """테스트 환경 설정"""
        self.logger.log("테스트 환경 설정 중...", "INFO")
        self.temp_dir = tempfile.mkdtemp()
        
        # 테스트 파일 생성
        survey_file = Path(self.temp_dir) / "test_survey.txt"
        data_file = Path(self.temp_dir) / "test_data.xlsx"  # CSV에서 Excel로 변경
        
        survey_file.write_text(self.fixtures.create_survey_content(), encoding='utf-8')
        
        # Excel 파일로 저장
        df = self.fixtures.create_test_dataframe()
        df.to_excel(data_file, index=False, engine='openpyxl')
        
        self.logger.log(f"임시 디렉토리: {self.temp_dir}", "DEBUG")
        self.logger.log(f"설문 파일: {survey_file}", "DEBUG")
        self.logger.log(f"데이터 파일: {data_file}", "DEBUG")
        
        return str(survey_file), str(data_file)
    
    def cleanup_test_environment(self):
        """테스트 환경 정리"""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.logger.log("테스트 환경 정리 완료", "INFO")
    
    def test_single_node(self, node_name: str, node_function: Callable, 
                        input_state: Dict[str, Any], 
                        test_description: str = "") -> Tuple[bool, Dict[str, Any], str]:
        """단일 노드 테스트"""
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
            
            # 결과 상태 로깅
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
                    elif input_state[key] != value:
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


class Stage1NodeTests:
    """Stage 1 노드들의 개별 테스트"""
    
    def __init__(self, tester: IndividualNodeTester):
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
            if 'survey_raw_content' not in result:
                self.logger.log("Missing 'survey_raw_content' in result", "ERROR")
                success = False
            elif len(result['survey_raw_content']) == 0:
                self.logger.log("Empty survey content", "WARNING")
            else:
                self.logger.log(f"Loaded survey content ({len(result['survey_raw_content'])} chars)", "SUCCESS")
        
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
            if 'data' not in result:
                self.logger.log("Missing 'data' in result", "ERROR")
                success = False
            elif not isinstance(result['data'], pd.DataFrame):
                self.logger.log("Data is not a DataFrame", "ERROR")
                success = False
            elif len(result['data']) == 0:
                self.logger.log("Empty DataFrame", "WARNING")
            else:
                df = result['data']
                self.logger.log(f"Loaded DataFrame: {df.shape} ({list(df.columns)})", "SUCCESS")
        
        return success, result
    
    def test_survey_parser_node(self, survey_content: str) -> Tuple[bool, Dict[str, Any]]:
        """Survey Parser 노드 테스트"""
        from nodes.stage1_data_preparation.survey_parser import parse_survey_node
        
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
        
        with patch('io_layer.llm.client.LLMClient') as mock_llm_client, \
             patch('config.prompt.prompt_loader.resolve_branch') as mock_resolve_branch:
            
            mock_llm_instance = Mock()
            mock_llm_instance.chat.return_value = (mock_response, Mock())
            mock_llm_client.return_value = mock_llm_instance
            
            mock_resolve_branch.return_value = {
                'system': 'You are a survey parser',
                'user_template': 'Parse this survey: {survey_content}',
                'schema': Mock()
            }
            
            state = self.tester.fixtures.create_base_state()
            state['survey_raw_content'] = survey_content
            
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
                    
                    # 질문 구조 검증
                    for q_id, q_data in questions.items():
                        if 'question_type' not in q_data:
                            self.logger.log(f"Missing question_type for {q_id}", "WARNING")
        
        return success, result
    
    def test_survey_context_node(self, survey_content: str) -> Tuple[bool, Dict[str, Any]]:
        """Survey Context 노드 테스트"""
        from nodes.stage1_data_preparation.survey_context import survey_context_node
        
        with patch('io_layer.llm.client.LLMClient') as mock_llm_client, \
             patch('config.prompt.prompt_loader.resolve_branch') as mock_resolve_branch:
            
            mock_llm_instance = Mock()
            mock_llm_instance.chat.return_value = ("브랜드 만족도 및 제품 품질 평가 조사", Mock())
            mock_llm_client.return_value = mock_llm_instance
            
            mock_resolve_branch.return_value = {
                'system': 'Extract survey context',
                'user_template': 'Survey: {survey_content}',
                'schema': None
            }
            
            state = self.tester.fixtures.create_base_state()
            state['survey_raw_content'] = survey_content
            
            success, result, error = self.tester.test_single_node(
                "survey_context",
                survey_context_node,
                state,
                "설문의 전체적인 맥락과 목적을 추출"
            )
            
            if success:
                # 추가 검증
                if 'survey_context' not in result:
                    self.logger.log("Missing 'survey_context' in result", "ERROR")
                    success = False
                elif not result['survey_context'] or len(result['survey_context'].strip()) == 0:
                    self.logger.log("Empty survey context", "WARNING")
                else:
                    context = result['survey_context']
                    self.logger.log(f"Extracted context: '{context}' ({len(context)} chars)", "SUCCESS")
        
        return success, result
    
    def test_column_extractor_node(self, questions: Dict[str, Any], 
                                  data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Column Extractor 노드 테스트"""
        from nodes.stage1_data_preparation.column_extractor import get_open_column_node
        
        state = self.tester.fixtures.create_base_state()
        state['questions'] = questions
        state['data'] = data
        
        success, result, error = self.tester.test_single_node(
            "column_extractor",
            get_open_column_node,
            state,
            "개방형 질문에 해당하는 데이터 컬럼을 추출"
        )
        
        if success:
            # 추가 검증
            if 'open_columns' not in result:
                self.logger.log("Missing 'open_columns' in result", "ERROR")
                success = False
            elif not isinstance(result['open_columns'], list):
                self.logger.log("open_columns is not a list", "ERROR")
                success = False
            else:
                open_cols = result['open_columns']
                self.logger.log(f"Found {len(open_cols)} open columns: {open_cols}", "SUCCESS")
                
                # 개방형 질문 검증
                open_questions = [q_id for q_id, q_data in questions.items() 
                                if q_data.get('question_type') == 'OPEN_ENDED']
                self.logger.log(f"Expected open questions: {open_questions}", "DEBUG")
        
        return success, result
    
    def test_question_matcher_node(self, questions: Dict[str, Any], open_columns: List[str],
                                  data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Question Matcher 노드 테스트"""
        from nodes.stage1_data_preparation.question_matcher import question_data_matcher_node
        
        # Mock 설정
        mock_response = {
            "raw": "Questions matched successfully",
            "parsed": [
                {
                    "column_name": "Q2",
                    "question_number": "Q2",
                    "match_confidence": 0.95,
                    "reasoning": "Perfect match between question and column"
                },
                {
                    "column_name": "Q4",
                    "question_number": "Q4", 
                    "match_confidence": 0.90,
                    "reasoning": "Strong match for recommendation question"
                }
            ]
        }
        
        with patch('io_layer.llm.client.LLMClient') as mock_llm_client, \
             patch('config.prompt.prompt_loader.resolve_branch') as mock_resolve_branch:
            
            mock_llm_instance = Mock()
            mock_llm_instance.chat.return_value = (mock_response, Mock())
            mock_llm_client.return_value = mock_llm_instance
            
            mock_resolve_branch.return_value = {
                'system': 'Match questions to data columns',
                'user_template': 'Questions: {questions}\nColumns: {columns}',
                'schema': Mock()
            }
            
            state = self.tester.fixtures.create_base_state()
            state['questions'] = questions
            state['open_columns'] = open_columns
            state['data'] = data
            
            success, result, error = self.tester.test_single_node(
                "question_matcher",
                question_data_matcher_node,
                state,
                "질문과 데이터 컬럼 간의 매칭을 수행"
            )
            
            if success:
                # 추가 검증
                if 'question_data_match' not in result:
                    self.logger.log("Missing 'question_data_match' in result", "ERROR")
                    success = False
                else:
                    matches = result['question_data_match']
                    if isinstance(matches, list):
                        self.logger.log(f"Found {len(matches)} question-data matches", "SUCCESS")
                        for match in matches:
                            if isinstance(match, dict):
                                self.logger.log(f"Match: {match.get('question_number')} -> {match.get('column_name')} "
                                             f"(confidence: {match.get('match_confidence', 'N/A')})", "DEBUG")
                    else:
                        self.logger.log(f"Matches type: {type(matches)}", "DEBUG")
        
        return success, result
    
    def test_memory_optimizer_node(self, state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Memory Optimizer 노드 테스트"""
        from nodes.stage1_data_preparation.memory_optimizer import stage1_memory_flush_node
        
        success, result, error = self.tester.test_single_node(
            "memory_optimizer",
            stage1_memory_flush_node,
            state.copy(),
            "Stage 1 완료 후 메모리 최적화 수행"
        )
        
        if success:
            self.logger.log("Memory optimization completed", "SUCCESS")
        
        return success, result


class Stage2NodeTests:
    """Stage 2 노드들의 개별 테스트"""
    
    def __init__(self, tester: IndividualNodeTester):
        self.tester = tester
        self.logger = tester.logger
    
    def test_stage2_main_node(self, processing_queue: List[Dict], 
                             data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Stage2 Main 노드 테스트"""
        from nodes.stage2_data_preprocessing.stage2_main import stage2_data_preprocessing_node
        
        state = self.tester.fixtures.create_base_state()
        state['question_processing_queue'] = processing_queue
        state['current_question_idx'] = 0
        state['data'] = data
        
        success, result, error = self.tester.test_single_node(
            "stage2_main",
            stage2_data_preprocessing_node,
            state,
            "Stage 2 전처리의 메인 노드 - 현재 질문과 데이터 샘플 준비"
        )
        
        if success:
            # 추가 검증
            if 'current_question' not in result:
                self.logger.log("Missing 'current_question' in result", "ERROR")
                success = False
            elif 'current_data_sample' not in result:
                self.logger.log("Missing 'current_data_sample' in result", "ERROR") 
                success = False
            else:
                current_q = result['current_question']
                data_sample = result['current_data_sample']
                
                if current_q:
                    self.logger.log(f"Current question: {current_q.get('question_number')} "
                                  f"({current_q.get('question_type')})", "SUCCESS")
                else:
                    self.logger.log("No current question (end of queue)", "INFO")
                
                if isinstance(data_sample, list):
                    self.logger.log(f"Data sample size: {len(data_sample)}", "SUCCESS")
                    if len(data_sample) > 0:
                        self.logger.log(f"First sample: '{data_sample[0][:50]}...'", "DEBUG")
        
        return success, result
    
    def test_stage2_word_node(self, current_question: Dict[str, Any], 
                             data_sample: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """Stage2 Word 노드 테스트"""
        from nodes.stage2_data_preprocessing.stage2_word_node import stage2_word_node
        
        # Mock 설정
        mock_response = {
            "raw": "Word analysis completed",
            "parsed": {
                "keywords": ["품질", "만족", "좋음", "추천"],
                "sentiment_distribution": {"positive": 3, "neutral": 1, "negative": 1},
                "word_frequency": {"품질": 4, "만족": 3, "좋음": 2, "추천": 2}
            }
        }
        
        with patch('io_layer.llm.client.LLMClient') as mock_llm_client, \
             patch('config.prompt.prompt_loader.resolve_branch') as mock_resolve_branch:
            
            mock_llm_instance = Mock()
            mock_llm_instance.chat.return_value = (mock_response, Mock())
            mock_llm_client.return_value = mock_llm_instance
            
            mock_resolve_branch.return_value = {
                'system': 'You are a word analyzer',
                'user_template': 'Analyze words: {text_data}',
                'schema': Mock()
            }
            
            state = self.tester.fixtures.create_base_state()
            state['current_question'] = current_question
            state['current_data_sample'] = data_sample
            state['survey_context'] = '제품 만족도 조사'
            
            success, result, error = self.tester.test_single_node(
                "stage2_word_node",
                stage2_word_node,
                state,
                "단어 수준의 텍스트 분석 수행"
            )
            
            if success:
                # 추가 검증
                if 'stage2_word_results' not in result:
                    self.logger.log("Missing 'stage2_word_results' in result", "ERROR")
                    success = False
                else:
                    word_results = result['stage2_word_results']
                    self.logger.log(f"Word analysis results count: {len(word_results)}", "SUCCESS")
        
        return success, result
    
    def test_stage2_sentence_node(self, current_question: Dict[str, Any],
                                 data_sample: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """Stage2 Sentence 노드 테스트"""
        from nodes.stage2_data_preprocessing.stage2_sentence_node import stage2_sentence_node
        
        # Mock 설정 - 문법 체크와 문장 분석 응답
        mock_grammar_response = {
            "raw": "Grammar corrected",
            "parsed": {"corrected": "품질이 정말 좋습니다."}
        }
        
        mock_analysis_response = {
            "raw": "Sentence analyzed",
            "parsed": {
                "matching_question": True,
                "pos_neg": "POSITIVE",
                "automic_sentence": ["품질이 좋습니다", "추천하고 싶습니다"],
                "SVC_keywords": {
                    "sentence1": {"S": ["품질"], "V": ["좋다"], "C": []},
                    "sentence2": {"S": [""], "V": ["추천하다"], "C": []}
                }
            }
        }
        
        with patch('io_layer.llm.client.LLMClient') as mock_llm_client, \
             patch('config.prompt.prompt_loader.resolve_branch') as mock_resolve_branch:
            
            mock_llm_instance = Mock()
            # 각 데이터 샘플마다 2번 호출 (문법 체크 + 분석)
            responses = []
            for _ in data_sample:
                responses.extend([
                    (mock_grammar_response, Mock()),
                    (mock_analysis_response, Mock())
                ])
            mock_llm_instance.chat.side_effect = responses
            mock_llm_client.return_value = mock_llm_instance
            
            mock_resolve_branch.side_effect = [
                {
                    'system': 'Grammar checker',
                    'user_template': 'Check: {survey_context}\n{answer}',
                    'schema': Mock()
                },
                {
                    'system': 'Sentence analyzer', 
                    'user_template': 'Analyze: {survey_context}\n{question_summary}\n{corrected_answer}',
                    'schema': Mock()
                }
            ] * len(data_sample)
            
            state = self.tester.fixtures.create_base_state()
            state['current_question'] = current_question
            state['current_data_sample'] = data_sample
            state['survey_context'] = '제품 만족도 조사'
            
            success, result, error = self.tester.test_single_node(
                "stage2_sentence_node",
                stage2_sentence_node,
                state,
                "문장 수준의 텍스트 분석 수행 (문법 체크 + 의미 분석)"
            )
            
            if success:
                # 추가 검증
                if 'stage2_sentence_results' not in result:
                    self.logger.log("Missing 'stage2_sentence_results' in result", "ERROR")
                    success = False
                else:
                    sentence_results = result['stage2_sentence_results']
                    self.logger.log(f"Sentence analysis results count: {len(sentence_results)}", "SUCCESS")
                    
                    # LLM 호출 횟수 검증
                    expected_calls = len(data_sample) * 2  # 문법 체크 + 분석
                    actual_calls = mock_llm_instance.chat.call_count
                    self.logger.log(f"LLM calls: {actual_calls} (expected: {expected_calls})", "DEBUG")
        
        return success, result
    
    def test_stage2_etc_node(self, current_question: Dict[str, Any],
                           data_sample: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """Stage2 ETC 노드 테스트"""
        from nodes.stage2_data_preprocessing.stage2_etc_node import stage2_etc_node
        
        state = self.tester.fixtures.create_base_state()
        state['current_question'] = current_question
        state['current_data_sample'] = data_sample
        
        success, result, error = self.tester.test_single_node(
            "stage2_etc_node",
            stage2_etc_node,
            state,
            "기타 유형의 질문 처리"
        )
        
        if success:
            # 추가 검증
            if 'stage2_etc_results' not in result:
                self.logger.log("Missing 'stage2_etc_results' in result", "ERROR")
                success = False
            else:
                etc_results = result['stage2_etc_results']
                self.logger.log(f"ETC processing results count: {len(etc_results)}", "SUCCESS")
        
        return success, result


class SharedNodeTests:
    """공유 노드들의 개별 테스트"""
    
    def __init__(self, tester: IndividualNodeTester):
        self.tester = tester
        self.logger = tester.logger
    
    def test_survey_data_integrate_node(self, questions: Dict[str, Any],
                                      question_matches: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """Survey Data Integrate 노드 테스트"""
        from nodes.survey_data_integrate import survey_data_integrate_node
        
        state = self.tester.fixtures.create_base_state()
        state['questions'] = questions
        state['question_data_match'] = question_matches
        
        success, result, error = self.tester.test_single_node(
            "survey_data_integrate",
            survey_data_integrate_node,
            state,
            "설문 데이터와 질문 정보를 통합하여 처리 큐 생성"
        )
        
        if success:
            # 추가 검증
            if 'question_processing_queue' not in result:
                self.logger.log("Missing 'question_processing_queue' in result", "ERROR")
                success = False
            else:
                queue = result['question_processing_queue']
                self.logger.log(f"Created processing queue with {len(queue)} items", "SUCCESS")
                
                for i, item in enumerate(queue):
                    if isinstance(item, dict):
                        self.logger.log(f"Queue[{i}]: {item.get('question_number')} "
                                      f"({item.get('question_type')}) -> {item.get('column_name')}", "DEBUG")
        
        return success, result
    
    def test_stage2_next_question_node(self, processing_queue: List[Dict[str, Any]], 
                                     current_idx: int) -> Tuple[bool, Dict[str, Any]]:
        """Stage2 Next Question 노드 테스트"""
        from nodes.stage2_next_question import stage2_next_question_node
        
        state = self.tester.fixtures.create_base_state()
        state['question_processing_queue'] = processing_queue
        state['current_question_idx'] = current_idx
        
        success, result, error = self.tester.test_single_node(
            "stage2_next_question",
            stage2_next_question_node,
            state,
            "다음 질문으로 이동하는 반복자 노드"
        )
        
        if success:
            # 추가 검증
            new_idx = result.get('current_question_idx', current_idx)
            self.logger.log(f"Question index: {current_idx} -> {new_idx}", "SUCCESS")
            
            if new_idx >= len(processing_queue):
                self.logger.log("Reached end of processing queue", "INFO")
            else:
                self.logger.log(f"Ready for next question: {processing_queue[new_idx].get('question_number')}", "INFO")
        
        return success, result
    
    def test_state_flush_node(self, state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """State Flush 노드 테스트"""
        from nodes.state_flush_node import memory_status_check_node
        
        success, result, error = self.tester.test_single_node(
            "state_flush_node",
            memory_status_check_node,
            state.copy(),
            "메모리 상태 체크 및 플러시"
        )
        
        if success:
            if 'memory_status' in result:
                self.logger.log(f"Memory status: {result['memory_status']}", "SUCCESS")
        
        return success, result
    
    def test_stage_tracker_nodes(self, state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Stage Tracker 노드들 테스트"""
        from nodes.shared.stage_tracker import (
            stage1_data_preparation_completion,
            stage1_memory_flush_completion, 
            stage2_classification_start,
            update_stage_tracking,
            print_pipeline_status
        )
        
        test_state = state.copy()
        
        # Stage 1 completion test
        success1, result1, _ = self.tester.test_single_node(
            "stage1_data_preparation_completion",
            stage1_data_preparation_completion,
            test_state,
            "Stage 1 데이터 준비 완료 추적"
        )
        
        if success1:
            test_state = result1
        
        # Stage 1 memory flush completion test  
        success2, result2, _ = self.tester.test_single_node(
            "stage1_memory_flush_completion",
            stage1_memory_flush_completion,
            test_state,
            "Stage 1 메모리 플러시 완료 추적"
        )
        
        if success2:
            test_state = result2
        
        # Stage 2 start test
        success3, result3, _ = self.tester.test_single_node(
            "stage2_classification_start",
            stage2_classification_start,
            test_state,
            "Stage 2 분류 시작 추적"
        )
        
        overall_success = success1 and success2 and success3
        
        if overall_success:
            self.logger.log("All stage tracking nodes passed", "SUCCESS")
        
        return overall_success, result3 if success3 else test_state


class RouterTests:
    """라우터들의 개별 테스트"""
    
    def __init__(self, tester: IndividualNodeTester):
        self.tester = tester
        self.logger = tester.logger
    
    def test_stage2_type_router(self) -> bool:
        """Stage2 Type Router 테스트"""
        from router.stage2_router import stage2_type_router
        
        self.logger.log("Testing Stage2 Type Router", "NODE")
        self.logger.indent()
        
        test_cases = [
            ({'current_question': {'question_type': 'WORD', 'question_id': 'Q1'}}, 'WORD'),
            ({'current_question': {'question_type': 'SENTENCE', 'question_id': 'Q2'}}, 'SENTENCE'),
            ({'current_question': {'question_type': 'ETC', 'question_id': 'Q3'}}, 'ETC'),
            ({'current_question': None}, '__END__'),
            ({'current_question': {'question_type': 'UNKNOWN', 'question_id': 'Q4'}}, 'ETC'),
        ]
        
        success_count = 0
        for i, (input_state, expected) in enumerate(test_cases):
            try:
                self.logger.log(f"Test case {i+1}: {input_state['current_question']}", "TEST")
                result = stage2_type_router(input_state)
                
                if result == expected:
                    self.logger.log(f"✅ Expected: {expected}, Got: {result}", "SUCCESS")
                    success_count += 1
                else:
                    self.logger.log(f"❌ Expected: {expected}, Got: {result}", "ERROR")
                    
            except Exception as e:
                self.logger.log(f"❌ Test case {i+1} failed: {e}", "ERROR")
        
        overall_success = success_count == len(test_cases)
        self.logger.log(f"Router test: {success_count}/{len(test_cases)} passed", 
                       "SUCCESS" if overall_success else "ERROR")
        self.logger.dedent()
        
        return overall_success
    
    def test_stage2_completion_router(self) -> bool:
        """Stage2 Completion Router 테스트"""
        from nodes.stage2_next_question import stage2_completion_router
        
        self.logger.log("Testing Stage2 Completion Router", "NODE")
        self.logger.indent()
        
        test_cases = [
            ({'current_question_idx': 0, 'question_processing_queue': [{'q1': 'data'}, {'q2': 'data'}]}, 'CONTINUE'),
            ({'current_question_idx': 1, 'question_processing_queue': [{'q1': 'data'}, {'q2': 'data'}]}, 'CONTINUE'),
            ({'current_question_idx': 2, 'question_processing_queue': [{'q1': 'data'}, {'q2': 'data'}]}, '__END__'),
            ({'current_question_idx': 0, 'question_processing_queue': []}, '__END__'),
        ]
        
        success_count = 0
        for i, (input_state, expected) in enumerate(test_cases):
            try:
                self.logger.log(f"Test case {i+1}: idx={input_state['current_question_idx']}, "
                              f"queue_len={len(input_state['question_processing_queue'])}", "TEST")
                result = stage2_completion_router(input_state)
                
                if result == expected:
                    self.logger.log(f"✅ Expected: {expected}, Got: {result}", "SUCCESS")
                    success_count += 1
                else:
                    self.logger.log(f"❌ Expected: {expected}, Got: {result}", "ERROR")
                    
            except Exception as e:
                self.logger.log(f"❌ Test case {i+1} failed: {e}", "ERROR")
        
        overall_success = success_count == len(test_cases)
        self.logger.log(f"Completion router test: {success_count}/{len(test_cases)} passed",
                       "SUCCESS" if overall_success else "ERROR")
        self.logger.dedent()
        
        return overall_success


def main():
    """메인 테스트 함수"""
    tester = IndividualNodeTester()
    
    # 테스트 시작
    tester.logger.log("세부적인 개별 노드 테스트 시작", "HEADER")
    tester.logger.start_timer()
    
    try:
        # 테스트 환경 설정
        survey_file_path, data_file_path = tester.setup_test_environment()
        
        # 테스트 클래스들 초기화
        stage1_tests = Stage1NodeTests(tester)
        stage2_tests = Stage2NodeTests(tester)
        shared_tests = SharedNodeTests(tester)
        router_tests = RouterTests(tester)
        
        # 테스트 결과 수집
        results = {}
        
        # =============================================================================
        # STAGE 1 NODE TESTS
        # =============================================================================
        tester.logger.log("STAGE 1 노드들 개별 테스트", "HEADER")
        
        # 1. Survey Loader
        success, survey_state = stage1_tests.test_survey_loader_node(survey_file_path)
        results['survey_loader'] = success
        
        # 2. Data Loader
        success, data_state = stage1_tests.test_data_loader_node(data_file_path)
        results['data_loader'] = success
        
        # 3. Survey Parser (with mocking)
        survey_content = tester.fixtures.create_survey_content()
        success, parser_state = stage1_tests.test_survey_parser_node(survey_content)
        results['survey_parser'] = success
        
        # 4. Survey Context (with mocking)
        success, context_state = stage1_tests.test_survey_context_node(survey_content)
        results['survey_context'] = success
        
        # 5. Column Extractor
        questions = tester.fixtures.create_questions_dict()
        data = tester.fixtures.create_test_dataframe()
        success, column_state = stage1_tests.test_column_extractor_node(questions, data)
        results['column_extractor'] = success
        
        # 6. Question Matcher (with mocking)
        open_columns = ['Q2', 'Q4']  # Open-ended questions
        success, matcher_state = stage1_tests.test_question_matcher_node(questions, open_columns, data)
        results['question_matcher'] = success
        
        # 7. Memory Optimizer
        combined_state = tester.fixtures.create_base_state()
        combined_state.update(survey_state)
        combined_state.update(data_state)
        success, memory_state = stage1_tests.test_memory_optimizer_node(combined_state)
        results['memory_optimizer'] = success
        
        # =============================================================================
        # SHARED NODE TESTS
        # =============================================================================
        tester.logger.log("공유 노드들 개별 테스트", "HEADER")
        
        # 8. Survey Data Integration
        question_matches = [
            {"column_name": "Q2", "question_number": "Q2", "match_confidence": 0.95},
            {"column_name": "Q4", "question_number": "Q4", "match_confidence": 0.90}
        ]
        success, integrate_state = shared_tests.test_survey_data_integrate_node(questions, question_matches)
        results['survey_data_integrate'] = success
        
        # 9. Stage Tracker Nodes
        success, tracker_state = shared_tests.test_stage_tracker_nodes(combined_state)
        results['stage_tracker'] = success
        
        # 10. State Flush Node
        success, flush_state = shared_tests.test_state_flush_node(combined_state)
        results['state_flush'] = success
        
        # =============================================================================
        # STAGE 2 NODE TESTS
        # =============================================================================
        tester.logger.log("STAGE 2 노드들 개별 테스트", "HEADER")
        
        # Processing queue 준비
        processing_queue = [
            {
                'column_name': 'Q2',
                'question_number': 'Q2',
                'question_text': '제품 품질에 대한 의견을 자세히 말씀해주세요.',
                'question_type': 'SENTENCE',
                'question_id': 'Q2'
            },
            {
                'column_name': 'Q4', 
                'question_number': 'Q4',
                'question_text': '추천하고 싶은 정도를 말씀해주세요.',
                'question_type': 'WORD',
                'question_id': 'Q4'
            }
        ]
        
        # 11. Stage2 Main
        success, stage2_main_state = stage2_tests.test_stage2_main_node(processing_queue, data)
        results['stage2_main'] = success
        
        # 현재 질문과 데이터 샘플 준비
        if success and stage2_main_state.get('current_question'):
            current_question = stage2_main_state['current_question']
            data_sample = stage2_main_state.get('current_data_sample', [])
            
            # 12. Stage2 Word Node (with mocking)
            word_question = {
                'question_number': 'Q4',
                'question_text': '추천하고 싶은 정도를 말씀해주세요.',
                'question_type': 'WORD',
                'question_id': 'Q4'
            }
            success, word_state = stage2_tests.test_stage2_word_node(word_question, data_sample)
            results['stage2_word'] = success
            
            # 13. Stage2 Sentence Node (with mocking)
            sentence_question = {
                'question_number': 'Q2',
                'question_text': '제품 품질에 대한 의견을 자세히 말씀해주세요.',
                'question_type': 'SENTENCE',
                'question_id': 'Q2'
            }
            success, sentence_state = stage2_tests.test_stage2_sentence_node(sentence_question, data_sample)
            results['stage2_sentence'] = success
            
            # 14. Stage2 ETC Node
            etc_question = {
                'question_number': 'Q99',
                'question_text': '기타 질문입니다.',
                'question_type': 'ETC',
                'question_id': 'Q99'
            }
            success, etc_state = stage2_tests.test_stage2_etc_node(etc_question, data_sample)
            results['stage2_etc'] = success
            
        else:
            tester.logger.log("Stage2 main 실패로 인한 후속 테스트 스킵", "WARNING")
            results['stage2_word'] = False
            results['stage2_sentence'] = False
            results['stage2_etc'] = False
        
        # 15. Stage2 Next Question
        success, next_state = shared_tests.test_stage2_next_question_node(processing_queue, 0)
        results['stage2_next_question'] = success
        
        # =============================================================================
        # ROUTER TESTS
        # =============================================================================
        tester.logger.log("라우터들 개별 테스트", "HEADER")
        
        # 16. Stage2 Type Router
        success = router_tests.test_stage2_type_router()
        results['stage2_type_router'] = success
        
        # 17. Stage2 Completion Router
        success = router_tests.test_stage2_completion_router()
        results['stage2_completion_router'] = success
        
        # =============================================================================
        # 최종 결과 리포트
        # =============================================================================
        total_time = tester.logger.end_timer()
        
        tester.logger.log("전체 테스트 결과 요약", "HEADER")
        
        # 카테고리별 결과
        categories = {
            "Stage 1 Nodes": [
                'survey_loader', 'data_loader', 'survey_parser', 'survey_context',
                'column_extractor', 'question_matcher', 'memory_optimizer'
            ],
            "Stage 2 Nodes": [
                'stage2_main', 'stage2_word', 'stage2_sentence', 'stage2_etc', 'stage2_next_question'
            ],
            "Shared Nodes": [
                'survey_data_integrate', 'stage_tracker', 'state_flush'
            ],
            "Routers": [
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
        tester.logger.log("🎯 FINAL SUMMARY", "HEADER")
        tester.logger.log(f"Total Tests: {total_tests}", "INFO")
        tester.logger.log(f"Passed: {total_passed}", "SUCCESS")
        tester.logger.log(f"Failed: {total_tests - total_passed}", "ERROR")
        tester.logger.log(f"Success Rate: {(total_passed/total_tests)*100:.1f}%", "INFO")
        tester.logger.log(f"Total Runtime: {total_time:.2f} seconds", "TIMER")
        
        if total_passed == total_tests:
            tester.logger.log("🎉 ALL TESTS PASSED! 모든 노드가 정상 동작합니다!", "SUCCESS")
        else:
            tester.logger.log(f"⚠️ {total_tests - total_passed}개 노드에서 문제가 발견되었습니다.", "WARNING")
            tester.logger.log("실패한 노드들을 개별적으로 검토해주세요.", "INFO")
        
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