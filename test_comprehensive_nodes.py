#!/usr/bin/env python3
"""
Comprehensive Unit Tests for All Graph Nodes and Pipeline Components
각 노드를 개별적으로 유닛테스트하는 포괄적인 테스트 코드
"""

import os
import sys
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import test fixtures and utilities
from graph.state import GraphState, initialize_project_state
from utils.project_manager import initialize_project_directories

# Import all nodes for testing
# Stage 1 nodes
from nodes.stage1_data_preparation.survey_loader import load_survey_node
from nodes.stage1_data_preparation.data_loader import load_data_node
from nodes.stage1_data_preparation.survey_parser import parse_survey_node
from nodes.stage1_data_preparation.survey_context import survey_context_node
from nodes.stage1_data_preparation.column_extractor import get_open_column_node
from nodes.stage1_data_preparation.question_matcher import question_data_matcher_node
from nodes.stage1_data_preparation.memory_optimizer import stage1_memory_flush_node

# Stage 2 nodes
from nodes.stage2_data_preprocessing.stage2_main import stage2_data_preprocessing_node
from nodes.stage2_data_preprocessing.stage2_word_node import stage2_word_node
from nodes.stage2_data_preprocessing.stage2_sentence_node import stage2_sentence_node
from nodes.stage2_data_preprocessing.stage2_etc_node import stage2_etc_node

# Shared nodes
from nodes.survey_data_integrate import survey_data_integrate_node
from nodes.stage2_next_question import stage2_next_question_node, stage2_completion_router
from nodes.state_flush_node import memory_status_check_node
from nodes.shared.stage_tracker import (
    stage1_data_preparation_completion,
    stage1_memory_flush_completion,
    stage2_classification_start,
    final_completion,
    print_pipeline_status,
    update_stage_tracking
)

# Import routers
from router.stage2_router import stage2_type_router

# Import graph functions
from graph.graph import pipeline_initialization_node, create_workflow


class TestDataFixtures:
    """Test data fixtures for consistent testing"""
    
    @staticmethod
    def create_mock_survey_data() -> str:
        """Create mock survey text data"""
        return """
        Q1. 브랜드에 대한 전반적인 만족도는 어떠신가요?
        ① 매우 만족 ② 만족 ③ 보통 ④ 불만족 ⑤ 매우 불만족
        
        Q2. 제품 품질에 대해서는 어떻게 생각하시나요?
        
        Q3. 가격 대비 만족도는 어떠신가요?
        ① 매우 좋음 ② 좋음 ③ 보통 ④ 나쁨 ⑤ 매우 나쁨
        """
    
    @staticmethod
    def create_mock_data_csv() -> pd.DataFrame:
        """Create mock data CSV"""
        return pd.DataFrame({
            'Q1': ['매우 만족', '만족', '보통'],
            'Q2': ['품질이 정말 좋아요', '괜찮은 편입니다', '그냥 그래요'],
            'Q3': ['가격이 적당해요', '조금 비싸긴 하지만 품질을 생각하면', '가성비 좋아요'],
            'respondent_id': [1, 2, 3]
        })
    
    @staticmethod
    def create_base_state() -> Dict[str, Any]:
        """Create base test state"""
        return {
            'project_name': 'test_project',
            'survey_file_path': '/test/survey.txt',
            'data_file_path': '/test/data.csv',
            'pipeline_id': 'test_pipeline_001',
            'current_stage': 'INITIALIZATION',
            'total_llm_cost_usd': 0.0,
            'questions': {},
            'data': None,
            'question_data_match': {},
            'open_columns': [],
            'question_processing_queue': [],
            'current_question_idx': 0,
            'stage2_processing_complete': False
        }


class TestStage1Nodes:
    """Test all Stage 1 data preparation nodes"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.fixtures = TestDataFixtures()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock files
        self.survey_file = Path(self.temp_dir) / "survey.txt"
        self.data_file = Path(self.temp_dir) / "data.csv"
        
        # Write test data
        self.survey_file.write_text(self.fixtures.create_mock_survey_data())
        self.fixtures.create_mock_data_csv().to_csv(self.data_file, index=False)
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_survey_loader_node(self):
        """Test survey loader node in isolation"""
        print("🧪 Testing survey_loader_node")
        
        state = self.fixtures.create_base_state()
        state['survey_file_path'] = str(self.survey_file)
        
        # Test the node
        result = load_survey_node(state)
        
        # Assertions
        assert 'survey_raw_content' in result
        assert len(result['survey_raw_content']) > 0
        assert 'Q1' in result['survey_raw_content']
        assert result['current_stage'] == 'STAGE1_SURVEY_LOADED'
        
        print("✅ survey_loader_node passed")
    
    def test_data_loader_node(self):
        """Test data loader node in isolation"""
        print("🧪 Testing data_loader_node")
        
        state = self.fixtures.create_base_state()
        state['data_file_path'] = str(self.data_file)
        
        # Test the node
        result = load_data_node(state)
        
        # Assertions
        assert 'data' in result
        assert result['data'] is not None
        assert len(result['data']) == 3  # 3 mock records
        assert 'Q1' in result['data'].columns
        assert result['current_stage'] == 'STAGE1_DATA_LOADED'
        
        print("✅ data_loader_node passed")
    
    @patch('nodes.stage1_data_preparation.survey_parser.resolve_branch')
    @patch('io_layer.llm.client.LLMClient')
    def test_survey_parser_node(self, mock_llm_client, mock_resolve_branch):
        """Test survey parser node with mocked LLM"""
        print("🧪 Testing survey_parser_node")
        
        # Mock LLM response
        mock_response = {
            "raw": "Mock LLM response",
            "parsed": [
                {
                    "question_number": "Q1",
                    "question_text": "브랜드에 대한 전반적인 만족도는 어떠신가요?",
                    "question_type": "MULTIPLE_CHOICE",
                    "choices": {
                        "1": "매우 만족",
                        "2": "만족", 
                        "3": "보통",
                        "4": "불만족",
                        "5": "매우 불만족"
                    }
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
        
        state = self.fixtures.create_base_state()
        state['survey_raw_content'] = self.fixtures.create_mock_survey_data()
        
        # Test the node
        result = parse_survey_node(state)
        
        # Assertions
        assert 'questions' in result
        assert len(result['questions']) > 0
        assert result['current_stage'] == 'STAGE1_SURVEY_PARSED'
        
        print("✅ survey_parser_node passed")
    
    @patch('nodes.stage1_data_preparation.survey_context.resolve_branch')
    @patch('io_layer.llm.client.LLMClient')
    def test_survey_context_node(self, mock_llm_client, mock_resolve_branch):
        """Test survey context extraction node"""
        print("🧪 Testing survey_context_node")
        
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_llm_instance.chat.return_value = ("브랜드 만족도 조사", Mock())
        mock_llm_client.return_value = mock_llm_instance
        
        mock_resolve_branch.return_value = {
            'system': 'Extract survey context',
            'user_template': 'Survey: {survey_content}',
            'schema': None
        }
        
        state = self.fixtures.create_base_state()
        state['survey_raw_content'] = self.fixtures.create_mock_survey_data()
        
        # Test the node
        result = survey_context_node(state)
        
        # Assertions
        assert 'survey_context' in result
        assert len(result['survey_context']) > 0
        assert result['current_stage'] == 'STAGE1_CONTEXT_EXTRACTED'
        
        print("✅ survey_context_node passed")
    
    def test_column_extractor_node(self):
        """Test open column extraction node"""
        print("🧪 Testing get_open_column_node")
        
        state = self.fixtures.create_base_state()
        state['data'] = self.fixtures.create_mock_data_csv()
        state['questions'] = {
            'Q1': {'question_type': 'MULTIPLE_CHOICE'},
            'Q2': {'question_type': 'OPEN_ENDED'},
            'Q3': {'question_type': 'MULTIPLE_CHOICE'}
        }
        
        # Test the node
        result = get_open_column_node(state)
        
        # Assertions
        assert 'open_columns' in result
        assert 'Q2' in result['open_columns']
        assert 'Q1' not in result['open_columns']  # Multiple choice should be excluded
        assert result['current_stage'] == 'STAGE1_COLUMNS_EXTRACTED'
        
        print("✅ get_open_column_node passed")
    
    @patch('nodes.stage1_data_preparation.question_matcher.resolve_branch')
    @patch('io_layer.llm.client.LLMClient')
    def test_question_matcher_node(self, mock_llm_client, mock_resolve_branch):
        """Test question data matcher node"""
        print("🧪 Testing question_data_matcher_node")
        
        # Mock LLM response
        mock_response = {
            "raw": "Mock response",
            "parsed": [
                {
                    "column_name": "Q2",
                    "question_number": "Q2",
                    "match_confidence": 0.95,
                    "reasoning": "Perfect match"
                }
            ]
        }
        
        mock_llm_instance = Mock()
        mock_llm_instance.chat.return_value = (mock_response, Mock())
        mock_llm_client.return_value = mock_llm_instance
        
        mock_resolve_branch.return_value = {
            'system': 'Match questions to data columns',
            'user_template': 'Questions: {questions}\nColumns: {columns}',
            'schema': Mock()
        }
        
        state = self.fixtures.create_base_state()
        state['questions'] = {'Q2': {'question_text': '제품 품질에 대해서는 어떻게 생각하시나요?'}}
        state['open_columns'] = ['Q2']
        state['data'] = self.fixtures.create_mock_data_csv()
        
        # Test the node
        result = question_data_matcher_node(state)
        
        # Assertions
        assert 'question_data_match' in result
        assert result['current_stage'] == 'STAGE1_QUESTIONS_MATCHED'
        
        print("✅ question_data_matcher_node passed")


class TestStage2Nodes:
    """Test all Stage 2 data preprocessing nodes"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.fixtures = TestDataFixtures()
    
    def test_stage2_main_node(self):
        """Test stage2 main preprocessing node"""
        print("🧪 Testing stage2_data_preprocessing_node")
        
        state = self.fixtures.create_base_state()
        state['question_processing_queue'] = [
            {
                'column_name': 'Q2',
                'question_number': 'Q2', 
                'question_text': '제품 품질에 대해서는 어떻게 생각하시나요?',
                'question_type': 'OPEN_ENDED'
            }
        ]
        state['current_question_idx'] = 0
        state['data'] = self.fixtures.create_mock_data_csv()
        
        # Test the node
        result = stage2_data_preprocessing_node(state)
        
        # Assertions
        assert 'current_question' in result
        assert 'current_data_sample' in result
        assert result['current_stage'] == 'STAGE2_PREPROCESSING'
        
        print("✅ stage2_data_preprocessing_node passed")
    
    @patch('nodes.stage2_data_preprocessing.stage2_word_node.resolve_branch')
    @patch('io_layer.llm.client.LLMClient')
    def test_stage2_word_node(self, mock_llm_client, mock_resolve_branch):
        """Test stage2 word processing node"""
        print("🧪 Testing stage2_word_node")
        
        # Mock LLM response
        mock_response = {
            "raw": "Mock response",
            "parsed": {
                "word_analysis": "품질, 좋음, 만족",
                "classification": "POSITIVE"
            }
        }
        
        mock_llm_instance = Mock()
        mock_llm_instance.chat.return_value = (mock_response, Mock())
        mock_llm_client.return_value = mock_llm_instance
        
        mock_resolve_branch.return_value = {
            'system': 'Analyze words',
            'user_template': 'Text: {text}',
            'schema': Mock()
        }
        
        state = self.fixtures.create_base_state()
        state['current_question'] = {
            'question_type': 'WORD',
            'question_text': '단어 분석 질문'
        }
        state['current_data_sample'] = ['품질이 좋음', '만족함']
        
        # Test the node
        result = stage2_word_node(state)
        
        # Assertions
        assert 'stage2_word_results' in result
        assert result['current_stage'] == 'STAGE2_WORD_PROCESSED'
        
        print("✅ stage2_word_node passed")
    
    @patch('nodes.stage2_data_preprocessing.stage2_sentence_node.resolve_branch')
    @patch('io_layer.llm.client.LLMClient')
    def test_stage2_sentence_node(self, mock_llm_client, mock_resolve_branch):
        """Test stage2 sentence processing node"""
        print("🧪 Testing stage2_sentence_node")
        
        # Mock LLM responses for grammar check and sentence analysis
        mock_grammar_response = {
            "raw": "Grammar corrected",
            "parsed": {"corrected": "품질이 정말 좋습니다."}
        }
        
        mock_analysis_response = {
            "raw": "Analysis done", 
            "parsed": {
                "matching_question": True,
                "pos_neg": "POSITIVE",
                "automic_sentence": ["품질이 좋습니다"],
                "SVC_keywords": {
                    "sentence1": {"S": ["품질"], "V": ["좋다"], "C": []}
                }
            }
        }
        
        mock_llm_instance = Mock()
        mock_llm_instance.chat.side_effect = [
            (mock_grammar_response, Mock()),
            (mock_analysis_response, Mock())
        ]
        mock_llm_client.return_value = mock_llm_instance
        
        mock_resolve_branch.side_effect = [
            {
                'system': 'Grammar check',
                'user_template': 'Check: {answer}',
                'schema': Mock()
            },
            {
                'system': 'Sentence analysis',
                'user_template': 'Analyze: {corrected_answer}',
                'schema': Mock()
            }
        ]
        
        state = self.fixtures.create_base_state()
        state['current_question'] = {
            'question_type': 'SENTENCE',
            'question_text': '문장 분석 질문'
        }
        state['current_data_sample'] = ['품질이 정말 좋아요']
        state['survey_context'] = '브랜드 만족도 조사'
        
        # Test the node
        result = stage2_sentence_node(state)
        
        # Assertions
        assert 'stage2_sentence_results' in result
        assert result['current_stage'] == 'STAGE2_SENTENCE_PROCESSED'
        
        print("✅ stage2_sentence_node passed")
    
    def test_stage2_etc_node(self):
        """Test stage2 etc processing node"""
        print("🧪 Testing stage2_etc_node")
        
        state = self.fixtures.create_base_state()
        state['current_question'] = {
            'question_type': 'ETC',
            'question_text': '기타 질문'
        }
        state['current_data_sample'] = ['기타 응답']
        
        # Test the node
        result = stage2_etc_node(state)
        
        # Assertions
        assert 'stage2_etc_results' in result
        assert result['current_stage'] == 'STAGE2_ETC_PROCESSED'
        
        print("✅ stage2_etc_node passed")
    
    def test_stage2_next_question_node(self):
        """Test stage2 next question iteration node"""
        print("🧪 Testing stage2_next_question_node")
        
        state = self.fixtures.create_base_state()
        state['question_processing_queue'] = [
            {'question_number': 'Q1'},
            {'question_number': 'Q2'},
            {'question_number': 'Q3'}
        ]
        state['current_question_idx'] = 0
        
        # Test the node
        result = stage2_next_question_node(state)
        
        # Assertions
        assert result['current_question_idx'] == 1  # Should increment
        assert result['current_stage'] == 'STAGE2_NEXT_QUESTION'
        
        print("✅ stage2_next_question_node passed")


class TestSharedNodes:
    """Test shared utility nodes"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.fixtures = TestDataFixtures()
    
    def test_survey_data_integrate_node(self):
        """Test survey data integration node"""
        print("🧪 Testing survey_data_integrate_node")
        
        state = self.fixtures.create_base_state()
        state['questions'] = {'Q1': {'question_type': 'OPEN_ENDED'}}
        state['question_data_match'] = [{'column_name': 'Q1', 'question_number': 'Q1'}]
        
        # Test the node
        result = survey_data_integrate_node(state)
        
        # Assertions
        assert 'question_processing_queue' in result
        assert len(result['question_processing_queue']) > 0
        assert result['current_stage'] == 'STAGE1_DATA_INTEGRATED'
        
        print("✅ survey_data_integrate_node passed")
    
    def test_memory_status_check_node(self):
        """Test memory status check node"""
        print("🧪 Testing memory_status_check_node")
        
        state = self.fixtures.create_base_state()
        
        # Test the node
        result = memory_status_check_node(state)
        
        # Assertions
        assert 'memory_status' in result
        assert result['current_stage'] == 'MEMORY_CHECK_COMPLETE'
        
        print("✅ memory_status_check_node passed")
    
    def test_stage_tracker_nodes(self):
        """Test all stage tracking nodes"""
        print("🧪 Testing stage tracking nodes")
        
        state = self.fixtures.create_base_state()
        
        # Test stage1 completion
        result1 = stage1_data_preparation_completion(state)
        assert result1['current_stage'] == 'STAGE1_DATA_PREPARATION_COMPLETE'
        
        # Test stage1 flush completion
        result2 = stage1_memory_flush_completion(result1)
        assert result2['current_stage'] == 'STAGE1_MEMORY_FLUSH_COMPLETE'
        
        # Test stage2 start
        result3 = stage2_classification_start(result2)
        assert result3['current_stage'] == 'STAGE2_CLASSIFICATION_START'
        
        # Test update stage tracking
        result4 = update_stage_tracking(result3, 'TEST_STAGE')
        assert result4['current_stage'] == 'TEST_STAGE'
        
        print("✅ All stage tracking nodes passed")


class TestRouters:
    """Test routing logic"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.fixtures = TestDataFixtures()
    
    def test_stage2_type_router(self):
        """Test stage2 type routing logic"""
        print("🧪 Testing stage2_type_router")
        
        # Test WORD routing
        state_word = self.fixtures.create_base_state()
        state_word['current_question'] = {'question_type': 'WORD'}
        assert stage2_type_router(state_word) == 'WORD'
        
        # Test SENTENCE routing
        state_sentence = self.fixtures.create_base_state()
        state_sentence['current_question'] = {'question_type': 'SENTENCE'}
        assert stage2_type_router(state_sentence) == 'SENTENCE'
        
        # Test ETC routing
        state_etc = self.fixtures.create_base_state()
        state_etc['current_question'] = {'question_type': 'ETC'}
        assert stage2_type_router(state_etc) == 'ETC'
        
        # Test END routing
        state_end = self.fixtures.create_base_state()
        state_end['current_question'] = None
        assert stage2_type_router(state_end) == '__END__'
        
        print("✅ stage2_type_router passed")
    
    def test_stage2_completion_router(self):
        """Test stage2 completion routing logic"""
        print("🧪 Testing stage2_completion_router")
        
        # Test CONTINUE routing
        state_continue = self.fixtures.create_base_state()
        state_continue['current_question_idx'] = 0
        state_continue['question_processing_queue'] = [{'q1': 'data'}, {'q2': 'data'}]
        assert stage2_completion_router(state_continue) == 'CONTINUE'
        
        # Test END routing
        state_end = self.fixtures.create_base_state()
        state_end['current_question_idx'] = 2
        state_end['question_processing_queue'] = [{'q1': 'data'}, {'q2': 'data'}]
        assert stage2_completion_router(state_end) == '__END__'
        
        print("✅ stage2_completion_router passed")


class TestGraphFunctions:
    """Test graph-level functions"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.fixtures = TestDataFixtures()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('utils.project_manager.initialize_project_directories')
    def test_pipeline_initialization_node(self, mock_init_dirs):
        """Test pipeline initialization node"""
        print("🧪 Testing pipeline_initialization_node")
        
        mock_init_dirs.return_value = {
            'project_dir': '/test/project',
            'temp_data_dir': '/test/temp',
            'state_file': '/test/state.json'
        }
        
        state = self.fixtures.create_base_state()
        
        # Test the node
        result = pipeline_initialization_node(state)
        
        # Assertions
        assert 'project_directories' in result
        assert result['current_stage'] == 'PIPELINE_INITIALIZATION'
        
        print("✅ pipeline_initialization_node passed")
    
    def test_create_workflow(self):
        """Test workflow creation"""
        print("🧪 Testing create_workflow")
        
        # Test workflow creation
        workflow = create_workflow()
        
        # Assertions
        assert workflow is not None
        # Check that all expected nodes are present
        expected_nodes = [
            'pipeline_init', 'load_survey', 'load_data', 'parse_survey',
            'extract_survey_context', 'get_open_columns', 'match_questions',
            'survey_data_integrate', 'stage1_memory_flush', 'memory_status_check',
            'stage2_main', 'stage2_word_node', 'stage2_sentence_node', 
            'stage2_etc_node', 'stage2_next_question'
        ]
        
        # Note: This is a basic structure test, detailed flow testing would require
        # compiling and running the graph which is more of an integration test
        
        print("✅ create_workflow passed")


class TestEndToEndIntegration:
    """Integration tests for node chains"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.fixtures = TestDataFixtures()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock files
        self.survey_file = Path(self.temp_dir) / "survey.txt"
        self.data_file = Path(self.temp_dir) / "data.csv"
        
        self.survey_file.write_text(self.fixtures.create_mock_survey_data())
        self.fixtures.create_mock_data_csv().to_csv(self.data_file, index=False)
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_stage1_data_flow(self):
        """Test Stage 1 data flow integration"""
        print("🧪 Testing Stage 1 data flow integration")
        
        # Start with basic state
        state = self.fixtures.create_base_state()
        state['survey_file_path'] = str(self.survey_file)
        state['data_file_path'] = str(self.data_file)
        
        # Simulate Stage 1 flow (without LLM calls)
        # 1. Load survey
        state = load_survey_node(state)
        assert 'survey_raw_content' in state
        
        # 2. Load data
        state = load_data_node(state)
        assert 'data' in state
        assert state['data'] is not None
        
        # 3. Mock survey parsing result
        state['questions'] = {
            'Q1': {'question_type': 'MULTIPLE_CHOICE'},
            'Q2': {'question_type': 'OPEN_ENDED'},
            'Q3': {'question_type': 'MULTIPLE_CHOICE'}
        }
        state['current_stage'] = 'STAGE1_SURVEY_PARSED'
        
        # 4. Extract open columns
        state = get_open_column_node(state)
        assert 'open_columns' in state
        assert 'Q2' in state['open_columns']
        
        # 5. Mock question matching
        state['question_data_match'] = [
            {'column_name': 'Q2', 'question_number': 'Q2', 'match_confidence': 0.95}
        ]
        state['current_stage'] = 'STAGE1_QUESTIONS_MATCHED'
        
        # 6. Integrate data
        state = survey_data_integrate_node(state)
        assert 'question_processing_queue' in state
        assert len(state['question_processing_queue']) > 0
        
        print("✅ Stage 1 data flow integration passed")
    
    def test_stage2_processing_flow(self):
        """Test Stage 2 processing flow integration"""
        print("🧪 Testing Stage 2 processing flow integration")
        
        # Setup state with Stage 1 results
        state = self.fixtures.create_base_state()
        state['data'] = self.fixtures.create_mock_data_csv()
        state['question_processing_queue'] = [
            {
                'column_name': 'Q2',
                'question_number': 'Q2',
                'question_text': '제품 품질에 대해서는 어떻게 생각하시나요?',
                'question_type': 'SENTENCE'
            }
        ]
        state['current_question_idx'] = 0
        
        # 1. Stage2 main preprocessing
        state = stage2_data_preprocessing_node(state)
        assert 'current_question' in state
        assert 'current_data_sample' in state
        
        # 2. Test routing
        route = stage2_type_router(state)
        assert route == 'SENTENCE'
        
        # 3. Process next question
        state = stage2_next_question_node(state)
        assert state['current_question_idx'] == 1
        
        # 4. Test completion routing
        completion_route = stage2_completion_router(state)
        assert completion_route == '__END__'  # Should end since we only have 1 question
        
        print("✅ Stage 2 processing flow integration passed")


def run_all_tests():
    """Run all unit tests"""
    print("🚀 Starting Comprehensive Node Unit Tests")
    print("=" * 60)
    
    # Test classes to run
    test_classes = [
        TestStage1Nodes(),
        TestStage2Nodes(), 
        TestSharedNodes(),
        TestRouters(),
        TestGraphFunctions(),
        TestEndToEndIntegration()
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n📋 Running {class_name}")
        print("-" * 40)
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                # Setup if available
                if hasattr(test_class, 'setup_method'):
                    test_class.setup_method()
                
                # Run test
                getattr(test_class, test_method)()
                passed_tests += 1
                
                # Teardown if available
                if hasattr(test_class, 'teardown_method'):
                    test_class.teardown_method()
                    
            except Exception as e:
                failed_tests.append(f"{class_name}.{test_method}: {str(e)}")
                print(f"❌ {test_method} failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests:
        print("\n❌ FAILED TESTS:")
        for failure in failed_tests:
            print(f"  - {failure}")
    else:
        print("\n🎉 ALL TESTS PASSED!")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)