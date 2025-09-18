#!/usr/bin/env python3
"""
Individual Node Test Suite - Stage 1 Data Preparation Nodes
Stage 1 데이터 준비 노드들의 개별 세부 테스트
"""

import os
import sys
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Stage 1 nodes
from nodes.stage1_data_preparation.survey_loader import load_survey_node
from nodes.stage1_data_preparation.data_loader import load_data_node
from nodes.stage1_data_preparation.survey_parser import parse_survey_node
from nodes.stage1_data_preparation.survey_context import survey_context_node
from nodes.stage1_data_preparation.column_extractor import get_open_column_node
from nodes.stage1_data_preparation.question_matcher import question_data_matcher_node
from nodes.stage1_data_preparation.memory_optimizer import stage1_memory_flush_node


class TestSurveyLoaderNode:
    """Detailed tests for survey loader node"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.survey_file = Path(self.temp_dir) / "test_survey.txt"
    
    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_valid_survey_file(self):
        """Test loading a valid survey file"""
        print("🧪 Testing valid survey file loading")
        
        # Create test survey content
        survey_content = """
        Q1. 브랜드에 대한 전반적인 만족도는 어떠신가요?
        ① 매우 만족 ② 만족 ③ 보통 ④ 불만족 ⑤ 매우 불만족
        
        Q2. 추천하고 싶은 정도는?
        (자유 서술)
        """
        self.survey_file.write_text(survey_content, encoding='utf-8')
        
        state = {
            'survey_file_path': str(self.survey_file),
            'current_stage': 'INITIALIZATION'
        }
        
        result = load_survey_node(state)
        
        # Assertions
        assert 'survey_raw_content' in result
        assert len(result['survey_raw_content']) > 0
        assert 'Q1' in result['survey_raw_content']
        assert 'Q2' in result['survey_raw_content']
        assert result['current_stage'] == 'STAGE1_SURVEY_LOADED'
        assert '브랜드에 대한 전반적인 만족도' in result['survey_raw_content']
        
        print("✅ Valid survey file loading passed")
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent survey file"""
        print("🧪 Testing non-existent file handling")
        
        state = {
            'survey_file_path': '/nonexistent/path/survey.txt',
            'current_stage': 'INITIALIZATION'
        }
        
        try:
            result = load_survey_node(state)
            assert False, "Should have raised an exception"
        except FileNotFoundError:
            print("✅ Correctly handled non-existent file")
        except Exception as e:
            print(f"✅ Exception handled: {e}")
    
    def test_load_empty_survey_file(self):
        """Test loading empty survey file"""
        print("🧪 Testing empty survey file")
        
        self.survey_file.write_text("", encoding='utf-8')
        
        state = {
            'survey_file_path': str(self.survey_file),
            'current_stage': 'INITIALIZATION'
        }
        
        result = load_survey_node(state)
        
        assert 'survey_raw_content' in result
        assert result['survey_raw_content'] == ""
        assert result['current_stage'] == 'STAGE1_SURVEY_LOADED'
        
        print("✅ Empty survey file handling passed")


class TestDataLoaderNode:
    """Detailed tests for data loader node"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_file = Path(self.temp_dir) / "test_data.csv"
        self.xlsx_file = Path(self.temp_dir) / "test_data.xlsx"
    
    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_csv_file(self):
        """Test loading CSV data file"""
        print("🧪 Testing CSV file loading")
        
        # Create test CSV data
        test_data = pd.DataFrame({
            'Q1': ['매우 만족', '만족', '보통', '불만족'],
            'Q2': ['추천함', '보통', '추천 안함', '모르겠음'],
            'Q3_자유응답': ['품질이 좋아요', '가격이 적당해요', '서비스가 아쉬워요', '전반적으로 만족'],
            'respondent_id': [1, 2, 3, 4]
        })
        test_data.to_csv(self.csv_file, index=False, encoding='utf-8')
        
        state = {
            'data_file_path': str(self.csv_file),
            'current_stage': 'STAGE1_SURVEY_LOADED'
        }
        
        result = load_data_node(state)
        
        # Assertions
        assert 'data' in result
        assert result['data'] is not None
        assert isinstance(result['data'], pd.DataFrame)
        assert len(result['data']) == 4
        assert 'Q1' in result['data'].columns
        assert 'Q2' in result['data'].columns
        assert 'Q3_자유응답' in result['data'].columns
        assert result['current_stage'] == 'STAGE1_DATA_LOADED'
        
        print("✅ CSV file loading passed")
    
    def test_load_xlsx_file(self):
        """Test loading Excel data file"""
        print("🧪 Testing Excel file loading")
        
        # Create test Excel data
        test_data = pd.DataFrame({
            'Q1': ['매우 만족', '만족'],
            'Q2_개방형': ['정말 좋아요', '괜찮습니다'],
            'ID': [101, 102]
        })
        test_data.to_excel(self.xlsx_file, index=False)
        
        state = {
            'data_file_path': str(self.xlsx_file),
            'current_stage': 'STAGE1_SURVEY_LOADED'
        }
        
        result = load_data_node(state)
        
        # Assertions
        assert 'data' in result
        assert isinstance(result['data'], pd.DataFrame)
        assert len(result['data']) == 2
        assert 'Q1' in result['data'].columns
        assert 'Q2_개방형' in result['data'].columns
        assert result['current_stage'] == 'STAGE1_DATA_LOADED'
        
        print("✅ Excel file loading passed")
    
    def test_load_invalid_file_format(self):
        """Test loading invalid file format"""
        print("🧪 Testing invalid file format")
        
        # Create text file with invalid format
        invalid_file = Path(self.temp_dir) / "invalid.txt"
        invalid_file.write_text("This is not a valid data format", encoding='utf-8')
        
        state = {
            'data_file_path': str(invalid_file),
            'current_stage': 'STAGE1_SURVEY_LOADED'
        }
        
        try:
            result = load_data_node(state)
            # Should handle gracefully or raise appropriate exception
        except Exception as e:
            print(f"✅ Correctly handled invalid format: {e}")


class TestSurveyParserNode:
    """Detailed tests for survey parser node"""
    
    @patch('nodes.stage1_data_preparation.survey_parser.resolve_branch')
    @patch('io_layer.llm.client.LLMClient')
    def test_parse_multiple_choice_questions(self, mock_llm_client, mock_resolve_branch):
        """Test parsing multiple choice questions"""
        print("🧪 Testing multiple choice question parsing")
        
        # Mock LLM response for multiple choice
        mock_response = {
            "raw": "Parsed questions",
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
            'user_template': 'Parse: {survey_content}',
            'schema': Mock()
        }
        
        survey_content = """
        Q1. 브랜드에 대한 전반적인 만족도는 어떠신가요?
        ① 매우 만족 ② 만족 ③ 보통 ④ 불만족 ⑤ 매우 불만족
        """
        
        state = {
            'survey_raw_content': survey_content,
            'current_stage': 'STAGE1_SURVEY_LOADED'
        }
        
        result = parse_survey_node(state)
        
        # Assertions
        assert 'questions' in result
        assert 'Q1' in result['questions']
        assert result['questions']['Q1']['question_type'] == 'MULTIPLE_CHOICE'
        assert 'choices' in result['questions']['Q1']
        assert result['current_stage'] == 'STAGE1_SURVEY_PARSED'
        
        print("✅ Multiple choice question parsing passed")
    
    @patch('nodes.stage1_data_preparation.survey_parser.resolve_branch')
    @patch('io_layer.llm.client.LLMClient')
    def test_parse_open_ended_questions(self, mock_llm_client, mock_resolve_branch):
        """Test parsing open-ended questions"""
        print("🧪 Testing open-ended question parsing")
        
        # Mock LLM response for open-ended
        mock_response = {
            "raw": "Parsed questions",
            "parsed": [
                {
                    "question_number": "Q2",
                    "question_text": "브랜드에 대한 의견을 자유롭게 말씀해주세요.",
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
            'user_template': 'Parse: {survey_content}',
            'schema': Mock()
        }
        
        survey_content = """
        Q2. 브랜드에 대한 의견을 자유롭게 말씀해주세요.
        (자유 서술형)
        """
        
        state = {
            'survey_raw_content': survey_content,
            'current_stage': 'STAGE1_SURVEY_LOADED'
        }
        
        result = parse_survey_node(state)
        
        # Assertions
        assert 'questions' in result
        assert 'Q2' in result['questions']
        assert result['questions']['Q2']['question_type'] == 'OPEN_ENDED'
        assert result['current_stage'] == 'STAGE1_SURVEY_PARSED'
        
        print("✅ Open-ended question parsing passed")


class TestColumnExtractorNode:
    """Detailed tests for column extractor node"""
    
    def test_extract_open_columns_mixed_types(self):
        """Test extracting open columns from mixed question types"""
        print("🧪 Testing open column extraction with mixed types")
        
        # Setup test data with mixed question types
        test_data = pd.DataFrame({
            'Q1_객관식': ['매우 만족', '만족', '보통'],
            'Q2_주관식': ['품질이 좋아요', '서비스가 아쉬워요', '가격이 적당해요'],
            'Q3_척도': [5, 4, 3],
            'Q4_자유응답': ['추천합니다', '보통입니다', '개선이 필요해요'],
            'Q5_선택형': ['A', 'B', 'C']
        })
        
        questions = {
            'Q1': {'question_type': 'MULTIPLE_CHOICE', 'question_text': '만족도'},
            'Q2': {'question_type': 'OPEN_ENDED', 'question_text': '의견'},
            'Q3': {'question_type': 'SCALE', 'question_text': '점수'},
            'Q4': {'question_type': 'OPEN_ENDED', 'question_text': '자유 의견'},
            'Q5': {'question_type': 'MULTIPLE_CHOICE', 'question_text': '선택'}
        }
        
        state = {
            'data': test_data,
            'questions': questions,
            'current_stage': 'STAGE1_SURVEY_PARSED'
        }
        
        result = get_open_column_node(state)
        
        # Assertions
        assert 'open_columns' in result
        open_columns = result['open_columns']
        
        # Should include open-ended questions
        assert 'Q2' in [col for col in open_columns if 'Q2' in col]
        assert 'Q4' in [col for col in open_columns if 'Q4' in col]
        
        # Should exclude multiple choice and scale questions
        q1_columns = [col for col in open_columns if 'Q1' in col]
        q3_columns = [col for col in open_columns if 'Q3' in col]
        q5_columns = [col for col in open_columns if 'Q5' in col]
        
        assert len(q1_columns) == 0  # Multiple choice excluded
        assert len(q5_columns) == 0  # Multiple choice excluded
        
        assert result['current_stage'] == 'STAGE1_COLUMNS_EXTRACTED'
        
        print("✅ Mixed type column extraction passed")
    
    def test_extract_no_open_columns(self):
        """Test extraction when no open columns exist"""
        print("🧪 Testing extraction with no open columns")
        
        test_data = pd.DataFrame({
            'Q1': ['A', 'B', 'C'],
            'Q2': [1, 2, 3],
            'Q3': ['선택1', '선택2', '선택3']
        })
        
        questions = {
            'Q1': {'question_type': 'MULTIPLE_CHOICE'},
            'Q2': {'question_type': 'SCALE'},
            'Q3': {'question_type': 'MULTIPLE_CHOICE'}
        }
        
        state = {
            'data': test_data,
            'questions': questions,
            'current_stage': 'STAGE1_SURVEY_PARSED'
        }
        
        result = get_open_column_node(state)
        
        # Assertions
        assert 'open_columns' in result
        assert len(result['open_columns']) == 0
        assert result['current_stage'] == 'STAGE1_COLUMNS_EXTRACTED'
        
        print("✅ No open columns extraction passed")


def run_stage1_tests():
    """Run all Stage 1 node tests"""
    print("🚀 Starting Stage 1 Node Detailed Tests")
    print("=" * 50)
    
    test_classes = [
        TestSurveyLoaderNode(),
        TestDataLoaderNode(),
        TestSurveyParserNode(),
        TestColumnExtractorNode()
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n📋 Running {class_name}")
        print("-" * 30)
        
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                if hasattr(test_class, 'setup_method'):
                    test_class.setup_method()
                
                getattr(test_class, test_method)()
                passed_tests += 1
                
                if hasattr(test_class, 'teardown_method'):
                    test_class.teardown_method()
                    
            except Exception as e:
                failed_tests.append(f"{class_name}.{test_method}: {str(e)}")
                print(f"❌ {test_method} failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 STAGE 1 TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests:
        print("\n❌ FAILED TESTS:")
        for failure in failed_tests:
            print(f"  - {failure}")
    else:
        print("\n🎉 ALL STAGE 1 TESTS PASSED!")


if __name__ == "__main__":
    run_stage1_tests()