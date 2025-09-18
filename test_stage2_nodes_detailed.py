#!/usr/bin/env python3
"""
Individual Node Test Suite - Stage 2 Data Preprocessing Nodes
Stage 2 데이터 전처리 노드들의 개별 세부 테스트
"""

import os
import sys
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Stage 2 nodes
from nodes.stage2_data_preprocessing.stage2_main import stage2_data_preprocessing_node
from nodes.stage2_data_preprocessing.stage2_word_node import stage2_word_node
from nodes.stage2_data_preprocessing.stage2_sentence_node import stage2_sentence_node
from nodes.stage2_data_preprocessing.stage2_etc_node import stage2_etc_node
from nodes.stage2_next_question import stage2_next_question_node, stage2_completion_router

# Import routers
from router.stage2_router import stage2_type_router


class TestStage2MainNode:
    """Detailed tests for stage2 main preprocessing node"""
    
    def test_main_node_with_valid_question_queue(self):
        """Test main node with valid question processing queue"""
        print("🧪 Testing stage2 main node with valid queue")
        
        # Setup test data
        test_data = pd.DataFrame({
            'Q2_자유응답': ['품질이 정말 좋아요', '서비스가 아쉬워요', '가격이 적당합니다', '전반적으로 만족해요'],
            'Q3_의견': ['추천하고 싶어요', '보통입니다', '개선이 필요해요', '매우 만족합니다'],
            'respondent_id': [1, 2, 3, 4]
        })
        
        question_queue = [
            {
                'column_name': 'Q2_자유응답',
                'question_number': 'Q2',
                'question_text': '제품 품질에 대한 의견을 말씀해주세요.',
                'question_type': 'SENTENCE'
            },
            {
                'column_name': 'Q3_의견',
                'question_number': 'Q3',
                'question_text': '전반적인 의견을 말씀해주세요.',
                 'question_type': 'WORD'
            }
        ]
        
        state = {
            'data': test_data,
            'question_processing_queue': question_queue,
            'current_question_idx': 0,
            'current_stage': 'STAGE2_CLASSIFICATION_START'
        }
        
        result = stage2_data_preprocessing_node(state)
        
        # Assertions
        assert 'current_question' in result
        assert 'current_data_sample' in result
        assert result['current_question']['column_name'] == 'Q2_자유응답'
        assert result['current_question']['question_type'] == 'SENTENCE'
        assert len(result['current_data_sample']) == 4  # All 4 responses
        assert '품질이 정말 좋아요' in result['current_data_sample']
        assert result['current_stage'] == 'STAGE2_PREPROCESSING'
        
        print("✅ Stage2 main node with valid queue passed")
    
    def test_main_node_with_empty_queue(self):
        """Test main node with empty question queue"""
        print("🧪 Testing stage2 main node with empty queue")
        
        state = {
            'data': pd.DataFrame(),
            'question_processing_queue': [],
            'current_question_idx': 0,
            'current_stage': 'STAGE2_CLASSIFICATION_START'
        }
        
        result = stage2_data_preprocessing_node(state)
        
        # Should handle gracefully
        assert 'current_question' in result
        assert result['current_question'] is None
        assert result['current_stage'] == 'STAGE2_PREPROCESSING'
        
        print("✅ Stage2 main node with empty queue passed")
    
    def test_main_node_with_out_of_bounds_index(self):
        """Test main node with out of bounds question index"""
        print("🧪 Testing stage2 main node with out of bounds index")
        
        question_queue = [
            {
                'column_name': 'Q1',
                'question_number': 'Q1', 
                'question_text': 'Test question',
                'question_type': 'SENTENCE'
            }
        ]
        
        state = {
            'data': pd.DataFrame({'Q1': ['test']}),
            'question_processing_queue': question_queue,
            'current_question_idx': 5,  # Out of bounds
            'current_stage': 'STAGE2_CLASSIFICATION_START'
        }
        
        result = stage2_data_preprocessing_node(state)
        
        # Should handle gracefully
        assert 'current_question' in result
        assert result['current_question'] is None
        assert result['current_stage'] == 'STAGE2_PREPROCESSING'
        
        print("✅ Stage2 main node with out of bounds index passed")


class TestStage2WordNode:
    """Detailed tests for stage2 word processing node"""
    
    @patch('nodes.stage2_data_preprocessing.stage2_word_node.resolve_branch')
    @patch('io_layer.llm.client.LLMClient')
    def test_word_node_processing(self, mock_llm_client, mock_resolve_branch):
        """Test word node processing functionality"""
        print("🧪 Testing stage2 word node processing")
        
        # Mock LLM response
        mock_response = {
            "raw": "Word analysis completed",
            "parsed": {
                "keywords": ["품질", "좋음", "만족", "추천"],
                "sentiment": "POSITIVE",
                "word_frequency": {
                    "품질": 3,
                    "좋음": 2,
                    "만족": 4,
                    "추천": 1
                }
            }
        }
        
        mock_llm_instance = Mock()
        mock_llm_instance.chat.return_value = (mock_response, Mock())
        mock_llm_client.return_value = mock_llm_instance
        
        mock_resolve_branch.return_value = {
            'system': 'You are a word analyzer',
            'user_template': 'Analyze words: {text_data}',
            'schema': Mock()
        }
        
        state = {
            'current_question': {
                'question_number': 'Q1',
                'question_text': '단어 분석 질문',
                'question_type': 'WORD'
            },
            'current_data_sample': [
                '품질이 좋아요',
                '만족합니다',
                '추천해요',
                '품질 좋음'
            ],
            'survey_context': '제품 만족도 조사',
            'current_stage': 'STAGE2_PREPROCESSING'
        }
        
        result = stage2_word_node(state)
        
        # Assertions
        assert 'stage2_word_results' in result
        assert result['current_stage'] == 'STAGE2_WORD_PROCESSED'
        
        # Check LLM was called with correct parameters
        mock_llm_instance.chat.assert_called_once()
        call_args = mock_llm_instance.chat.call_args
        assert 'system' in call_args.kwargs
        assert 'user' in call_args.kwargs
        
        print("✅ Stage2 word node processing passed")
    
    @patch('nodes.stage2_data_preprocessing.stage2_word_node.resolve_branch')
    @patch('io_layer.llm.client.LLMClient')
    def test_word_node_with_empty_data(self, mock_llm_client, mock_resolve_branch):
        """Test word node with empty data sample"""
        print("🧪 Testing stage2 word node with empty data")
        
        mock_llm_instance = Mock()
        mock_llm_instance.chat.return_value = ({"parsed": {}}, Mock())
        mock_llm_client.return_value = mock_llm_instance
        
        mock_resolve_branch.return_value = {
            'system': 'You are a word analyzer',
            'user_template': 'Analyze: {text_data}',
            'schema': Mock()
        }
        
        state = {
            'current_question': {
                'question_number': 'Q1',
                'question_type': 'WORD'
            },
            'current_data_sample': [],  # Empty data
            'survey_context': '조사',
            'current_stage': 'STAGE2_PREPROCESSING'
        }
        
        result = stage2_word_node(state)
        
        # Should handle gracefully
        assert 'stage2_word_results' in result
        assert result['current_stage'] == 'STAGE2_WORD_PROCESSED'
        
        print("✅ Stage2 word node with empty data passed")


class TestStage2SentenceNode:
    """Detailed tests for stage2 sentence processing node"""
    
    @patch('nodes.stage2_data_preprocessing.stage2_sentence_node.resolve_branch')
    @patch('io_layer.llm.client.LLMClient')
    def test_sentence_node_full_processing(self, mock_llm_client, mock_resolve_branch):
        """Test sentence node full processing pipeline"""
        print("🧪 Testing stage2 sentence node full processing")
        
        # Mock responses for grammar check and sentence analysis
        mock_grammar_response = {
            "raw": "Grammar checked",
            "parsed": {
                "corrected": "품질이 정말 좋습니다."
            }
        }
        
        mock_analysis_response = {
            "raw": "Analysis completed",
            "parsed": {
                "matching_question": True,
                "pos_neg": "POSITIVE",
                "automic_sentence": ["품질이 좋습니다", "추천하고 싶습니다"],
                "SVC_keywords": {
                    "sentence1": {
                        "S": ["품질"],
                        "V": ["좋다"],
                        "C": []
                    },
                    "sentence2": {
                        "S": [""],
                        "V": ["추천하다"],
                        "C": []
                    }
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
                'system': 'Grammar checker',
                'user_template': 'Check: {survey_context}\n{answer}',
                'schema': Mock()
            },
            {
                'system': 'Sentence analyzer',
                'user_template': 'Analyze: {survey_context}\n{question_summary}\n{corrected_answer}',
                'schema': Mock()
            }
        ]
        
        state = {
            'current_question': {
                'question_number': 'Q2',
                'question_text': '제품에 대한 의견을 말씀해주세요.',
                'question_type': 'SENTENCE'
            },
            'current_data_sample': [
                '품질이 정말 좋아요 추천하고 싶어요',
                '서비스가 아쉬워요',
                '전반적으로 만족합니다'
            ],
            'survey_context': '제품 만족도 조사',
            'current_stage': 'STAGE2_PREPROCESSING'
        }
        
        result = stage2_sentence_node(state)
        
        # Assertions
        assert 'stage2_sentence_results' in result
        assert result['current_stage'] == 'STAGE2_SENTENCE_PROCESSED'
        
        # Verify both LLM calls were made (grammar + analysis)
        assert mock_llm_instance.chat.call_count == len(state['current_data_sample']) * 2
        
        # Check result structure
        sentence_results = result['stage2_sentence_results']
        assert len(sentence_results) == len(state['current_data_sample'])
        
        print("✅ Stage2 sentence node full processing passed")
    
    @patch('nodes.stage2_data_preprocessing.stage2_sentence_node.resolve_branch')
    @patch('io_layer.llm.client.LLMClient')
    def test_sentence_node_grammar_only(self, mock_llm_client, mock_resolve_branch):
        """Test sentence node with grammar check only"""
        print("🧪 Testing stage2 sentence node grammar only")
        
        mock_grammar_response = {
            "raw": "Grammar checked",
            "parsed": {
                "corrected": "문법이 수정된 문장입니다."
            }
        }
        
        mock_llm_instance = Mock()
        mock_llm_instance.chat.return_value = (mock_grammar_response, Mock())
        mock_llm_client.return_value = mock_llm_instance
        
        mock_resolve_branch.return_value = {
            'system': 'Grammar checker',
            'user_template': 'Check: {survey_context}\n{answer}',
            'schema': Mock()
        }
        
        state = {
            'current_question': {
                'question_number': 'Q1',
                'question_text': '단순 문법 체크',
                'question_type': 'GRAMMAR_ONLY'  # Different type
            },
            'current_data_sample': ['문법이 틀린 문장 입니다'],
            'survey_context': '조사',
            'current_stage': 'STAGE2_PREPROCESSING'
        }
        
        result = stage2_sentence_node(state)
        
        # Should still process but maybe with different logic
        assert 'stage2_sentence_results' in result
        assert result['current_stage'] == 'STAGE2_SENTENCE_PROCESSED'
        
        print("✅ Stage2 sentence node grammar only passed")


class TestStage2EtcNode:
    """Detailed tests for stage2 etc processing node"""
    
    def test_etc_node_basic_processing(self):
        """Test etc node basic processing"""
        print("🧪 Testing stage2 etc node basic processing")
        
        state = {
            'current_question': {
                'question_number': 'Q99',
                'question_text': '기타 질문',
                'question_type': 'ETC'
            },
            'current_data_sample': [
                '기타 응답 1',
                '기타 응답 2',
                '특수한 응답'
            ],
            'current_stage': 'STAGE2_PREPROCESSING'
        }
        
        result = stage2_etc_node(state)
        
        # Assertions
        assert 'stage2_etc_results' in result
        assert result['current_stage'] == 'STAGE2_ETC_PROCESSED'
        
        # Check that it processed all data samples
        etc_results = result['stage2_etc_results']
        assert len(etc_results) == len(state['current_data_sample'])
        
        print("✅ Stage2 etc node basic processing passed")
    
    def test_etc_node_with_empty_data(self):
        """Test etc node with empty data"""
        print("🧪 Testing stage2 etc node with empty data")
        
        state = {
            'current_question': {
                'question_number': 'Q99',
                'question_type': 'ETC'
            },
            'current_data_sample': [],
            'current_stage': 'STAGE2_PREPROCESSING'
        }
        
        result = stage2_etc_node(state)
        
        # Should handle gracefully
        assert 'stage2_etc_results' in result
        assert result['current_stage'] == 'STAGE2_ETC_PROCESSED'
        assert len(result['stage2_etc_results']) == 0
        
        print("✅ Stage2 etc node with empty data passed")


class TestStage2NextQuestionNode:
    """Detailed tests for stage2 next question node"""
    
    def test_next_question_increment(self):
        """Test next question node increments index correctly"""
        print("🧪 Testing stage2 next question increment")
        
        state = {
            'question_processing_queue': [
                {'question_number': 'Q1'},
                {'question_number': 'Q2'},
                {'question_number': 'Q3'}
            ],
            'current_question_idx': 1,
            'current_stage': 'STAGE2_SENTENCE_PROCESSED'
        }
        
        result = stage2_next_question_node(state)
        
        # Assertions
        assert result['current_question_idx'] == 2  # Should increment
        assert result['current_stage'] == 'STAGE2_NEXT_QUESTION'
        
        print("✅ Stage2 next question increment passed")
    
    def test_next_question_at_end(self):
        """Test next question node at end of queue"""
        print("🧪 Testing stage2 next question at end")
        
        state = {
            'question_processing_queue': [
                {'question_number': 'Q1'},
                {'question_number': 'Q2'}
            ],
            'current_question_idx': 1,  # Last item
            'current_stage': 'STAGE2_WORD_PROCESSED'
        }
        
        result = stage2_next_question_node(state)
        
        # Should still increment (router will handle end condition)
        assert result['current_question_idx'] == 2
        assert result['current_stage'] == 'STAGE2_NEXT_QUESTION'
        
        print("✅ Stage2 next question at end passed")


class TestStage2Routers:
    """Detailed tests for stage2 routers"""
    
    def test_type_router_all_types(self):
        """Test type router with all possible question types"""
        print("🧪 Testing stage2 type router with all types")
        
        # Test WORD type
        state_word = {
            'current_question': {'question_type': 'WORD'}
        }
        assert stage2_type_router(state_word) == 'WORD'
        
        # Test SENTENCE type
        state_sentence = {
            'current_question': {'question_type': 'SENTENCE'}
        }
        assert stage2_type_router(state_sentence) == 'SENTENCE'
        
        # Test ETC type
        state_etc = {
            'current_question': {'question_type': 'ETC'}
        }
        assert stage2_type_router(state_etc) == 'ETC'
        
        # Test unknown type (should default to ETC)
        state_unknown = {
            'current_question': {'question_type': 'UNKNOWN_TYPE'}
        }
        result = stage2_type_router(state_unknown)
        assert result in ['ETC', '__END__']  # Depending on implementation
        
        # Test no current question
        state_none = {
            'current_question': None
        }
        assert stage2_type_router(state_none) == '__END__'
        
        print("✅ Stage2 type router all types passed")
    
    def test_completion_router_conditions(self):
        """Test completion router with various conditions"""
        print("🧪 Testing stage2 completion router conditions")
        
        # Test CONTINUE condition
        state_continue = {
            'current_question_idx': 1,
            'question_processing_queue': [
                {'q1': 'data'},
                {'q2': 'data'},
                {'q3': 'data'}
            ]
        }
        assert stage2_completion_router(state_continue) == 'CONTINUE'
        
        # Test END condition
        state_end = {
            'current_question_idx': 3,
            'question_processing_queue': [
                {'q1': 'data'},
                {'q2': 'data'},
                {'q3': 'data'}
            ]
        }
        assert stage2_completion_router(state_end) == '__END__'
        
        # Test empty queue
        state_empty = {
            'current_question_idx': 0,
            'question_processing_queue': []
        }
        assert stage2_completion_router(state_empty) == '__END__'
        
        print("✅ Stage2 completion router conditions passed")


def run_stage2_tests():
    """Run all Stage 2 node tests"""
    print("🚀 Starting Stage 2 Node Detailed Tests")
    print("=" * 50)
    
    test_classes = [
        TestStage2MainNode(),
        TestStage2WordNode(),
        TestStage2SentenceNode(),
        TestStage2EtcNode(),
        TestStage2NextQuestionNode(),
        TestStage2Routers()
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
    print("📊 STAGE 2 TEST SUMMARY")
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
        print("\n🎉 ALL STAGE 2 TESTS PASSED!")


if __name__ == "__main__":
    run_stage2_tests()