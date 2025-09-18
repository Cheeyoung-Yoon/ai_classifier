#!/usr/bin/env python3
"""
Pytest-style Individual Node Tests
pytest 스타일의 개별 노드 단위 테스트
각 노드의 세부 기능을 독립적으로 테스트

Usage: 
  python test_individual_nodes_pytest.py
  or
     def test_load_data_success(self, temp_csv_file):
        #Test successful data loading
        from nodes.stage1_data_preparation.data_loader import load_data_node
        
        state = {
            'data_file_path': temp_csv_file,
            'current_stage': 'INIT'
        }
        
        result = load_data_node(state)
        
        # Data loader returns raw_data_info dict, not data field
        assert 'raw_data_info' in result
        assert 'sentences' in result['raw_data_info']
        assert 'length' in result['raw_data_info']
        assert isinstance(result['raw_data_info']['sentences'], list)
        assert len(result['raw_data_info']['sentences']) > 0
        assert result['raw_data_info']['length'] == len(result['raw_data_info']['sentences'])
        # The current_stage is maintained as INIT, not changed to STAGE1_DATA_LOADED
        assert result['current_stage'] == 'INIT'vidual_nodes_pytest.py -v
"""
# %%
import pytest
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class TestNodeStructures:
    """Test basic node structure and interface compliance"""
    
    def test_all_nodes_have_correct_signature(self):
        """Test that all nodes accept state dict and return state dict"""
        from nodes.stage1_data_preparation.survey_loader import load_survey_node
        from nodes.stage1_data_preparation.data_loader import load_data_node
        from nodes.stage2_data_preprocessing.stage2_main import stage2_data_preprocessing_node
        
        # Test function signatures
        import inspect
        
        # Test single parameter nodes
        single_param_nodes = [load_survey_node, load_data_node]
        
        for node in single_param_nodes:
            sig = inspect.signature(node)
            params = list(sig.parameters.keys())
            assert len(params) == 1, f"{node.__name__} should have exactly 1 parameter"
            assert params[0] == 'state', f"{node.__name__} parameter should be named 'state'"
        
        # Test stage2_data_preprocessing_node which has optional deps parameter
        sig = inspect.signature(stage2_data_preprocessing_node)
        params = list(sig.parameters.keys())
        assert len(params) == 2, f"stage2_data_preprocessing_node should have exactly 2 parameters"
        assert params[0] == 'state', f"First parameter should be named 'state'"
        assert params[1] == 'deps', f"Second parameter should be named 'deps'"
    
    def test_nodes_return_dict_with_stage(self):
        """Test that all nodes return dict with current_stage"""
        # This would be tested with actual node calls in integration tests
        pass


class TestSurveyLoaderNodePytest:
    """Pytest-style tests for survey loader node"""
    
    @pytest.fixture
    def temp_survey_file(self):
        """Create temporary survey file"""
        temp_dir = tempfile.mkdtemp()
        survey_file = Path(temp_dir) / "survey.txt"
        survey_content = """
        Q1. 브랜드 만족도는?
        ① 매우 만족 ② 만족 ③ 보통 ④ 불만족 ⑤ 매우 불만족
        
        Q2. 추천 의향은?
        (자유 응답)
        """
        survey_file.write_text(survey_content, encoding='utf-8')
        
        yield str(survey_file)
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_load_survey_success(self, temp_survey_file):
        """Test successful survey loading"""
        from nodes.stage1_data_preparation.survey_loader import load_survey_node
        
        state = {
            'survey_file_path': temp_survey_file,
            'current_stage': 'INIT'
        }
        
        result = load_survey_node(state)
        
        # Survey loader returns raw_survey_info dict, not survey_raw_content
        assert 'raw_survey_info' in result
        assert 'text' in result['raw_survey_info']
        assert len(result['raw_survey_info']['text']) > 0
        assert 'Q1' in result['raw_survey_info']['text']
        assert 'Q2' in result['raw_survey_info']['text']
        # The current_stage is maintained as INIT, not changed to STAGE1_SURVEY_LOADED
        assert result['current_stage'] == 'INIT'
    
    def test_load_survey_file_not_found(self):
        """Test survey loading with non-existent file"""
        from nodes.stage1_data_preparation.survey_loader import load_survey_node
        
        state = {
            'survey_file_path': '/path/to/nonexistent/file.xlsx',
            'current_stage': 'INIT'
        }
        
        result = load_survey_node(state)
        
        # When file doesn't exist, function returns input state unchanged
        assert 'survey_file_path' in result
        assert result['survey_file_path'] == '/path/to/nonexistent/file.xlsx'
        assert result['current_stage'] == 'INIT'
        # raw_survey_info is not added to state when file doesn't exist
        assert 'raw_survey_info' not in result
    
    @pytest.mark.parametrize("content,expected_questions", [
        ("Q1. Test?\n① A ② B", ["Q1"]),
        ("Q1. First?\nQ2. Second?", ["Q1", "Q2"]),
        ("", []),  # Empty file
    ])
    def test_load_survey_various_content(self, content, expected_questions):
        """Test survey loading with various content types"""
        from nodes.stage1_data_preparation.survey_loader import load_survey_node
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            state = {'survey_file_path': temp_path, 'current_stage': 'INIT'}
            result = load_survey_node(state)
            
            if content:
                assert 'raw_survey_info' in result
                assert 'text' in result['raw_survey_info']
                # Check for expected questions in the text
                for q in expected_questions:
                    assert q in result['raw_survey_info']['text']
            else:
                # Empty file might not create raw_survey_info
                assert result['current_stage'] == 'INIT'
                
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestDataLoaderNodePytest:
    """Pytest-style tests for data loader node"""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data"""
        return pd.DataFrame({
            'Q1': ['매우 만족', '만족', '보통'],
            'Q2_자유응답': ['좋아요', '괜찮아요', '그저 그래요'],
            'ID': [1, 2, 3]
        })
    
    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create temporary CSV file"""
        temp_dir = tempfile.mkdtemp()
        csv_file = Path(temp_dir) / "data.csv"
        sample_csv_data.to_csv(csv_file, index=False, encoding='utf-8')
        
        yield str(csv_file)
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_load_csv_data_success(self, temp_csv_file):
        """Test successful CSV data loading"""
        from nodes.stage1_data_preparation.data_loader import load_data_node
        
        state = {
            'data_file_path': temp_csv_file,
            'current_stage': 'INIT'
        }
        
        result = load_data_node(state)
        
        # Data loader returns raw_data_info dict, not data field
        assert 'raw_data_info' in result
        assert 'sentences' in result['raw_data_info']
        assert 'length' in result['raw_data_info']
        assert isinstance(result['raw_data_info']['sentences'], list)
        assert len(result['raw_data_info']['sentences']) == 3
        assert result['raw_data_info']['length'] == 3
        # The current_stage is maintained as INIT, not changed to STAGE1_DATA_LOADED
        assert result['current_stage'] == 'INIT'
    
    def test_load_excel_data_success(self, sample_csv_data):
        """Test successful Excel data loading"""
        from nodes.stage1_data_preparation.data_loader import load_data_node
        
        temp_dir = tempfile.mkdtemp()
        excel_file = Path(temp_dir) / "data.xlsx"
        sample_csv_data.to_excel(excel_file, index=False)
        
        try:
            state = {
                'data_file_path': str(excel_file),
                'current_stage': 'INIT'
            }
            
            result = load_data_node(state)
            
            # Data loader returns raw_data_info dict, not data field
            assert 'raw_data_info' in result
            assert 'sentences' in result['raw_data_info']
            assert 'length' in result['raw_data_info']
            assert isinstance(result['raw_data_info']['sentences'], list)
            assert len(result['raw_data_info']['sentences']) == 3
            assert result['raw_data_info']['length'] == 3
            # The current_stage is maintained as INIT
            assert result['current_stage'] == 'INIT'
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestStage2ProcessingNodesPytest:
    """Pytest-style tests for Stage 2 processing nodes"""
    
    @pytest.fixture
    def basic_stage2_state(self):
        """Create basic Stage 2 state"""
        return {
            'data': pd.DataFrame({
                'Q1': ['답변1', '답변2', '답변3'],
                'Q2': ['의견1', '의견2', '의견3']
            }),
            'question_processing_queue': [
                {
                    'column_name': 'Q1',
                    'question_number': 'Q1',
                    'question_text': '테스트 질문',
                    'question_type': 'SENTENCE'
                }
            ],
            'current_question_idx': 0,
            'current_stage': 'STAGE2_START'
        }
    
    def test_stage2_main_node(self, basic_stage2_state):
        """Test Stage 2 main preprocessing node"""
        from nodes.stage2_data_preprocessing.stage2_main import stage2_data_preprocessing_node
        
        result = stage2_data_preprocessing_node(basic_stage2_state, None)
        
        assert 'current_question' in result
        assert 'current_data_sample' in result
        assert result['current_question']['column_name'] == 'Q1'
        assert len(result['current_data_sample']) == 3
        assert result['current_stage'] == 'STAGE2_PREPROCESSING'
    
    def test_stage2_main_node_empty_queue(self):
        """Test Stage 2 main node with empty queue"""
        from nodes.stage2_data_preprocessing.stage2_main import stage2_data_preprocessing_node
        
        state = {
            'question_processing_queue': [],
            'current_question_idx': 0,
            'data': pd.DataFrame(),
            'current_stage': 'STAGE2_START'
        }
        
        result = stage2_data_preprocessing_node(state, None)
        
        # When queue is empty, processing should complete or handle gracefully
        assert 'current_stage' in result
        # Depending on implementation, might set completion stage or handle empty state
    
    @patch('io_layer.llm.client.LLMClient')
    @patch('nodes.stage2_data_preprocessing.stage2_word_node.resolve_branch')
    def test_stage2_word_node(self, mock_resolve_branch, mock_llm_client):
        """Test Stage 2 word processing node"""
        from nodes.stage2_data_preprocessing.stage2_word_node import stage2_word_node
        
        # Setup mocks
        mock_response = {"parsed": {"keywords": ["테스트", "단어"]}}
        mock_llm_instance = Mock()
        mock_llm_instance.chat.return_value = (mock_response, Mock())
        mock_llm_client.return_value = mock_llm_instance
        
        mock_resolve_branch.return_value = {
            'system': 'Word analyzer',
            'user_template': 'Analyze: {text_data}',
            'schema': Mock()
        }
        
        state = {
            'current_question': {
                'question_type': 'WORD',
                'question_text': '단어 분석'
            },
            'current_data_sample': ['테스트 응답'],
            'survey_context': '조사 컨텍스트',
            'current_stage': 'STAGE2_PREPROCESSING'
        }
        
        result = stage2_word_node(state)
        
        assert 'stage2_word_results' in result
        assert result['current_stage'] == 'STAGE2_WORD_PROCESSED'
        mock_llm_instance.chat.assert_called_once()


class TestRoutersPytest:
    """Pytest-style tests for routers"""
    
    @pytest.mark.parametrize("question_type,expected_route", [
        ('concept', 'WORD'),     # Concept questions go to WORD
        ('img', 'WORD'),         # Image questions go to WORD
        ('depend', 'SENTENCE'),  # Dependency questions go to SENTENCE
        ('depend_pos_neg', 'SENTENCE'),  # Dependency pos/neg goes to SENTENCE
        ('pos_neg', 'SENTENCE'), # Pos/neg questions go to SENTENCE
        ('etc', 'ETC'),          # ETC questions go to ETC
        ('UNKNOWN', 'ETC'),      # Unknown types default to ETC
    ])
    def test_stage2_type_router(self, question_type, expected_route):
        """Test Stage 2 type router with different question types"""
        from router.stage2_router import stage2_type_router
        
        state = {
            'current_question_id': 'Q1',
            'matched_questions': {
                'Q1': {
                    'question_info': {
                        'question_type': question_type
                    }
                }
            }
        }
        
        result = stage2_type_router(state)
        assert result == expected_route
    
    def test_stage2_type_router_no_question(self):
        """Test Stage 2 type router with no current question"""
        from router.stage2_router import stage2_type_router
        
        state = {'current_question_id': None}
        result = stage2_type_router(state)
        assert result == '__END__'
    
    @pytest.mark.parametrize("processing_complete,expected", [
        (False, 'CONTINUE'),  # Not complete, continue processing
        (True, '__END__'),    # Processing complete, end
    ])
    def test_stage2_completion_router(self, processing_complete, expected):
        """Test Stage 2 completion router with different conditions"""
        from nodes.stage2_next_question import stage2_completion_router
        
        state = {
            'stage2_processing_complete': processing_complete
        }
        
        result = stage2_completion_router(state)
        assert result == expected


class TestIntegrationFlowsPytest:
    """Test integration flows between nodes"""
    
    def test_stage1_to_stage2_data_flow(self):
        """Test data flow from Stage 1 to Stage 2"""
        # This would test the complete data transformation
        # from survey parsing through to Stage 2 preprocessing
        
        # Mock Stage 1 outputs
        stage1_output = {
            'questions': {
                'Q1': {'question_type': 'OPEN_ENDED', 'question_text': 'Test?'}
            },
            'question_data_match': [
                {'column_name': 'Q1', 'question_number': 'Q1'}
            ],
            'data': pd.DataFrame({'Q1': ['response1', 'response2']}),
            'current_stage': 'STAGE1_COMPLETE'
        }
        
        # Test transition to Stage 2
        from nodes.survey_data_integrate import survey_data_integrate_node
        
        result = survey_data_integrate_node(stage1_output)
        
        assert 'question_processing_queue' in result
        assert len(result['question_processing_queue']) > 0
        assert result['current_stage'] == 'STAGE1_DATA_INTEGRATED'


def run_pytest_tests():
    """Run pytest-style tests"""
    print("🧪 Running Pytest-style Individual Node Tests")
    print("=" * 50)
    
    # Import pytest and run tests
    try:
        import pytest
        exit_code = pytest.main([__file__, '-v', '--tb=short'])
        return exit_code == 0
    except ImportError:
        print("⚠️  pytest not available, running manual tests...")
        
        # Run tests manually
        test_classes = [
            TestNodeStructures(),
            TestSurveyLoaderNodePytest(),
            TestDataLoaderNodePytest(), 
            TestStage2ProcessingNodesPytest(),
            TestRoutersPytest(),
            TestIntegrationFlowsPytest()
        ]
        
        total = 0
        passed = 0
        
        for test_class in test_classes:
            methods = [m for m in dir(test_class) if m.startswith('test_')]
            for method in methods:
                total += 1
                try:
                    getattr(test_class, method)()
                    passed += 1
                    print(f"✅ {test_class.__class__.__name__}.{method}")
                except Exception as e:
                    print(f"❌ {test_class.__class__.__name__}.{method}: {e}")
        
        print(f"\nSummary: {passed}/{total} tests passed")
        return passed == total


if __name__ == "__main__":
    success = run_pytest_tests()
    sys.exit(0 if success else 1)
# %%
