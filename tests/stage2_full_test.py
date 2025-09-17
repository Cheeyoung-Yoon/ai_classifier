"""
Stage2 Full Test with detailed debugging - Stage2만 테스트
"""
import sys
import os

# Add project root to path
project_root = "/home/cyyoon/test_area/ai_text_classification/2.langgraph"
sys.path.insert(0, project_root)

import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# Import the modules
from router.stage2_router import stage2_type_router
from nodes.stage2_data_preprocessing import (
    stage2_data_preprocessing_node,
    stage2_word_node,
    stage2_sentence_node
)

def print_debug_separator(title: str):
    """디버깅 섹션 구분자 출력"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_state_summary(state: Dict[str, Any], phase: str):
    """State 요약 정보 출력"""
    print(f"\n📊 State Summary - {phase}")
    print(f"   Current Question ID: {state.get('current_question_id', 'None')}")
    print(f"   Current Question Type: {state.get('current_question_type', 'None')}")
    print(f"   Project Name: {state.get('project_name', 'None')}")
    print(f"   Raw DataFrame Path: {state.get('raw_dataframe_path', 'None')}")
    print(f"   Matched Questions Count: {len(state.get('matched_questions', {}))}")
    if 'stage2_result_csv' in state:
        print(f"   Stage2 Result CSV: {state['stage2_result_csv']}")
    if 'stage2_processed_rows' in state:
        print(f"   Processed Rows: {state['stage2_processed_rows']}")
    if 'stage2_total_cost' in state:
        print(f"   Total Cost: ${state['stage2_total_cost']:.4f}")

def log_router_decision(state: Dict[str, Any]):
    """라우터 결정 과정 로깅"""
    question_id = state.get('current_question_id')
    question_type = state.get('current_question_type')
    matched_questions = state.get('matched_questions', {})
    
    if question_id in matched_questions:
        q_info = matched_questions[question_id]['question_info']
        actual_type = q_info.get('question_type')
        
        print(f"\n🎯 Router Decision Process:")
        print(f"   Question ID: {question_id}")
        print(f"   Question Type: {actual_type}")
        print(f"   Question Text: {q_info.get('question_text', 'N/A')}")
        
        router_decision = stage2_type_router(state)
        print(f"   Router Decision: {router_decision}")
        
        return router_decision
    return None


def create_mock_csv_data():
    """Create mock CSV data for testing"""
    # Create output directory
    os.makedirs("test_data", exist_ok=True)
    
    # Create mock survey data
    mock_data = {
        'respondent_id': range(1, 6),
        'depend_col': [1, 2, 1, 3, 2],  # For depend type questions
        'text_col1': [
            "이 제품은 정말 좋습니다",
            "별로 마음에 들지 않아요", 
            "가격대비 만족스러워요",
            "품질이 우수하다고 생각합니다",
            "다음에도 구매하고 싶어요"
        ],
        'text_col2': [
            "추천하고 싶습니다",
            "개선이 필요해요",
            "전체적으로 괜찮아요", 
            "매우 만족합니다",
            "계속 이용할 예정입니다"
        ]
    }
    
    df = pd.DataFrame(mock_data)
    csv_path = "test_data/mock_survey.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    return csv_path


def test_sentence_processing_with_data():
    """Test SENTENCE processing with actual data"""
    print_debug_separator("SENTENCE Processing Test")
    
    # Create mock data
    csv_path = create_mock_csv_data()
    print(f"📂 Mock CSV created: {csv_path}")
    
    # Mock state with realistic data structure
    sentence_state = {
        'current_question_id': 'test_pos_neg',
        'matched_questions': {
            'test_pos_neg': {
                'question_info': {
                    'question_type': 'pos_neg',
                    'question_text': '이 제품에 대한 전반적인 만족도는 어떠신가요?',
                    'question_summary': '제품 만족도 평가'
                },
                'matched_columns': ['text_col1', 'text_col2']
            }
        },
        'raw_dataframe_path': csv_path,
        'survey_context': '제품 만족도 조사'
    }
    
    print_state_summary(sentence_state, "Initial State")
    
    # 1. Router decision
    print_debug_separator("Router Decision Phase")
    router_decision = log_router_decision(sentence_state)
    
    # 2. Wrapper node
    print_debug_separator("Wrapper Node Phase") 
    print("🔄 Executing stage2_data_preprocessing_node...")
    wrapper_result = stage2_data_preprocessing_node(sentence_state)
    print(f"✅ Wrapper completed with status: {wrapper_result.get('stage2_status')}")
    print_state_summary(wrapper_result, "After Wrapper")
    
    # 3. SENTENCE node processing (if router decision is SENTENCE)
    if router_decision == "SENTENCE":
        print_debug_separator("SENTENCE Node Processing")
        print("🧠 Starting SENTENCE processing with two LLMs...")
        
        start_time = datetime.now()
        sentence_result = stage2_sentence_node(wrapper_result)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print_state_summary(sentence_result, "Final Result")
        
        if sentence_result.get('stage2_status') == 'completed':
            print(f"\n✅ SENTENCE processing SUCCESS!")
            print(f"📁 CSV saved to: {sentence_result.get('stage2_result_csv')}")
            print(f"📊 Processed rows: {sentence_result.get('stage2_processed_rows')}")
            print(f"💰 Total cost: ${sentence_result.get('stage2_total_cost', 0):.4f}")
            
            # CSV 파일 내용 미리보기
            if os.path.exists(sentence_result.get('stage2_result_csv', '')):
                print(f"\n📋 CSV File Preview:")
                df = pd.read_csv(sentence_result.get('stage2_result_csv'))
                print(f"   Columns: {list(df.columns)}")
                print(f"   Shape: {df.shape}")
        else:
            print(f"\n❌ SENTENCE processing FAILED:")
            print(f"   Error: {sentence_result.get('stage2_error')}")


def test_depend_processing_with_data():
    """Test depend_pos_neg processing with actual data"""
    print_debug_separator("DEPEND_POS_NEG Processing Test")
    
    csv_path = "test_data/mock_survey.csv"
    print(f"📂 Using existing CSV: {csv_path}")
    
    # Mock state for depend_pos_neg type
    depend_state = {
        'current_question_id': 'test_depend_pos_neg',
        'matched_questions': {
            'test_depend_pos_neg': {
                'question_info': {
                    'question_type': 'depend_pos_neg',
                    'question_text': '위에서 선택한 이유를 구체적으로 설명해주세요',
                    'question_summary': '선택 이유 설명'
                },
                'matched_columns': ['text_col1']  # depend_col will be auto-detected
            }
        },
        'raw_dataframe_path': csv_path,
        'survey_context': '제품 만족도 조사'
    }
    
    print_state_summary(depend_state, "Initial State")
    
    # 1. Router decision
    print_debug_separator("Router Decision Phase")
    router_decision = log_router_decision(depend_state)
    
    # 2. Wrapper node
    print_debug_separator("Wrapper Node Phase")
    print("🔄 Executing stage2_data_preprocessing_node...")
    wrapper_result = stage2_data_preprocessing_node(depend_state)
    print(f"✅ Wrapper completed with status: {wrapper_result.get('stage2_status')}")
    print_state_summary(wrapper_result, "After Wrapper")
    
    # 3. SENTENCE node processing (depend_pos_neg routes to SENTENCE)
    if router_decision == "SENTENCE":
        print_debug_separator("DEPEND_POS_NEG Node Processing")
        print("🧠 Starting DEPEND_POS_NEG processing with two LLMs...")
        
        start_time = datetime.now()
        sentence_result = stage2_sentence_node(wrapper_result)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print_state_summary(sentence_result, "Final Result")
        
        if sentence_result.get('stage2_status') == 'completed':
            print(f"\n✅ DEPEND_POS_NEG processing SUCCESS!")
            print(f"📁 CSV saved to: {sentence_result.get('stage2_result_csv')}")
            print(f"📊 Processed rows: {sentence_result.get('stage2_processed_rows')}")
            print(f"💰 Total cost: ${sentence_result.get('stage2_total_cost', 0):.4f}")
            
            # CSV 파일 내용 미리보기
            if os.path.exists(sentence_result.get('stage2_result_csv', '')):
                print(f"\n📋 CSV File Preview:")
                df = pd.read_csv(sentence_result.get('stage2_result_csv'))
                print(f"   Columns: {list(df.columns)}")
                print(f"   Shape: {df.shape}")
        else:
            print(f"\n❌ DEPEND_POS_NEG processing FAILED:")
            print(f"   Error: {sentence_result.get('stage2_error')}")


def test_word_processing():
    """Test WORD processing"""
    print_debug_separator("WORD Processing Test")
    
    # Use the same mock CSV file
    csv_path = "test_data/mock_survey.csv"
    print(f"📂 Using existing CSV: {csv_path}")
    
    word_state = {
        'current_question_id': 'test_concept',
        'matched_questions': {
            'test_concept': {
                'question_info': {
                    'question_type': 'concept',
                    'question_text': '가장 중요하게 생각하는 요소는 무엇인가요?'
                },
                'matched_columns': ['text_col1']  # Use actual column from mock data
            }
        },
        'raw_dataframe_path': csv_path,
        'survey_context': '개념 조사'
    }
    
    print_state_summary(word_state, "Initial State")
    
    # 1. Router decision
    print_debug_separator("Router Decision Phase")
    router_decision = log_router_decision(word_state)
    
    # 2. Wrapper node
    print_debug_separator("Wrapper Node Phase")
    print("🔄 Executing stage2_data_preprocessing_node...")
    wrapper_result = stage2_data_preprocessing_node(word_state)
    print(f"✅ Wrapper completed with status: {wrapper_result.get('stage2_status')}")
    print_state_summary(wrapper_result, "After Wrapper")
    
    # 3. WORD node processing
    if router_decision == "WORD":
        print_debug_separator("WORD Node Processing")
        print("📝 Starting WORD processing (no LLM, embedding only)...")
        
        start_time = datetime.now()
        word_result = stage2_word_node(wrapper_result)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print_state_summary(word_result, "Final Result")
        
        if word_result.get('stage2_status') == 'completed':
            print(f"\n✅ WORD processing SUCCESS!")
            print(f"📁 CSV saved to: {word_result.get('stage2_result_csv')}")
            print(f"📊 Processed rows: {word_result.get('stage2_processed_rows')}")
            
            # CSV 파일 내용 미리보기
            if os.path.exists(word_result.get('stage2_result_csv', '')):
                print(f"\n📋 CSV File Preview:")
                df = pd.read_csv(word_result.get('stage2_result_csv'))
                print(f"   Columns: {list(df.columns)}")
                print(f"   Shape: {df.shape}")
                
                # Embedding 컬럼 확인
                embed_columns = [col for col in df.columns if 'embed' in col]
                print(f"   Embedding Columns: {embed_columns}")
        else:
            print(f"\n❌ WORD processing FAILED:")
            print(f"   Error: {word_result.get('stage2_error')}")


if __name__ == "__main__":
    print_debug_separator("Stage2 Full Test Suite")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        test_word_processing()
        test_sentence_processing_with_data()
        test_depend_processing_with_data()
        
        print_debug_separator("All Tests Completed Successfully")
        print("🎉 모든 Stage2 테스트가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print_debug_separator("Test Failed")
        print(f"❌ 테스트 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()