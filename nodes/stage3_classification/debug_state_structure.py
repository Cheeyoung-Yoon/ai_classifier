"""
Debug script to understand the actual data structure
실제 데이터 구조를 파악하기 위한 디버그 스크립트
"""

import json
from pathlib import Path

def analyze_state_structure(state_file: str):
    """상태 구조 분석"""
    print("🔍 Analyzing state structure...")
    
    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    print(f"📊 Top-level keys: {list(state.keys())}")
    
    if "matched_questions" in state:
        matched_questions = state["matched_questions"]
        print(f"\n📝 Matched questions: {len(matched_questions)}")
        
        for question_id, question_data in matched_questions.items():
            print(f"\n🔹 {question_id}:")
            print(f"   Keys: {list(question_data.keys())}")
            
            if "stage2_data" in question_data:
                stage2_data = question_data["stage2_data"]
                print(f"   stage2_data type: {type(stage2_data)}")
                
                if isinstance(stage2_data, dict):
                    print(f"   stage2_data keys: {list(stage2_data.keys())}")
                    
                    # dataframe_path 확인
                    if "dataframe_path" in stage2_data:
                        df_path = stage2_data["dataframe_path"]
                        print(f"   dataframe_path: {df_path}")
                        
                        # 파일 존재 확인
                        if Path(df_path).exists():
                            print(f"   ✅ File exists: {Path(df_path).stat().st_size} bytes")
                            
                            # CSV 헤더 확인
                            try:
                                import pandas as pd
                                df = pd.read_csv(df_path, nrows=0)  # 헤더만 읽기
                                print(f"   📊 Columns: {list(df.columns)}")
                                
                                # 임베딩 컬럼 찾기
                                embedding_cols = [col for col in df.columns if 'embed' in col.lower()]
                                print(f"   🎯 Embedding columns: {embedding_cols}")
                                
                            except Exception as e:
                                print(f"   ❌ Error reading CSV: {e}")
                        else:
                            print(f"   ❌ File does not exist")
                    else:
                        print(f"   ⚠️  No dataframe_path found")
                        # 다른 가능한 키들 확인
                        for key, value in stage2_data.items():
                            if isinstance(value, str) and value.endswith('.csv'):
                                print(f"   📁 Possible CSV path in '{key}': {value}")
                else:
                    print(f"   stage2_data value: {stage2_data}")
            else:
                print(f"   ⚠️  No stage2_data found")
            
            # 최대 3개만 분석
            if len([q for q in matched_questions.keys() if q <= question_id]) >= 3:
                print(f"\n... (showing first 3 questions only)")
                break

if __name__ == "__main__":
    state_file = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/state_history/20250918_113231_081_STAGE2_WORD_문23_COMPLETED_state.json"
    analyze_state_structure(state_file)