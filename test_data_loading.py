#!/usr/bin/env python3
"""실제 Stage2 데이터로 향상된 Stage3 파이프라인 직접 테스트"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('/home/cyyoon/test_area/ai_text_classification/2.langgraph')

def load_column_wise_data(directory_path):
    """질문별, 컬럼별로 데이터를 로드합니다."""
    data_groups = {}
    
    # 모든 CSV 파일 찾기
    if not os.path.exists(directory_path):
        print(f"⚠️ Directory not found: {directory_path}")
        return data_groups
    
    csv_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print(f"⚠️ No CSV files found in {directory_path}")
        return data_groups
    
    print(f"📁 Found {len(csv_files)} CSV files")
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            print(f"📄 Processing {os.path.basename(file_path)}: {len(df)} rows")
            print(f"   Columns: {list(df.columns)}")
            
            # 질문 ID 추출 (파일명 또는 question_id 컬럼에서)
            question_id = None
            if 'question_id' in df.columns and not df['question_id'].isna().all():
                question_id = df['question_id'].iloc[0]
            else:
                # 파일명에서 추출 (예: stage2_문4_img_20250917_154740.csv -> 문4)
                filename = os.path.basename(file_path)
                import re
                match = re.search(r'문(\d+)', filename)
                if match:
                    question_id = f"문{match.group(1)}"
            
            if not question_id:
                print(f"   ⚠️ Could not extract question_id from {file_path}")
                continue
            
            # 텍스트 컬럼 확인 - 정확한 패턴 매칭
            text_columns = []
            for col in df.columns:
                # text_1, text_2 등의 패턴만 매칭 (embed가 포함된 것은 제외)
                if col.startswith('text_') and 'embed' not in col:
                    text_columns.append(col)
            
            if not text_columns:
                print(f"   ⚠️ No text columns found in {file_path}")
                print(f"   Available columns: {list(df.columns)}")
                continue
            
            # 각 텍스트 컬럼에 대해 처리
            for text_col in text_columns:
                # 해당하는 임베딩 컬럼 찾기
                embed_col = f"embed_{text_col}"
                if embed_col not in df.columns:
                    print(f"   ⚠️ No embedding column {embed_col} found for {text_col}")
                    continue
                
                # 유효한 데이터만 필터링
                valid_mask = df[text_col].notna() & df[embed_col].notna()
                valid_df = df[valid_mask].copy()
                
                if len(valid_df) == 0:
                    print(f"   ⚠️ No valid data for {text_col}")
                    continue
                
                # 임베딩 문자열을 numpy 배열로 변환
                try:
                    embeddings = []
                    for embed_str in valid_df[embed_col]:
                        if isinstance(embed_str, str):
                            embed_array = np.array(eval(embed_str))
                        else:
                            embed_array = np.array(embed_str)
                        embeddings.append(embed_array)
                    
                    embeddings = np.array(embeddings)
                    
                    # 그룹 키 생성
                    group_key = f"{question_id}_{text_col}"
                    
                    data_groups[group_key] = {
                        'texts': valid_df[text_col].tolist(),
                        'embeddings': embeddings,
                        'question_id': question_id,
                        'column': text_col,
                        'file_path': file_path
                    }
                    
                    print(f"   ✅ Loaded {len(valid_df)} samples for {group_key}")
                    print(f"   📐 Embedding shape: {embeddings.shape}")
                    
                except Exception as e:
                    print(f"   ❌ Error processing embeddings for {text_col}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue
    
    print(f"\n📊 Total data groups loaded: {len(data_groups)}")
    return data_groups

def test_data_loading():
    """실제 Stage2 데이터 로딩 테스트"""
    
    print("🚀 실제 Stage2 데이터 로딩 테스트 시작")
    print("=" * 60)
    
    # 실제 Stage2 데이터 경로
    data_directory = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/temp_data/stage2_results copy/"
    
    try:
        # 데이터 로딩
        data_groups = load_column_wise_data(data_directory)
        
        print("\n" + "=" * 60)
        print("📊 로딩된 데이터 요약")
        print("=" * 60)
        
        if data_groups:
            for group_key, data in data_groups.items():
                print(f"\n🔍 {group_key}:")
                print(f"   • 텍스트 수: {len(data['texts'])}")
                print(f"   • 임베딩 형태: {data['embeddings'].shape}")
                print(f"   • 질문 ID: {data['question_id']}")
                print(f"   • 컬럼: {data['column']}")
                
                # 첫 번째 텍스트 샘플 표시
                if data['texts']:
                    sample_text = data['texts'][0]
                    if len(sample_text) > 50:
                        sample_text = sample_text[:50] + "..."
                    print(f"   • 샘플 텍스트: '{sample_text}'")
        else:
            print("❌ 로딩된 데이터가 없습니다.")
            
        print("\n✅ 데이터 로딩 테스트 완료!")
        return data_groups
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    test_data_loading()