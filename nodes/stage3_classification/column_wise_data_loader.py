"""
Enhanced data loader for column-wise clustering
컬럼별 클러스터링을 위한 데이터 로더
"""

import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

try:
    from .config import Stage3Config
except ImportError:
    from config import Stage3Config


def load_data_by_columns_from_state(state: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    상태에서 컬럼별로 분리된 임베딩 데이터 로드
    
    Args:
        state: LangGraph state containing matched_questions
        
    Returns:
        Tuple of (embeddings_by_column, metadata)
        embeddings_by_column: {column_name: embeddings_array}
        metadata: 메타데이터 정보
    """
    print("🔍 Loading data by columns from state...")
    
    if "matched_questions" not in state:
        raise ValueError("No matched_questions found in state")
    
    matched_questions = state["matched_questions"]
    embeddings_by_column = {}
    all_dataframes = []
    
    # 각 질문별로 처리
    for question_id, question_data in matched_questions.items():
        if "stage2_data" not in question_data:
            print(f"⚠️  Skipping {question_id}: No stage2_data found")
            continue
        
        stage2_data = question_data["stage2_data"]
        
        if not isinstance(stage2_data, dict) or "csv_path" not in stage2_data:
            print(f"⚠️  Skipping {question_id}: Invalid stage2_data format")
            continue
        
        csv_path = stage2_data["csv_path"]
        
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded {question_id}: {len(df)} rows from {Path(csv_path).name}")
            
            # 임베딩 컬럼 추출
            column_embeddings = extract_embeddings_by_column(df, question_id)
            
            # 컬럼별 임베딩을 전체 딕셔너리에 추가
            for col_name, embeddings in column_embeddings.items():
                full_col_name = f"{question_id}_{col_name}"
                embeddings_by_column[full_col_name] = embeddings
                print(f"   📊 {full_col_name}: {len(embeddings)} embeddings")
            
            all_dataframes.append(df)
            
        except Exception as e:
            print(f"❌ Error loading {question_id} from {csv_path}: {e}")
            continue
    
    # 메타데이터 구성
    metadata = {
        'format': 'column_wise',
        'total_questions': len(matched_questions),
        'loaded_questions': len(all_dataframes),
        'column_info': {},
        'question_mapping': {}
    }
    
    # 컬럼 정보 메타데이터
    for col_name, embeddings in embeddings_by_column.items():
        question_id, col_type = col_name.split('_', 1)
        metadata['column_info'][col_name] = {
            'question_id': question_id,
            'column_type': col_type,
            'embedding_count': len(embeddings),
            'embedding_dim': embeddings.shape[1] if len(embeddings) > 0 else 0
        }
        
        if question_id not in metadata['question_mapping']:
            metadata['question_mapping'][question_id] = []
        metadata['question_mapping'][question_id].append(col_name)
    
    print(f"📊 Loaded {len(embeddings_by_column)} columns from {len(all_dataframes)} questions")
    return embeddings_by_column, metadata


def extract_embeddings_by_column(df: pd.DataFrame, question_id: str) -> Dict[str, np.ndarray]:
    """
    DataFrame에서 컬럼별로 임베딩 추출
    
    Args:
        df: 임베딩이 포함된 DataFrame
        question_id: 질문 ID
        
    Returns:
        컬럼별 임베딩 딕셔너리
    """
    embeddings_by_column = {}
    
    # 임베딩 컬럼 찾기
    embedding_columns = [col for col in df.columns if 'embed' in col.lower()]
    
    print(f"🔍 Found embedding columns in {question_id}: {embedding_columns}")
    
    for col in embedding_columns:
        try:
            # 임베딩 데이터 추출
            embeddings = []
            
            for idx, value in df[col].items():
                if pd.isna(value):
                    continue
                    
                try:
                    # 문자열로 저장된 임베딩을 배열로 변환
                    if isinstance(value, str):
                        embedding = ast.literal_eval(value)
                    else:
                        embedding = value
                    
                    # numpy 배열로 변환
                    embedding_array = np.array(embedding, dtype=np.float32)
                    
                    # 1차원 배열인지 확인
                    if embedding_array.ndim == 1 and len(embedding_array) > 0:
                        embeddings.append(embedding_array)
                        
                except Exception as e:
                    continue  # 변환 실패한 임베딩은 스킵
            
            if embeddings:
                # 모든 임베딩을 2D 배열로 변환
                embeddings_array = np.vstack(embeddings)
                
                # 컬럼명 정리 (embed_ 제거)
                clean_col_name = col.replace('embed_', '').replace('_embed', '')
                embeddings_by_column[clean_col_name] = embeddings_array
                
                print(f"   ✅ {clean_col_name}: {len(embeddings_array)} embeddings, dim={embeddings_array.shape[1]}")
            else:
                print(f"   ⚠️  No valid embeddings found in column: {col}")
                
        except Exception as e:
            print(f"   ❌ Error processing column {col}: {e}")
            continue
    
    return embeddings_by_column


def group_columns_by_type(embeddings_by_column: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    컬럼들을 타입별로 그룹화
    
    Args:
        embeddings_by_column: 컬럼별 임베딩 딕셔너리
        
    Returns:
        타입별로 그룹화된 임베딩 딕셔너리
    """
    grouped = {}
    
    for col_name, embeddings in embeddings_by_column.items():
        # 질문ID와 컬럼타입 분리
        parts = col_name.split('_', 1)
        if len(parts) == 2:
            question_id, col_type = parts
        else:
            question_id = parts[0]
            col_type = 'unknown'
        
        # 컬럼 타입별로 그룹화
        if col_type not in grouped:
            grouped[col_type] = {}
        
        grouped[col_type][col_name] = embeddings
    
    return grouped


def select_reference_column(embeddings_by_column: Dict[str, np.ndarray], 
                           strategy: str = 'largest') -> str:
    """
    기준 컬럼 선택
    
    Args:
        embeddings_by_column: 컬럼별 임베딩 딕셔너리
        strategy: 선택 전략 ('largest', 'first', 'most_diverse')
        
    Returns:
        선택된 기준 컬럼명
    """
    if not embeddings_by_column:
        raise ValueError("No columns available")
    
    if strategy == 'largest':
        # 가장 많은 샘플을 가진 컬럼 선택
        return max(embeddings_by_column.keys(), 
                  key=lambda k: len(embeddings_by_column[k]))
    
    elif strategy == 'first':
        # 첫 번째 컬럼 선택
        return list(embeddings_by_column.keys())[0]
    
    elif strategy == 'most_diverse':
        # 가장 다양성이 높은 컬럼 선택 (분산이 큰 컬럼)
        max_variance = -1
        best_column = None
        
        for col_name, embeddings in embeddings_by_column.items():
            if len(embeddings) > 10:  # 최소 샘플 수 확인
                variance = np.var(embeddings, axis=0).mean()
                if variance > max_variance:
                    max_variance = variance
                    best_column = col_name
        
        return best_column or list(embeddings_by_column.keys())[0]
    
    else:
        return list(embeddings_by_column.keys())[0]


if __name__ == "__main__":
    # Test the column-wise data loader
    import json
    
    print("🧪 Testing Column-wise Data Loader")
    print("=" * 50)
    
    # Load test state
    state_file = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/state_history/20250918_113231_081_STAGE2_WORD_문23_COMPLETED_state.json"
    
    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    # Test column-wise loading
    embeddings_by_column, metadata = load_data_by_columns_from_state(state)
    
    print(f"\n📊 Loaded Columns:")
    for col_name, embeddings in embeddings_by_column.items():
        print(f"   {col_name}: {embeddings.shape}")
    
    # Test column grouping
    grouped = group_columns_by_type(embeddings_by_column)
    print(f"\n📂 Grouped by type:")
    for col_type, columns in grouped.items():
        print(f"   {col_type}: {len(columns)} columns")
        for col_name in columns.keys():
            print(f"     - {col_name}")
    
    # Test reference column selection
    ref_col = select_reference_column(embeddings_by_column, 'largest')
    print(f"\n🎯 Selected reference column: {ref_col}")
    
    print("\n✅ Column-wise data loader test completed!")