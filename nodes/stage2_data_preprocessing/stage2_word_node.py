"""
Stage2 WORD Node - Handles WORD type questions
"""
import os
import pandas as pd
from datetime import datetime
import tqdm
from typing import Dict, Any, Optional
from io_layer.embedding import VectorEmbedding
from .prep_sentence import get_column_locations
from utils.project_manager import get_project_manager
from config.config import settings


def stage2_word_node(state: Dict[str, Any], deps: Optional[Any] = None) -> Dict[str, Any]:
    """
    WORD 타입 질문 처리 노드
    
    WORD 타입은 concept, img 등으로 ID와 컬럼 텍스트만 추출하여 CSV로 저장
    LLM 처리 없이 단순 데이터 추출만 수행
    
    Args:
        state: Current graph state
        deps: Dependencies
        
    Returns:
        Updated state with WORD processing results and CSV file path
    """
    embed = VectorEmbedding()
    
    current_question_id = state.get('current_question_id')
    current_question_type = state.get('current_question_type')
    
    print(f"stage2_word_node: Processing WORD type question {current_question_id} ({current_question_type})")
    
    try:
        # 데이터 추출 (LLM 처리 없음)
        result = get_column_locations(state, current_question_id, "WORD")
        if isinstance(result, tuple):
            text_df = result[1] if len(result) > 1 else result[0]
        else:
            text_df = result
            
        print(f"Found text data: {len(text_df)} rows")
        
        # 간단한 DataFrame 생성 (ID + 텍스트 컬럼들)
        result_data = []
        
        for i in range(len(text_df)):
            # 모든 컬럼의 텍스트 데이터 추출
            row_texts = text_df.iloc[i].dropna().astype(str).tolist()
            
            # 기본 행 데이터
            row_data = {
                'id': i,
                'question_id': current_question_id,
                'question_type': current_question_type
            }
            
            # 각 컬럼별로 텍스트 추가
            for col_idx, col_name in enumerate(text_df.columns):
                col_value = text_df.iloc[i, col_idx]
                if pd.notna(col_value):
                    row_data[f'text_{col_idx+1}'] = str(col_value).strip()
                    row_data[f'embed_text_{col_idx+1}'] = embed.embed(str(col_value).strip()) if str(col_value).strip() else []
                    
                else:
                    row_data[f'text_{col_idx+1}'] = ""
            
            # 모든 텍스트를 하나로 합친 컬럼도 추가
            # combined_text = " ".join([str(val).strip() for val in row_texts if pd.notna(val) and str(val).strip()])
            # row_data['combined_text'] = combined_text
            
            result_data.append(row_data)
        
        # DataFrame 생성
        result_df = pd.DataFrame(result_data)
        
        # CSV 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "output/stage2_results"
        # 프로젝트 매니저를 통해 CSV 파일 경로 생성
        project_name = state.get('project_name', 'unknown')
        project_manager = get_project_manager(project_name)
        
        csv_path = project_manager.get_stage2_csv_path(
            current_question_id, 
            current_question_type, 
            timestamp
        )
        
        result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"WORD processing completed for {current_question_id}")
        print(f"Results saved to: {csv_path}")
        print(f"Processed {len(result_df)} rows")
        
        # 기존 matched_questions에 데이터 경로 정보 추가
        matched_questions = state.get('matched_questions', {})
        if current_question_id in matched_questions:
            matched_questions[current_question_id]['stage2_data'] = {
                'csv_path': csv_path,
                'processing_type': 'WORD',
                'timestamp': timestamp,
                'status': 'completed'
            }
        
        # 결과 상태 반환
        updated_state = {
            **state,
            'matched_questions': matched_questions  # 업데이트된 matched_questions
        }
        
        # Stage 2 WORD 처리 후 state 저장
        project_name = state.get('project_name')
        if project_name and settings.SAVE_STATE_LOG:
            project_manager = get_project_manager(project_name)
            config = {'save_state_log': settings.SAVE_STATE_LOG}
            # current_stage 업데이트 후 저장
            updated_state['current_stage'] = f'STAGE2_WORD_{current_question_id}_COMPLETED'
            project_manager.save_state(dict(updated_state), config)
        
        return updated_state
        
    except Exception as e:
        print(f"Error in WORD processing for {current_question_id}: {e}")
        return {
            **state,
            'stage2_error': f"WORD processing failed: {str(e)}",
            'stage2_status': 'error',
            'stage2_processing_type': 'WORD'
        }