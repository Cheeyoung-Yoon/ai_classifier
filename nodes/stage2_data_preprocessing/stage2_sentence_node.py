"""
Stage2 SENTENCE Node - Handles SENTENCE type questions
"""
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from io_layer.llm.client import LLMClient
from services.embedding import get_embedding_provider
from .prep_sentence import get_column_locations, extract_question_choices
from utils.project_manager import get_project_manager
from config.config import settings
from config.prompt.prompt_loader import load_prompt_config, resolve_branch
from utils.pydantic_utils import extract_llm_response_data

logger = logging.getLogger(__name__)


def stage2_sentence_node(state: Dict[str, Any], deps: Optional[Any] = None) -> Dict[str, Any]:
    """
    SENTENCE 타입 질문 처리 노드
    
    SENTENCE 타입은 depend, depend_pos_neg, pos_neg 등으로 텍스트 전처리가 필요
    두 개의 LLM을 사용: 1) 문법 교정 (gpt-4.1), 2) 문장 분석 (gpt-4.1-nano)
    
    Args:
        state: Current graph state
        deps: Dependencies (contains llm_client and shared embedding provider)
        
    Returns:
        Updated state with sentence processing results and CSV file path
    """
    current_question_id = state.get('current_question_id')
    current_question_type = state.get('current_question_type')
    
    logger.info(f"Processing SENTENCE type question {current_question_id} ({current_question_type})")
    
    # Use shared embedding provider instead of creating new instance
    embedding_provider = get_embedding_provider()
    
    try:
        # Config에서 prompt 로드
        grammar_branch = resolve_branch("sentence_grammar_check")
        
        # Question type에 따른 analysis branch 선택
        analysis_branch_key = f"sentence_{current_question_type}_split"
        analysis_branch = resolve_branch(analysis_branch_key)
        
        # Fallback: 기본 sentence_only branch 사용
        if not analysis_branch:
            analysis_branch = resolve_branch("sentence_only")
            
        if not grammar_branch or not analysis_branch:
            print(f"Warning: Missing prompt config for {current_question_type}")
            # Fallback to hardcoded prompts (기존 방식 유지)
            grammar_branch = None
            analysis_branch = None
        
        # 두 개의 LLM 클라이언트 설정 (stage2_prompt_work.py 패턴 따라)
        llm_client_large = LLMClient(model_key="gpt-4.1")     # 문법 교정용
        llm_client_nano = LLMClient(model_key="gpt-4.1-nano") # 문장 분석용
        
        # 컬럼 데이터 추출
        if current_question_type in ["depend", "depend_pos_neg"]:
            depend_df, text_df = get_column_locations(state, current_question_id, "SENTENCE")
            print(f"Found depend data: {len(depend_df)} rows, text data: {len(text_df)} rows")
        else:
            result = get_column_locations(state, current_question_id, "SENTENCE")
            if isinstance(result, tuple):
                text_df = result[1] if len(result) > 1 else result[0]
            else:
                text_df = result
            depend_df = None
            print(f"Found text data: {len(text_df)} rows")
        
        # 질문 정보 추출
        matched_questions = state.get('matched_questions', {})
        
        # matched_questions가 list인 경우 dict로 변환
        if isinstance(matched_questions, list):
            matched_questions_dict = {}
            for item in matched_questions:
                if isinstance(item, dict) and 'question_id' in item:
                    matched_questions_dict[item['question_id']] = {'question_info': item}
            matched_questions = matched_questions_dict
        
        question_info = matched_questions[current_question_id]['question_info']
        survey_context = state.get('survey_context', '')
        question_summary = question_info.get('question_summary', question_info.get('question_text', ''))
        
        # 응답 사전 구성 (depend 타입인 경우)
        response_dict = {}
        if current_question_type in ["depend", "depend_pos_neg"] and depend_df is not None:
            unique_depends = depend_df.iloc[:, 0].dropna().unique().tolist()
            unique_depends = [str(int(x)) for x in unique_depends if pd.notna(x)]
            
            try:
                response_mapping = extract_question_choices(llm_client_nano, question_summary, unique_depends)
                
                # response_mapping이 dict인지 확인하고 answers 키가 있는지 체크
                if isinstance(response_mapping, dict) and 'answers' in response_mapping:
                    response_dict = response_mapping['answers']
                    if not isinstance(response_dict, dict):
                        print(f"Warning: response_mapping['answers'] is not a dict, type: {type(response_dict)}")
                        response_dict = {}
                else:
                    print(f"Warning: response_mapping is not valid, type: {type(response_mapping)}")
                    response_dict = {}
            except Exception as e:
                print(f"Error extracting question choices: {e}")
                response_dict = {}
        
        # 두 단계 LLM 처리 (멀티스레드 지원)
        fallback_grammar_system = (
            """Correct grammar and typos in Korean survey responses.\n\n"
            "- Correct ONLY the answer, making it natural Korean.\n"
            "- Use the question only as context if needed.\n\n"
            "Return result as valid JSON in the format:\n"
            '{ "corrected": "<corrected answer>" }\n\n'
            "Rules:\n"
            "- Do not change the meaning.\n"
            "- Correct typos and unnatural expressions into the most natural and common form in Korean survey responses.\n"
            "- When multiple corrections are possible, prefer the one that best fits everyday consumer feedback context.\n"
            "- Do not output anything except the JSON."""
        )

        grammar_schema = grammar_branch.get('schema') if grammar_branch else None
        grammar_system = grammar_branch['system'] if grammar_branch else None
        grammar_template = grammar_branch['user_template'] if grammar_branch else None

        analysis_schema = analysis_branch.get('schema') if analysis_branch else None
        analysis_system = analysis_branch['system'] if analysis_branch else None
        analysis_template = analysis_branch['user_template'] if analysis_branch else None

        process_items = []
        for idx in range(len(text_df)):
            row_values = text_df.iloc[idx].dropna().astype(str).tolist()
            if not row_values:
                continue
            original_text = row_values[0].strip()
            if not original_text:
                continue

            depend_value = None
            if current_question_type in ["depend", "depend_pos_neg"] and depend_df is not None and idx < len(depend_df):
                depend_raw = depend_df.iloc[idx, 0]
                if pd.notna(depend_raw):
                    try:
                        depend_value = str(int(depend_raw))
                    except (ValueError, TypeError):
                        depend_value = str(depend_raw)

            process_items.append({
                'index': idx,
                'original_text': original_text,
                'depend_value': depend_value
            })

        total_targets = len(process_items)
        if total_targets == 0:
            print("No valid responses found for SENTENCE processing.")
            result_data = []
            total_price = 0.0
        else:
            max_workers = min(8, total_targets)
            results_map: Dict[int, Dict[str, Any]] = {}
            total_price = 0.0
            processed_count = 0

            def process_row(item: Dict[str, Any]):
                idx = item['index']
                original_text = item['original_text']
                depend_value = item.get('depend_value')
                row_cost = 0.0

                if grammar_template:
                    grammar_prompt = grammar_template.format(
                        survey_context=survey_context,
                        answer=original_text
                    )
                    grammar_system_local = grammar_system
                else:
                    grammar_prompt = (
                        f"summary of the survey: {survey_context}\n"
                        f"answer: {original_text}\n"
                    )
                    grammar_system_local = fallback_grammar_system

                grammar_resp, grammar_log = llm_client_large.chat(
                    system=grammar_system_local,
                    user=grammar_prompt,
                    schema=grammar_schema
                )
                row_cost += getattr(grammar_log, 'cost_usd', 0.0)

                if grammar_schema:
                    grammar_data = extract_llm_response_data(grammar_resp)
                    corrected_text = grammar_data.get('corrected', original_text)
                else:
                    try:
                        corrected_data = json.loads(grammar_resp)
                        corrected_text = corrected_data.get('corrected', original_text)
                    except (json.JSONDecodeError, TypeError):
                        corrected_text = original_text

                if analysis_template:
                    if current_question_type in ["depend", "depend_pos_neg"] and depend_value is not None:
                        sub_explanation = ""
                        if isinstance(response_dict, dict):
                            sub_explanation = response_dict.get(depend_value, "")
                        else:
                            print(f"Warning: response_dict is not a dictionary, type: {type(response_dict)}")
                        analysis_prompt = analysis_template.format(
                            survey_context=survey_context,
                            question_summary=question_summary,
                            sub_explanation=sub_explanation,
                            corrected_answer=corrected_text
                        )
                    else:
                        analysis_prompt = analysis_template.format(
                            survey_context=survey_context,
                            question_summary=question_summary,
                            corrected_answer=corrected_text
                        )
                    analysis_system_local = analysis_system
                else:
                    if current_question_type in ["depend", "depend_pos_neg"] and depend_value is not None:
                        sub_explanation = ""
                        if isinstance(response_dict, dict):
                            sub_explanation = response_dict.get(depend_value, "")
                        else:
                            print(f"Warning: response_dict is not a dictionary, type: {type(response_dict)}")
                        analysis_prompt = (
                            f"question: {question_summary}\n"
                            f"sub_explanation: {sub_explanation}\n"
                            f"answer: {corrected_text}\n"
                        )
                    else:
                        analysis_prompt = (
                            f"question: {question_summary}\n"
                            f"answer: {corrected_text}\n"
                        )
                    analysis_system_local = None

                analysis_resp, analysis_log = llm_client_nano.chat(
                    system=analysis_system_local,
                    user=analysis_prompt,
                    schema=analysis_schema
                )
                row_cost += getattr(analysis_log, 'cost_usd', 0.0)

                if analysis_schema:
                    data = extract_llm_response_data(analysis_resp)
                else:
                    try:
                        data = json.loads(analysis_resp)
                    except json.JSONDecodeError:
                        try:
                            cleaned = analysis_resp.replace('```json', '').replace('```', '')
                            data = json.loads(cleaned)
                        except Exception:
                            data = None

                if isinstance(data, list):
                    data = data[0] if data and isinstance(data[0], dict) else None

                if not isinstance(data, dict):
                    data = {
                        'matching_question': False,
                        'pos_neg': 'NEUTRAL',
                        'automic_sentence': [],
                        'SVC_keywords': {}
                    }

                data.setdefault('matching_question', False)
                data.setdefault('pos_neg', data.get('pos.neg', 'NEUTRAL'))
                data.setdefault('automic_sentence', [])
                data.setdefault('SVC_keywords', {})

                if not isinstance(data['automic_sentence'], list):
                    data['automic_sentence'] = []
                if not isinstance(data['SVC_keywords'], dict):
                    data['SVC_keywords'] = {}

                return {
                    'index': idx,
                    'original_text': original_text,
                    'corrected_text': corrected_text,
                    'matching_question': data.get('matching_question', False),
                    'pos_neg': data.get('pos_neg', 'NEUTRAL') or 'NEUTRAL',
                    'automic_sentence': data.get('automic_sentence', []),
                    'svc_keywords': data.get('SVC_keywords', {})
                }, row_cost

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(process_row, item): item['index'] for item in process_items}

                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        print(f"Error processing response #{idx + 1}: {exc}")
                        continue

                    if not result:
                        continue

                    row_payload, row_cost = result
                    results_map[idx] = row_payload
                    total_price += row_cost
                    processed_count += 1
                    print(f"Processed {processed_count}/{total_targets} responses, Total cost: ${total_price:.4f}")

            result_data = []
            for idx in sorted(results_map.keys()):
                payload = results_map[idx]
                original_text = payload['original_text']
                corrected_text = payload['corrected_text']
                automic_sentence = payload.get('automic_sentence', [])
                svc_keywords = payload.get('svc_keywords', {})

                row_data = {
                    'id': idx,
                    'pos.neg': payload.get('pos_neg', 'NEUTRAL'),
                    'matching_question': payload.get('matching_question', False),
                    'org_text': original_text,
                    'correction_text': corrected_text,
                    'org_text_embed': embedding_provider.encode(original_text) if original_text.strip() else [],
                    'correction_text_embed': embedding_provider.encode(corrected_text) if corrected_text.strip() else [],
                    'sentence_1': automic_sentence[0] if len(automic_sentence) > 0 else None,
                    'sentence_2': automic_sentence[1] if len(automic_sentence) > 1 else None,
                    'sentence_3': automic_sentence[2] if len(automic_sentence) > 2 else None,
                    'sentence_1_embed': embedding_provider.encode(automic_sentence[0]) if len(automic_sentence) > 0 and automic_sentence[0] and str(automic_sentence[0]).strip() else [],
                    'sentence_2_embed': embedding_provider.encode(automic_sentence[1]) if len(automic_sentence) > 1 and automic_sentence[1] and str(automic_sentence[1]).strip() else [],
                    'sentence_3_embed': embedding_provider.encode(automic_sentence[2]) if len(automic_sentence) > 2 and automic_sentence[2] and str(automic_sentence[2]).strip() else [],
                    'S_1': ', '.join(svc_keywords.get('sentence1', {}).get('S', [])),
                    'V_1': ', '.join(svc_keywords.get('sentence1', {}).get('V', [])),
                    'C_1': ', '.join(svc_keywords.get('sentence1', {}).get('C', [])),
                    'S_2': ', '.join(svc_keywords.get('sentence2', {}).get('S', [])),
                    'V_2': ', '.join(svc_keywords.get('sentence2', {}).get('V', [])),
                    'C_2': ', '.join(svc_keywords.get('sentence2', {}).get('C', [])),
                    'S_3': ', '.join(svc_keywords.get('sentence3', {}).get('S', [])),
                    'V_3': ', '.join(svc_keywords.get('sentence3', {}).get('V', [])),
                    'C_3': ', '.join(svc_keywords.get('sentence3', {}).get('C', []))
                }
                result_data.append(row_data)
        
        # DataFrame 생성
        result_df = pd.DataFrame(result_data)
        
        # CSV 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 프로젝트 매니저를 통해 CSV 파일 경로 생성
        project_name = state.get('project_name', 'unknown')
        project_manager = get_project_manager(project_name)
        
        csv_path = project_manager.get_stage2_csv_path(
            current_question_id, 
            current_question_type, 
            timestamp
        )
        
        result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"SENTENCE processing completed for {current_question_id}")
        print(f"Results saved to: {csv_path}")
        print(f"Total processing cost: ${total_price:.4f}")
        print(f"Processed {len(result_df)} rows")
        
        # 기존 matched_questions에 데이터 경로 정보 추가
        matched_questions = state.get('matched_questions', {})
        if current_question_id in matched_questions:
            matched_questions[current_question_id]['stage2_data'] = {
                'csv_path': csv_path,
                'processing_type': 'SENTENCE',
                'rows_count': len(result_df),
                'timestamp': timestamp,
                'total_cost': total_price,
                'status': 'completed'
            }
        
        # 결과 상태 반환
        current_total_cost = state.get('total_llm_cost_usd', 0.0)
        updated_total_cost = current_total_cost + total_price
        
        final_state = {
            **state,
            'stage2_sentence_processed': True,
            'stage2_processing_type': 'SENTENCE',
            'stage2_status': 'completed',
            'stage2_result_csv': csv_path,
            'stage2_processed_rows': len(result_df),
            'stage2_total_cost': total_price,
            'total_llm_cost_usd': updated_total_cost,  # 전체 LLM 비용 누적 업데이트
            'matched_questions': matched_questions  # 업데이트된 matched_questions
        }
        
        # Stage 2 SENTENCE 처리 후 state 저장
        project_name = state.get('project_name')
        if project_name and settings.SAVE_STATE_LOG:
            project_manager = get_project_manager(project_name)
            config = {'save_state_log': settings.SAVE_STATE_LOG}
            # current_stage 업데이트 후 저장
            final_state['current_stage'] = f'STAGE2_SENTENCE_{current_question_id}_COMPLETED'
            project_manager.save_state(dict(final_state), config)
        
        return final_state
        
    except Exception as e:
        print(f"Error in SENTENCE processing for {current_question_id}: {e}")
        return {
            **state,
            'stage2_error': f"SENTENCE processing failed: {str(e)}",
            'stage2_status': 'error',
            'stage2_processing_type': 'SENTENCE'
        }
