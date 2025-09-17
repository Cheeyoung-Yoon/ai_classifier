"""
Stage2 SENTENCE Node - Handles SENTENCE type questions
"""
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from io_layer.llm.client import LLMClient
from io_layer.embedding import VectorEmbedding
from .prep_sentence import get_column_locations, extract_question_choices
from utils.project_manager import get_project_manager
from config.config import settings
from config.prompt.prompt_loader import load_prompt_config, resolve_branch
from utils.pydantic_utils import extract_llm_response_data


def stage2_sentence_node(state: Dict[str, Any], deps: Optional[Any] = None) -> Dict[str, Any]:
    """
    SENTENCE 타입 질문 처리 노드
    
    SENTENCE 타입은 depend, depend_pos_neg, pos_neg 등으로 텍스트 전처리가 필요
    두 개의 LLM을 사용: 1) 문법 교정 (gpt-4.1), 2) 문장 분석 (gpt-4.1-nano)
    
    Args:
        state: Current graph state
        deps: Dependencies (contains llm_client)
        
    Returns:
        Updated state with sentence processing results and CSV file path
    """
    current_question_id = state.get('current_question_id')
    current_question_type = state.get('current_question_type')
    
    print(f"stage2_sentence_node: Processing SENTENCE type question {current_question_id} ({current_question_type})")
    
    # VectorEmbedding 인스턴스 생성
    embed = VectorEmbedding()
    
    try:
        # Config에서 prompt 로드
        prompt_config = load_prompt_config()
        grammar_branch = resolve_branch(prompt_config, "sentence_grammar_check")
        
        # Question type에 따른 analysis branch 선택
        analysis_branch_key = f"sentence_{current_question_type}_split"
        analysis_branch = resolve_branch(prompt_config, analysis_branch_key)
        
        # Fallback: 기본 sentence_only branch 사용
        if not analysis_branch:
            analysis_branch = resolve_branch(prompt_config, "sentence_only")
            
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
        
        # 두 단계 LLM 처리
        result_data = []
        total_price = 0
        
        # 테스트용으로 10개만 처리
        # test_limit = min(10, len(text_df))
        test_limit = len(text_df)
        # print(f"🧪 TEST MODE: Processing only {test_limit} rows (out of {len(text_df)})")
        
        for i in range( len(text_df)):
            # 원본 텍스트 추출
            row_texts = text_df.iloc[i].dropna().astype(str).tolist()
            if not row_texts:
                continue
            original_text = row_texts[0].strip()
            
            # 1단계: 문법 교정 (llm_client_large)
            if grammar_branch:
                # Config에서 로드된 prompt 사용
                grammar_prompt = grammar_branch['user_template'].format(
                    survey_context=survey_context,
                    answer=original_text
                )
                grammar_system = grammar_branch['system']
            else:
                # Fallback: 기존 하드코딩된 prompt
                grammar_prompt = f"""
summary of the survey: {survey_context}
answer: {original_text}
"""
                grammar_system = """Correct grammar and typos in Korean survey responses.

- Correct ONLY the answer, making it natural Korean.  
- Use the question only as context if needed.  

Return result as valid JSON in the format:  
{ "corrected": "<corrected answer>"}  

Rules:  
- Do not change the meaning.  
- Correct typos and unnatural expressions into the most natural and common form in Korean survey responses.  
- When multiple corrections are possible, prefer the one that best fits everyday consumer feedback context.  
- Do not output anything except the JSON."""
            
            grammar_response = llm_client_large.chat(
                system=grammar_system,
                user=grammar_prompt,
                schema=grammar_branch.get('schema') if grammar_branch else None
            )
            
            # Schema 기반 응답 처리
            if grammar_branch and grammar_branch.get('schema'):
                data = extract_llm_response_data(grammar_response[0])
                corrected_text = data.get('corrected', original_text)
            else:
                # Fallback: JSON 파싱
                try:
                    corrected_data = json.loads(grammar_response[0])
                    corrected_text = corrected_data['corrected']
                except json.JSONDecodeError:
                    corrected_text = original_text
                
            # 비용 계산
            if hasattr(grammar_response[1], 'cost_usd'):
                total_price += grammar_response[1].cost_usd
            elif hasattr(grammar_response[1], 'total_cost'):
                total_price += grammar_response[1].total_cost
            
            # 2단계: 문장 분석 (llm_client_nano)
            if analysis_branch:
                # Config에서 로드된 prompt 사용
                if current_question_type in ["depend", "depend_pos_neg"] and depend_df is not None:
                    depend_value = str(int(depend_df.iloc[i, 0]))
                    # response_dict가 dict인지 확인하고 안전하게 접근
                    if isinstance(response_dict, dict):
                        sub_explanation = response_dict.get(depend_value, "")
                    else:
                        sub_explanation = ""
                        print(f"Warning: response_dict is not a dictionary, type: {type(response_dict)}")
                    
                    analysis_prompt = analysis_branch['user_template'].format(
                        survey_context=survey_context,
                        question_summary=question_summary,
                        sub_explanation=sub_explanation,
                        corrected_answer=corrected_text
                    )
                else:
                    analysis_prompt = analysis_branch['user_template'].format(
                        survey_context=survey_context,
                        question_summary=question_summary,
                        corrected_answer=corrected_text
                    )
                
                analysis_system = analysis_branch['system']
            else:
                # Fallback: 기존 하드코딩된 prompt
                if current_question_type in ["depend", "depend_pos_neg"] and depend_df is not None:
                    depend_value = str(int(depend_df.iloc[i, 0]))
                    # response_dict가 dict인지 확인하고 안전하게 접근
                    if isinstance(response_dict, dict):
                        sub_explanation = response_dict.get(depend_value, "")
                    else:
                        sub_explanation = ""
                        print(f"Warning: response_dict is not a dictionary, type: {type(response_dict)}")
                    
                    analysis_prompt = f"""
question: {question_summary}
sub_explanation: {sub_explanation}
answer: {corrected_text}
"""
                else:
                    analysis_prompt = f"""
question: {question_summary}
answer: {corrected_text}
"""
                
                # 시스템 프롬프트 (질문 타입별)
                if current_question_type == "depend_pos_neg":
                    pos_neg_rule = "7. **Pos/Neg classification**: If the question is about sentiment (positive/negative), classify the answer accordingly with sub_explanation and set `\"pos.neg\"` to `\"POSITIVE\"`or `\"NEGATIVE\"`."
                elif current_question_type == "pos_neg":
                    pos_neg_rule = "7. **Pos/Neg classification**: If the question is about sentiment (positive/negative), classify the answer accordingly and set `\"pos.neg\"` to `\"POSITIVE\"`or `\"NEGATIVE\"`."
                else:
                    pos_neg_rule = "7. pos.neg is just placeholder. just return \"NEUTRAL\""
                
                analysis_system = f"""You are a survey response interpreter.

### Rules
1. **Nuance reference**: Context fields are only for reference. Do not include them in the final output.  
2. type-of-error (typo) detection: Identify and correct any typographical errors in the answer including spelling mistakes, grammatical errors, and incorrect word usage.  
3. **Question matching judgment**:  
   - Set `"matching_question": true` unless the answer is clearly irrelevant or meaningless.  
   - Even short or abstract answers like *"trustworthy"*, *"it feels reliable"*, *"good"*, *"satisfying"* should be treated as `true`.  
   - Set `"matching_question": false` only if the answer is irrelevant, empty, meaningless tokens, or pure emotional expression with no semantic relation (e.g., "Wow!", "I'm so excited").  
   - Ignore filler tokens such as `merged`, `nan`, `NaN`, `None`, `NULL`, or `Name: 0, dtype: object` before evaluation.  
4. **Atomic sentence split**: If the answer contains multiple semantic units, split it into at most 3 atomic sentences (Subject–Verb–Complement based).  
5. **S/V/C keyword extraction**: For each atomic sentence, extract core keywords:  
   - S = subject (main entity)  
   - V = verb (main action/state) format in "-다" form
   - C = complement/object (if any)  
   - Only extract essential words, not particles or function words.  
6. **Output format**: Return the result format strictly in the following JSON format (schema stays in Korean):
{pos_neg_rule}

{{{{
  "matching_question": true/false,
  "pos.neg": "POSITIVE"/"NEGATIVE"/"NEUTRAL",
  "automic_sentence": ["문장1", "문장2", "문장3"],
  "SVC_keywords": {{{{
      "sentence1": {{{{
    "S": [],
    "V": [],
    "C": []}}}},
      "sentence2": {{{{
    "S": [],
    "V": [],
    "C": []}}}},
      "sentence3": {{{{
    "S": [],
    "V": [],
    "C": []
  }}}}
}}}}"""
            
            analysis_response = llm_client_nano.chat(
                system=analysis_system,
                user=analysis_prompt,
                schema=analysis_branch.get('schema') if analysis_branch else None
            )
            
            # Schema 기반 응답 처리
            if analysis_branch and analysis_branch.get('schema'):
                data = extract_llm_response_data(analysis_response[0])
            else:
                # Fallback: JSON 파싱
                try:
                    data = json.loads(analysis_response[0])
                except json.JSONDecodeError:
                    try:
                        data = json.loads(analysis_response[0].replace('```json', '').replace('```', ''))
                    except:
                        continue
            
            # data가 list인 경우 첫 번째 요소를 사용하거나 기본 dict 구조 생성
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    data = data[0]
                else:
                    # 기본 구조 생성
                    data = {
                        'matching_question': False,
                        'pos.neg': 'NEUTRAL',
                        'automic_sentence': [],
                        'SVC_keywords': {}
                    }
            elif not isinstance(data, dict):
                # data가 dict도 list도 아닌 경우 기본 구조 생성
                data = {
                    'matching_question': False,
                    'pos.neg': 'NEUTRAL', 
                    'automic_sentence': [],
                    'SVC_keywords': {}
                }
                    
            # 비용 계산
            if hasattr(analysis_response[1], 'cost_usd'):
                total_price += analysis_response[1].cost_usd
            elif hasattr(analysis_response[1], 'total_cost'):
                total_price += analysis_response[1].total_cost
            
            # 데이터 검증 및 기본값 설정
            if not isinstance(data, dict):
                data = {
                    'matching_question': False,
                    'pos_neg': 'NEUTRAL',
                    'automic_sentence': [],
                    'SVC_keywords': {}
                }
            
            # 기본값 설정
            data.setdefault('matching_question', False)
            data.setdefault('pos_neg', 'NEUTRAL') 
            data.setdefault('automic_sentence', [])
            data.setdefault('SVC_keywords', {})
            
            # automic_sentence가 리스트인지 확인
            if not isinstance(data['automic_sentence'], list):
                data['automic_sentence'] = []
                
            # SVC_keywords가 dict인지 확인
            if not isinstance(data['SVC_keywords'], dict):
                data['SVC_keywords'] = {}
            
            # 결과 DataFrame 행 생성
            automic_sentence = data['automic_sentence']
            svc_keywords = data['SVC_keywords']
            
            row_data = {
                'id': i,
                'pos.neg': data.get('pos_neg', 'NEUTRAL'),
                'matching_question': data.get('matching_question', False),
                'org_text': original_text,
                'correction_text': corrected_text,
                'org_text_embed': embed.embed(original_text) if original_text.strip() else [],
                'correction_text_embed': embed.embed(corrected_text) if corrected_text.strip() else [],
                'sentence_1': automic_sentence[0] if len(automic_sentence) > 0 else None,
                'sentence_2': automic_sentence[1] if len(automic_sentence) > 1 else None,
                'sentence_3': automic_sentence[2] if len(automic_sentence) > 2 else None,
                'sentence_1_embed': embed.embed(automic_sentence[0]) if len(automic_sentence) > 0 and automic_sentence[0] and str(automic_sentence[0]).strip() else [],
                'sentence_2_embed': embed.embed(automic_sentence[1]) if len(automic_sentence) > 1 and automic_sentence[1] and str(automic_sentence[1]).strip() else [],
                'sentence_3_embed': embed.embed(automic_sentence[2]) if len(automic_sentence) > 2 and automic_sentence[2] and str(automic_sentence[2]).strip() else [],
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
            print(f"Processed {i+1}/{test_limit} responses, Total cost: ${total_price:.4f}")
        
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