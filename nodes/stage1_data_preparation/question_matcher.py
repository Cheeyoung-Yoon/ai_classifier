# nodes/stage1_data_preparation/question_matcher.py
# Question-data matcher for Stage 1 - Data Preparation

from tools.llm_router import LLMRouter, CreditError, OpenAIError, LLMError
from typing import Dict, Any, Optional, List
from graph.state import GraphState
from utils.data_utils import DataHelper
import json

def question_data_matcher_original(
    state: Dict[str, Any],
    branch: str = "question_data_matcher",
    deps: Optional[Any] = None
) -> Dict[str, Any]:
    """기존 함수 - 단독 사용시 (하위 호환성)"""
    router = getattr(deps, "llm_router", LLMRouter())

    out = router.run(
        branch=branch,
        variables={
            "survey_questions": state["survey_questions"],
            "headers_by_row": state["headers_by_row"],
            "column_names": state["column_names"],
            "data_sample": state["data_sample"],
        },
    )
    return {
        "question_data_match": out["result"],  # dict 그대로
        "llm_log": out["usage"],
        "llm_meta": {"branch": out["branch"], "model": out["model"]},
    }

def question_data_matcher_from_path( parsed_survey, data_result, open_columns=None):
    """
    Path 기반으로 question-data 매칭 수행
    
    Args:
        data_path: CSV 파일 경로
        parsed_survey: 파싱된 설문 결과 (question들 포함)
        data_result: load_data_file의 결과 (column_labels 포함)
        open_columns: 오픈 컬럼 리스트 (선택사항)
    """
    from utils.data_utils import DataHelper
    from tools.llm_router import call_llm_router
    from io_layer.llm.client import CreditError, OpenAIError, LLMError
    import pandas as pd
    data_path = data_result['raw_data_info']['dataframe_path']
    try:
        # 1. 데이터 샘플 로드 (첫 5행)
        df_sample = pd.read_csv(data_path, nrows=5)
        data_sample = df_sample.to_dict('records')
        
        # 2. column_labels를 headers_by_row로 변환
        headers_by_row = []
        if data_result and 'raw_data_info' in data_result:
            column_labels = data_result['raw_data_info']['meta'].get('column_labels', [])
            if column_labels:
                # column_labels: [['Q0200', 'TEL'], ['Q0300', 'NAME'], ...]
                # headers_by_row로 변환: [['Q0200', 'Q0300', ...], ['TEL', 'NAME', ...]]
                col_names = [label[0] for label in column_labels]
                label_names = [label[1] for label in column_labels]
                headers_by_row = [col_names, label_names]
        
        # 3. 파싱된 설문 질문들 추출
        survey_questions = []
        if parsed_survey and 'questions' in parsed_survey:  # ✅ 'questions' 키 사용
            for q in parsed_survey['questions']:
                survey_questions.append({
                    'question_number': q.get('open_question_number'),
                    'question_text': q.get('question_text', ''),
                    'question_type': q.get('question_type'),
                    'question_that_is_related': q.get('question_that_is_related')
                })
        
        # 4. 데이터 정보
        df_info = DataHelper.get_dataframe_info(data_path)
        
        # 5. LLM 파라미터 준비 (config의 프롬프트 템플릿에 맞게)
        llm_params = {
            'data_sample': data_sample,
            'headers_by_row': headers_by_row,
            'survey_questions': survey_questions
        }
        
        # 6. LLM 호출
        try:
            result = call_llm_router("question_data_matcher", **llm_params)
            return {
                'question_column_mapping': result
                # 'data_sample': data_sample[:2],
                # 'headers_used': headers_by_row,
                # 'total_questions': len(survey_questions),
                # 'total_columns': len(df_info['columns']),
                # 'open_columns_used': open_columns or []
            }
        except (CreditError, OpenAIError, LLMError) as e:
            print(f"LLM Error in question_data_matcher: {e}")
            # Mock 매핑 생성 (column_labels 기반)
            mock_mapping = {}
            for i, q in enumerate(survey_questions):
                q_num = q['question_number']
                if i < len(column_labels):
                    mock_mapping[str(q_num)] = [column_labels[i][0]]  # 컬럼명만
                else:
                    mock_mapping[str(q_num)] = []
            
            return {
                'question_column_mapping': mock_mapping,
                'data_sample': data_sample[:2], 
                'headers_used': headers_by_row,
                'total_questions': len(survey_questions),
                'total_columns': len(df_info['columns']),
                'open_columns_used': open_columns or [],
                'error': str(e)
            }
    
    except Exception as e:
        return {
            'error': f"Question-data matching failed: {str(e)}",
            'question_column_mapping': {},
            'data_sample': [],
            'headers_used': [],
            'total_questions': 0,
            'total_columns': 0
        }

def question_data_matcher_node(
    state: GraphState,
    branch: str = "question_data_matcher",
    deps: Optional[Any] = None
) -> GraphState:
    """LangGraph용 노드 함수 - 경로 기반"""
    try:
        print(f"DEBUG: Starting question_data_matcher_node")
        print(f"DEBUG: State keys: {list(state.keys())}")
        
        # 경로 기반 처리
        if ("raw_data_info" in state and "dataframe_path" in state["raw_data_info"] and 
            "open_columns" in state and "parsed_survey" in state):
        
            print(f"DEBUG: Using path-based processing")
            open_columns = state["open_columns"]
            
            # parsed_survey는 dict이므로 올바르게 접근
            parsed_survey = state["parsed_survey"]
            print(f"DEBUG: parsed_survey type: {type(parsed_survey)}")
            print(f"DEBUG: parsed_survey keys: {list(parsed_survey.keys()) if isinstance(parsed_survey, dict) else 'not dict'}")
            
            if isinstance(parsed_survey, dict) and 'parsed' in parsed_survey:
                survey_result = parsed_survey['parsed']
                print(f"DEBUG: survey_result type: {type(survey_result)}")
                
                # ✅ 수정: dict 형태로 저장되어 있으므로 'questions' 키로 접근
                if isinstance(survey_result, dict) and 'questions' in survey_result:
                    questions_list = survey_result['questions']  # 이미 dict 리스트 형태
                    print(f"DEBUG: Found {len(questions_list)} questions in dict format")
                elif hasattr(survey_result, 'questions'):
                    # 객체 형태인 경우 (fallback)
                    survey_questions = survey_result.questions
                    questions_list = []
                    for q in survey_questions:
                        questions_list.append({
                            "open_question_number": q.open_question_number,
                            "question_text": q.question_text,
                            "question_type": q.question_type,
                            "question_that_is_related": q.question_that_is_related
                        })
                else:
                    print(f"DEBUG: survey_result has no questions")
                    state["error"] = "No questions found in parsed survey result"
                    return state
            else:
                # Fallback for other formats
                survey_result = parsed_survey
                if hasattr(survey_result, 'questions'):
                    survey_questions = survey_result.questions
                    # Convert to list of dicts for compatibility
                    questions_list = []
                    for q in survey_questions:
                        questions_list.append({
                            "open_question_number": q.open_question_number,
                            "question_text": q.question_text,
                            "question_type": q.question_type,
                            "question_that_is_related": q.question_that_is_related
                        })
                else:
                    questions_list = []
                print(f"DEBUG: Using fallback, found {len(questions_list)} questions")
            
            print(f"DEBUG: Calling question_data_matcher_from_path with {len(questions_list)} questions")
            result = question_data_matcher_from_path(
                parsed_survey={'questions': questions_list},
                data_result=state,
                open_columns=open_columns,
            )
            print(f"DEBUG: Result: {result}")
            print(f"DEBUG: Result type: {type(result)}")
            print(f"DEBUG: Result keys: {list(result.keys()) if isinstance(result, dict) else 'not dict'}")
            
            if "question_column_mapping" not in result:
                print(f"ERROR: 'question_column_mapping' not found in result!")
                print(f"Available keys: {list(result.keys()) if isinstance(result, dict) else result}")
                state["error"] = "question_column_mapping not found in matcher result"
                return state
            
            # Map the result to the expected state keys
            state["question_data_match"] = result["question_column_mapping"]
            
            # Memory cleanup은 이제 flush 노드에서 처리
            
            # For path-based processing, we already have LLM logs from the routing call
            # No additional LLM logs to add since question_data_matcher_from_path doesn't use LLM directly
            print(f"DEBUG: Path-based processing completed successfully")
            return state
            
        elif "raw_dataframe_path" in state:
            # 하위 호환성: DataFrame 객체가 직접 전달된 경우
            import pandas as pd
            df = state["raw_dataframe_path"] if isinstance(state["raw_dataframe_path"], pd.DataFrame) else pd.read_csv(state["raw_dataframe_path"], encoding='utf-8-sig', index_col=0)
            
            open_cols = state["open_columns"]
            data_sample = df[open_cols].head(3).to_dict()
            
            router = getattr(deps, "llm_router", LLMRouter())
            
            # parsed_survey는 dict이므로 올바르게 접근
            parsed_survey = state["parsed_survey"]
            if isinstance(parsed_survey, dict) and 'parsed' in parsed_survey:
                survey_result = parsed_survey['parsed']
                if isinstance(survey_result, dict) and 'questions' in survey_result:
                    survey_questions = survey_result['questions']
                elif hasattr(survey_result, 'questions'):
                    # 객체를 dict로 변환
                    survey_questions = [
                        {
                            "open_question_number": q.open_question_number,
                            "question_text": q.question_text,
                            "question_type": q.question_type,
                            "question_that_is_related": q.question_that_is_related
                        }
                        for q in survey_result.questions
                    ]
                else:
                    survey_questions = []
            else:
                # Fallback for other formats
                survey_questions = parsed_survey.questions if hasattr(parsed_survey, 'questions') else []
            
            out = router.run(
                branch=branch,
                variables={
                    "survey_questions": survey_questions,
                    "headers_by_row": df.columns.tolist(),
                    "column_names": open_cols,
                    "data_sample": data_sample,
                },
            )
            
            state["question_data_match"] = out["result"]
            result = {"llm_log": out["usage"], "llm_meta": {"branch": out["branch"], "model": out["model"]}}
            
            # Memory cleanup은 이제 flush 노드에서 처리
        else:
            raise KeyError("No dataframe_path or raw_dataframe_path found in state")
        
        # 로그 정보 저장 (DataFrame 경로에서만)
        if state.get("llm_logs") is None:
            state["llm_logs"] = []
        if state.get("llm_meta") is None:
            state["llm_meta"] = []
            
        state["llm_logs"].append(result["llm_log"])
        state["llm_meta"].append(result["llm_meta"])
        
        return state
    except Exception as e:
        state["error"] = f"Question matching error: {str(e)}"
        return state