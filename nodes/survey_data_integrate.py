from tools.file_preprocess.dict_integration import data_integration
from nodes.shared.stage_tracker import update_stage_tracking


def survey_data_integrate(survey_questions, question_data_match):
    """
    LangGraph node: survey_questions + question_data_match → integrated_map
    state 요구:
      - survey_questions: list
      - question_data_match: dict
    """
    integrated_map = data_integration(
        survey_questions,
        question_data_match
    )

    return {
        "integrated_map": integrated_map
    }


def survey_data_integrate_node(state):
    """
    LangGraph node: survey_questions + question_data_match → integrated_map
    state 요구:
      - parsed_survey: dict with 'parsed' -> 'questions'
      - question_data_match: dict
    """
    print(f"DEBUG: survey_data_integrate_node starting")
    print(f"DEBUG: State keys: {list(state.keys())}")
    
    # 1. survey_questions 추출 - 새로운 구조에 맞게
    survey_questions = []
    parsed_survey = state.get("parsed_survey", {})
    print(f"DEBUG: parsed_survey type: {type(parsed_survey)}")
    
    if isinstance(parsed_survey, dict) and 'parsed' in parsed_survey:
        survey_result = parsed_survey['parsed']
        if isinstance(survey_result, dict) and 'questions' in survey_result:
            survey_questions = survey_result['questions']
            print(f"DEBUG: Found {len(survey_questions)} questions in dict format")
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
            print(f"DEBUG: Converted {len(survey_questions)} questions from objects to dicts")
    elif hasattr(parsed_survey, 'questions'):
        # 직접 questions 속성이 있는 경우 (fallback)
        survey_questions = [
            {
                "open_question_number": q.open_question_number,
                "question_text": q.question_text,
                "question_type": q.question_type,
                "question_that_is_related": q.question_that_is_related
            }
            for q in parsed_survey.questions
        ]
        print(f"DEBUG: Fallback - converted {len(survey_questions)} questions")
    else:
        print(f"DEBUG: No valid survey questions found")
        
    # 2. question_data_match 확인
    question_data_match = state.get("question_data_match", {})
    print(f"DEBUG: question_data_match type: {type(question_data_match)}")
    print(f"DEBUG: question_data_match: {question_data_match}")
    
    # 3. 빈 매핑 처리
    if not question_data_match:
        print("WARNING: question_data_match is empty, creating mock mapping")
        # Mock 매핑 생성
        question_data_match = {}
        for i, q in enumerate(survey_questions):
            if isinstance(q, dict):
                q_num = q.get('open_question_number') or q.get('question_number')
            else:
                q_num = getattr(q, 'open_question_number', f'Q{i}')
            question_data_match[str(q_num)] = []  # 빈 매핑
    
    try:
        # data_integration 호출
        integrated_map = data_integration(
            survey_questions,
            question_data_match
        )
        print(f"DEBUG: data_integration successful, result type: {type(integrated_map)}")
        
    except Exception as e:
        print(f"ERROR: data_integration failed: {e}")
        import traceback
        traceback.print_exc()
        # 빈 결과 반환
        integrated_map = {}
    
    # 전체 state를 복사하고 새 필드 추가
    updated_state = state.copy()
    updated_state["matched_questions"] = integrated_map
    
    print(f"DEBUG: survey_data_integrate_node completed")
    return updated_state