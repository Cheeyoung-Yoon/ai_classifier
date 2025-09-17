# nodes/stage1_data_preparation/survey_parser.py
# Survey parser for Stage 1 - Data Preparation

from tools.llm_router import LLMRouter, CreditError, OpenAIError, LLMError
from pydantic import BaseModel
from typing import List, Optional
from graph.state import GraphState
import json

class OpenEndedQuestion(BaseModel):
    """개방형 질문 하나의 정보"""
    question_that_is_related: Optional[str] = None
    open_question_number: str
    question_text: str
    question_type: str  # img | concept | depend | pos_neg | depend_pos_neg | etc | etc_pos_neg


class SurveyParseResult(BaseModel):
    """설문 파싱 결과 - 개방형 질문들의 리스트"""
    questions: List[OpenEndedQuestion]


def parse_survey_original(state: dict, branch : str = "survey_parser_4.1", deps=None) -> dict:
    """기존 함수 - 단독 사용시"""
    text = state["file"]["text"]
    
    # LLMRouter 인스턴스 생성
    if deps and hasattr(deps, 'llm_router'):
        router = deps.llm_router
    else:
        router = LLMRouter()
    
    out = router.run(
        branch=branch, 
        variables={"text": text, "label_rules": state.get("label_rules", "")},
        schema=SurveyParseResult,
    )

    try:
        parsed_survey = out['result']['parsed']
        
        dict_out_survey = {
        "parsed_questions": [
            {
                "open_question_number": q.open_question_number,
                "question_text": q.question_text,
                "question_type": q.question_type,
                "question_that_is_related": q.question_that_is_related
            }
            for q in parsed_survey.questions
        ]
    }

    except (KeyError, AttributeError, TypeError) as e:
        print(f"extract/iterate failed: {e}")      # 무엇이 터졌는지 로그로 남기기
        dict_out_survey = {"questions": []}
        
    
    return {"doc_parsed": dict_out_survey, "llm_log": out["usage"], "llm_meta": {"branch": out["branch"], "model": out["model"]}}

def parse_survey_node(state: GraphState, branch: str = "survey_parser_4.1", deps=None) -> GraphState:
    """LangGraph용 노드 함수"""
    try:
        text = state["raw_survey_info"]["text"]
        
        # LLMRouter 인스턴스 생성
        if deps and hasattr(deps, 'llm_router'):
            router = deps.llm_router
        else:
            router = LLMRouter()
        
        out = router.run(
            branch=branch, 
            variables={"text": text, "label_rules": ""},
            schema=SurveyParseResult,
        )
        
        try:
            parsed_survey = out['result']['parsed']
            
            dict_out_survey = {
                "parsed": {
                    "questions": [  # ✅ 'questions' 키로 통일
                        {
                            "open_question_number": q.open_question_number,
                            "question_text": q.question_text,
                            "question_type": q.question_type,
                            "question_that_is_related": q.question_that_is_related
                        }
                        for q in parsed_survey.questions
                    ]
                }
            }
        except (KeyError, AttributeError, TypeError) as e:
            print(f"extract/iterate failed: {e}")      # 무엇이 터졌는지 로그로 남기기
            dict_out_survey = {
                "parsed": {
                    "questions": []
                }
            }
            
        state["parsed_survey"] = dict_out_survey
        
        # 로그 정보 저장
        if state.get("llm_logs") is None:
            state["llm_logs"] = []
        if state.get("llm_meta") is None:
            state["llm_meta"] = []
            
        state["llm_logs"].append(out["usage"])
        state["llm_meta"].append({"branch": out["branch"], "model": out["model"]})
        
        # Memory cleanup은 이제 flush 노드에서 처리
        
        return state
    except CreditError as e:
        state["error"] = f"Credit Error: {str(e)}"
        state["error_type"] = "credit"
        return state
    except OpenAIError as e:
        state["error"] = f"OpenAI Error: {str(e)}"
        state["error_type"] = "openai_error"
        return state
    except LLMError as e:
        state["error"] = f"LLM Error: {str(e)}"
        state["error_type"] = "llm_error"
        return state
    except Exception as e:
        state["error"] = f"Survey parsing error: {str(e)}"
        state["error_type"] = "unknown"
        return state