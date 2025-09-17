# nodes/stage1_data_preparation/survey_context.py
# Survey context extraction for Stage 1 - Data Preparation

from tools.llm_router import LLMRouter, CreditError, OpenAIError, LLMError
from typing import Optional
from graph.state import GraphState


def survey_context_node(state: GraphState, branch: str = "survey_context_summarizer", deps=None) -> GraphState:
    """
    Extract overall survey context and purpose using LLM
    
    This node analyzes the raw survey text to understand what the survey is about,
    providing context that can be useful for later stages of processing.
    
    Args:
        state: Current graph state containing raw_survey_info
        branch: LLM router branch to use for context extraction
        deps: Optional dependencies containing llm_router instance
        
    Returns:
        Updated state with survey_context field populated
    """
    try:
        # Get the raw survey text
        if not state.get("raw_survey_info") or not state["raw_survey_info"].get("text"):
            state["error"] = "No raw survey text available for context extraction"
            state["error_type"] = "missing_data"
            return state
            
        text = state["raw_survey_info"]["text"]
        
        # Initialize LLM router
        if deps and hasattr(deps, 'llm_router'):
            router = deps.llm_router
        else:
            router = LLMRouter()
        
        print(f"DEBUG: Extracting survey context from {len(text)} characters")
        
        # Call LLM to extract survey context
        out = router.run(
            branch=branch,
            variables={"text": text}
        )
        
        # Extract the context summary
        survey_context = out['result']
        
        # Store in state
        state["survey_context"] = survey_context
        
        # Log LLM usage
        if state.get("llm_logs") is None:
            state["llm_logs"] = []
        if state.get("llm_meta") is None:
            state["llm_meta"] = []
            
        state["llm_logs"].append(out["usage"])
        state["llm_meta"].append({"branch": out["branch"], "model": out["model"]})
        
        print(f"DEBUG: Survey context extracted: {survey_context[:100]}...")
        
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
        state["error"] = f"Survey context extraction error: {str(e)}"
        state["error_type"] = "unknown"
        return state