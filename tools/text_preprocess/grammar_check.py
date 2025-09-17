"""
Grammar check module for text preprocessing
"""
import json
from typing import Dict, Any, Optional
from io_layer.llm.client import LLMClient


def grammar_check(llm_client: LLMClient, text: str = "", survey_context: str = "") -> str:
    """
    Corrects grammar and typos in Korean text using LLM client
    
    Args:
        llm_client: LLM client instance
        text: Text to be corrected
        context: Survey context for reference
        
    Returns:
        Corrected text
    """
    system_prompt = """You are a Korean grammar and spelling corrector.  

You will be given:  
- summary of the survey (context)  
- An answer (may contain grammar, spelling, or QWERTY-typo mistakes).  
- Most errors are QWERTY keyboard typos, not phonetic or semantic substitutions.  
If a typo looks like it came from a nearby key press, prioritize that correction over more common dictionary words.  

Your task:  
- Correct ONLY the answer, making it natural Korean.  
- Use the question only as context if needed.  

Return result as valid JSON in the format:  
{ "corrected": "<corrected answer>"}  

Rules:  
- Do not change the meaning.  
- Correct typos and unnatural expressions into the most natural and common form in Korean survey responses.  
- When multiple corrections are possible, prefer the one that best fits everyday consumer feedback context.  
- Do not output anything except the JSON.
"""
    
    user_prompt = f"""
    summary of the survey: {survey_context}
    answer: {text.strip()}
    """
    
    try:
        response = llm_client.chat(
            system=system_prompt,
            user=user_prompt
        )
        
        result = json.loads(response[0])
        return result.get('corrected', text)
    except Exception as e:
        print(f"Grammar check error: {e}")
        # If processing fails, return original text
        return text