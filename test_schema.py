#!/usr/bin/env python3
"""
Quick test for schema-based LLM calls
"""
import sys
sys.path.append('/home/cyyoon/test_area/ai_text_classification/2.langgraph')

from config.prompt.prompt_loader import resolve_branch
from io_layer.llm.client import LLMClient

def test_schema_resolution():
    """Test if schema resolution works"""
    print("Testing schema resolution...")
    
    # Test grammar check
    try:
        grammar_branch = resolve_branch("sentence_grammar_check")
        print(f"Grammar branch resolved: {grammar_branch}")
        print(f"Schema: {grammar_branch.get('schema')}")
    except Exception as e:
        print(f"Error resolving grammar branch: {e}")
    
    # Test sentence analysis
    try:
        analysis_branch = resolve_branch("sentence_depend_pos_neg_split")
        print(f"Analysis branch resolved: {analysis_branch}")
        print(f"Schema: {analysis_branch.get('schema')}")
    except Exception as e:
        print(f"Error resolving analysis branch: {e}")

def test_llm_call_with_schema():
    """Test actual LLM call with schema"""
    print("\nTesting LLM call with schema...")
    
    try:
        grammar_branch = resolve_branch("sentence_grammar_check")
        schema = grammar_branch.get('schema')
        
        if schema:
            llm_client = LLMClient(model_key="gpt-4.1-mini")
            
            system = grammar_branch['system']
            user_prompt = grammar_branch['user_template'].format(
                survey_context="편의점 만족도 조사",
                answer="맛이좋아요"
            )
            
            response, log = llm_client.chat(
                system=system,
                user=user_prompt,
                schema=schema
            )
            
            print(f"Response type: {type(response)}")
            print(f"Response: {response}")
            print(f"Cost: {log.cost_usd}")
            
        else:
            print("No schema found")
            
    except Exception as e:
        print(f"Error in LLM call: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_schema_resolution()
    test_llm_call_with_schema()