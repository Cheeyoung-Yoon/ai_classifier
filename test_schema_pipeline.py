#!/usr/bin/env python3
"""
Test the complete schema-based pipeline
- Test schema resolution from prompt config
- Test LLM calls with schema-based structured output
- Validate data extraction and processing
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.prompt.prompt_loader import resolve_branch
from io_layer.llm.client import LLMClient  
from config.schemas import SCHEMA_REGISTRY, GrammarCorrectionSchema, SentenceAnalysisSchema
from utils.pydantic_utils import extract_llm_response_data


async def test_schema_resolution():
    """Test schema resolution from branch names"""
    print("=== Testing Schema Resolution ===")
    
    # Test different branch types
    test_branches = [
        "sentence_grammar_check",
        "sentence_only", 
        "survey_parser_4.1",
        "question_data_matcher"
    ]
    
    for branch in test_branches:
        try:
            resolved = resolve_branch(branch)
            print(f"✓ Branch '{branch}': Schema = {resolved['schema'].__name__ if resolved.get('schema') else 'None'}")
        except Exception as e:
            print(f"✗ Branch '{branch}': Error = {e}")
    
    print()


async def test_llm_with_schema():
    """Test LLM calls with schema-based structured output"""
    print("=== Testing LLM with Schema ===")
    
    # Initialize LLM client
    llm_client = LLMClient(model_key="gpt-4.1-mini")
    
    # Test 1: Grammar correction
    print("Testing Grammar Correction...")
    resolved = resolve_branch("sentence_grammar_check")
    
    test_sentence = "이 문장은 테스트용 문장 입니다."
    response = await llm_client.acall(
        messages=[
            {"role": "system", "content": resolved["system"]},
            {"role": "user", "content": resolved["user_template"].format(sentence=test_sentence)}
        ],
        schema=resolved.get("schema")
    )
    
    print(f"Raw response type: {type(response)}")
    print(f"Response keys: {list(response.keys())}")
    
    # Extract data using utility
    data = extract_llm_response_data(response)
    print(f"Extracted data type: {type(data)}")
    print(f"Grammar correction result: {data}")
    print()
    
    # Test 2: Sentence analysis
    print("Testing Sentence Analysis...")
    resolved = resolve_branch("sentence_only")
    
    response = await llm_client.acall(
        messages=[
            {"role": "system", "content": resolved["system"]},
            {"role": "user", "content": resolved["user_template"].format(sentence=test_sentence)}
        ],
        schema=resolved.get("schema")
    )
    
    data = extract_llm_response_data(response)
    print(f"Sentence analysis result: {data}")
    print()


async def test_data_extraction():
    """Test data extraction patterns"""
    print("=== Testing Data Extraction Patterns ===")
    
    # Simulate LLM response structure
    mock_response = {
        "raw": "Raw LLM output text...",
        "parsed": {
            "corrected_sentence": "이 문장은 테스트용 문장입니다.",
            "corrections": [
                {"original": "입니다", "corrected": "입니다", "reason": "spacing"}
            ]
        }
    }
    
    # Test extraction
    data = extract_llm_response_data(mock_response)
    print(f"Extracted data: {data}")
    
    # Test direct access patterns
    corrected = data.get('corrected_sentence', '')
    corrections = data.get('corrections', [])
    
    print(f"Corrected sentence: {corrected}")
    print(f"Number of corrections: {len(corrections)}")
    print()


async def test_schema_registry():
    """Test schema registry functionality"""
    print("=== Testing Schema Registry ===")
    
    print(f"Available schemas: {list(SCHEMA_REGISTRY.keys())}")
    
    for name, schema_cls in SCHEMA_REGISTRY.items():
        print(f"✓ {name}: {schema_cls.__name__}")
        
        # Try to create instance if possible
        if hasattr(schema_cls, 'model_fields'):
            fields = list(schema_cls.model_fields.keys())
            print(f"  Fields: {fields}")
    
    print()


async def main():
    """Run all tests"""
    print("🚀 Starting Schema-based Pipeline Tests\n")
    
    try:
        await test_schema_resolution()
        await test_schema_registry() 
        await test_data_extraction()
        await test_llm_with_schema()
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())