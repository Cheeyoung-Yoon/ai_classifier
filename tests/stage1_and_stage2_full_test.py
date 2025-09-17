#!/usr/bin/env python3
"""
Stage1 + Stage2 Full Pipeline Test with detailed debugging
"""
import sys
import os

# Add project root to path
project_root = "/home/cyyoon/test_area/ai_text_classification/2.langgraph"
sys.path.insert(0, project_root)

from graph.graph import run_pipeline
import pandas as pd
from datetime import datetime   
from typing import Dict, Any

import sys
import os

import warnings
warnings.filterwarnings('ignore')

from graph.graph import run_pipeline
from datetime import datetime

# Restore stderr after imports

def print_debug_separator(title: str):
    """디버깅 섹션 구분자 출력"""
    print(f"\n{'='*70}")
    print(f"🔍 {title}")
    print(f"{'='*70}")

def test_full_pipeline():
    """Test complete pipeline with Stage 1 and Stage 2"""
    print_debug_separator("Full Pipeline Test (Stage 1 + Stage 2)")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚀 Testing Full Pipeline (Stage 1 + Stage 2)")
    print("=" * 50)
    
    try:
        print("📂 Target files:")
        print("   - Survey: test.txt")
        print("   - Data: -SUV_776부.xlsx") 
        print("   - Project: test")
        
        start_time = datetime.now()
        
        # Run the complete pipeline
        result = run_pipeline(
            project_name="test",
            survey_filename="test.txt",
            data_filename="-SUV_776부.xlsx"
        )
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print_debug_separator("Pipeline Results")
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        
        print("\n📋 Pipeline Result Summary:")
        print("-" * 30)
        
        # Check essential results
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            return False
            
        print(f"✅ Project: {result.get('project_name', 'Unknown')}")
        print(f"✅ Current Stage: {result.get('current_stage', 'Unknown')}")
        print(f"✅ Pipeline ID: {result.get('pipeline_id', 'Unknown')}")
        print(f"✅ LLM Cost: ${result.get('total_llm_cost_usd', 0):.4f}")
        
        # Check Stage 1 results
        if result.get("matched_questions"):
            print(f"✅ Stage 1: Question matching completed")
        else:
            print(f"⚠️ Stage 1: No question matches found")
            
        # Check Stage 2 results
        if result.get("question_type"):
            print(f"✅ Stage 2: Question type identified as {result['question_type']}")
        else:
            print(f"⚠️ Stage 2: Question type not determined")
            
        if result.get("stage2_csv_output_path"):
            print(f"✅ Stage 2: CSV output saved to {result['stage2_csv_output_path']}")
        else:
            print(f"⚠️ Stage 2: No CSV output generated")
            
        print("\n🎉 Full pipeline test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    exit(0 if success else 1)