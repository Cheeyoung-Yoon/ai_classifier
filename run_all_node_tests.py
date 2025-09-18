#!/usr/bin/env python3
"""
Master Test Runner - All Node Unit Tests
모든 노드 유닛테스트를 실행하는 마스터 러너
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_test_file(test_file_path: str, test_name: str):
    """Run a single test file and return results"""
    print(f"\n🎯 Running {test_name}")
    print("=" * 60)
    
    try:
        # Import and run the test
        if test_file_path == 'test_comprehensive_nodes.py':
            from test_comprehensive_nodes import run_all_tests
            success = run_all_tests()
        elif test_file_path == 'test_stage1_nodes_detailed.py':
            from test_stage1_nodes_detailed import run_stage1_tests
            run_stage1_tests()
            success = True
        elif test_file_path == 'test_stage2_nodes_detailed.py':
            from test_stage2_nodes_detailed import run_stage2_tests
            run_stage2_tests()
            success = True
        else:
            print(f"❌ Unknown test file: {test_file_path}")
            return False
            
        return success
        
    except Exception as e:
        print(f"❌ Test suite {test_name} failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_individual_node_tests():
    """Run quick individual node tests for immediate feedback"""
    print("🔍 Quick Individual Node Tests")
    print("-" * 40)
    
    # Quick smoke tests for each node type
    individual_tests = [
        ("Survey Loader", test_survey_loader_quick),
        ("Data Loader", test_data_loader_quick),
        ("Stage2 Main", test_stage2_main_quick),
        ("Stage2 Router", test_stage2_router_quick)
    ]
    
    results = {}
    for test_name, test_func in individual_tests:
        try:
            test_func()
            results[test_name] = "✅ PASS"
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            results[test_name] = f"❌ FAIL: {e}"
            print(f"❌ {test_name}: FAILED - {e}")
    
    return results


def test_survey_loader_quick():
    """Quick test for survey loader"""
    from nodes.stage1_data_preparation.survey_loader import load_survey_node
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Q1. Test question?\n① Yes ② No")
        temp_path = f.name
    
    try:
        state = {'survey_file_path': temp_path}
        result = load_survey_node(state)
        assert 'survey_raw_content' in result
        assert 'Q1' in result['survey_raw_content']
    finally:
        os.unlink(temp_path)


def test_data_loader_quick():
    """Quick test for data loader"""
    from nodes.stage1_data_preparation.data_loader import load_data_node
    import pandas as pd
    import tempfile
    
    df = pd.DataFrame({'Q1': ['A', 'B'], 'Q2': ['response1', 'response2']})
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        state = {'data_file_path': temp_path}
        result = load_data_node(state)
        assert 'data' in result
        assert len(result['data']) == 2
    finally:
        os.unlink(temp_path)


def test_stage2_main_quick():
    """Quick test for stage2 main"""
    from nodes.stage2_data_preprocessing.stage2_main import stage2_data_preprocessing_node
    import pandas as pd
    
    state = {
        'data': pd.DataFrame({'Q1': ['test1', 'test2']}),
        'question_processing_queue': [
            {
                'column_name': 'Q1',
                'question_number': 'Q1',
                'question_text': 'Test?',
                'question_type': 'SENTENCE'
            }
        ],
        'current_question_idx': 0
    }
    
    result = stage2_data_preprocessing_node(state)
    assert 'current_question' in result
    assert 'current_data_sample' in result


def test_stage2_router_quick():
    """Quick test for stage2 router"""
    from router.stage2_router import stage2_type_router
    
    # Test different routing scenarios
    assert stage2_type_router({'current_question': {'question_type': 'WORD'}}) == 'WORD'
    assert stage2_type_router({'current_question': {'question_type': 'SENTENCE'}}) == 'SENTENCE'
    assert stage2_type_router({'current_question': {'question_type': 'ETC'}}) == 'ETC'
    assert stage2_type_router({'current_question': None}) == '__END__'


def main():
    """Main test runner"""
    print("🚀 COMPREHENSIVE NODE TESTING SUITE")
    print("=" * 80)
    print("Testing all graph nodes and pipeline components individually")
    print("=" * 80)
    
    start_time = time.time()
    
    # Test suites to run
    test_suites = [
        ("test_comprehensive_nodes.py", "Comprehensive Node Tests"),
        ("test_stage1_nodes_detailed.py", "Stage 1 Detailed Tests"),
        ("test_stage2_nodes_detailed.py", "Stage 2 Detailed Tests")
    ]
    
    # Run quick individual tests first
    print("\n🔥 Phase 1: Quick Smoke Tests")
    quick_results = run_individual_node_tests()
    
    # Run comprehensive test suites
    print("\n🔥 Phase 2: Comprehensive Test Suites")
    suite_results = {}
    
    for test_file, test_name in test_suites:
        test_path = project_root / test_file
        if test_path.exists():
            success = run_test_file(test_file, test_name)
            suite_results[test_name] = "✅ PASS" if success else "❌ FAIL"
        else:
            print(f"⚠️  Test file not found: {test_file}")
            suite_results[test_name] = "⚠️  NOT FOUND"
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 80)
    print("🏁 FINAL TEST SUMMARY")
    print("=" * 80)
    
    print(f"⏱️  Total Runtime: {total_time:.2f} seconds")
    print()
    
    print("🔍 Quick Smoke Tests:")
    for test_name, result in quick_results.items():
        print(f"  {test_name}: {result}")
    
    print("\n📋 Comprehensive Test Suites:")
    for suite_name, result in suite_results.items():
        print(f"  {suite_name}: {result}")
    
    # Overall success assessment
    all_quick_passed = all("✅" in result for result in quick_results.values())
    all_suites_passed = all("✅" in result for result in suite_results.values())
    
    print("\n🎯 Overall Assessment:")
    if all_quick_passed and all_suites_passed:
        print("🎉 ALL TESTS PASSED! Node implementation is solid.")
        exit_code = 0
    else:
        print("⚠️  Some tests failed. Review the detailed output above.")
        exit_code = 1
    
    print("\n💡 Next Steps:")
    print("  1. Review any failed tests and fix implementation issues")
    print("  2. Add more edge case tests as needed")
    print("  3. Run integration tests with the full pipeline")
    print("  4. Monitor performance and memory usage in production")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)