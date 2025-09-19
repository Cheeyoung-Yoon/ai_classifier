"""
Test Trail 3 data loader with real state structure.
Uses the actual JSON state format from user's data.
"""
import sys
from pathlib import Path
import json

# Add trail3 to path
trail3_path = Path(__file__).parent
sys.path.insert(0, str(trail3_path))

from data_loader import load_data_from_state


def test_real_state_structure():
    """Test data loader with the real state structure."""
    
    print("🧪 Testing Real State Structure")
    print("=" * 50)
    
    # Load the actual state file
    state_file = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/state_history/20250918_113231_081_STAGE2_WORD_문23_COMPLETED_state.json"
    
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        print(f"✅ Loaded state from: {Path(state_file).name}")
        
        # Check matched_questions structure
        matched_questions = state.get("matched_questions", {})
        print(f"📊 Found {len(matched_questions)} questions in state")
        
        # Show available questions
        for qid, qdata in matched_questions.items():
            stage2_data = qdata.get("stage2_data", {})
            status = stage2_data.get("status", "no_data")
            csv_path = stage2_data.get("csv_path", "")
            
            if csv_path:
                csv_name = Path(csv_path).name
                print(f"   {qid}: {status} -> {csv_name}")
            else:
                print(f"   {qid}: {status} (no CSV)")
        
        # Test the data loader
        print(f"\n🎯 Testing data loader...")
        
        embeddings, metadata = load_data_from_state(state)
        
        print(f"\n✅ Data loading successful!")
        print(f"   📊 Total embeddings: {embeddings.shape[0]}")
        print(f"   📏 Embedding dimensions: {embeddings.shape[1]}")
        print(f"   🔍 Format: {metadata.get('format')}")
        print(f"   📋 Original rows: {metadata.get('original_rows')}")
        print(f"   🏷️  Embedding columns: {metadata.get('embedding_columns')}")
        
        # Show question breakdown
        if 'original_dataframe' in metadata:
            df = metadata['original_dataframe']
            question_counts = df['question_id'].value_counts()
            print(f"\n📈 Question breakdown:")
            for qid, count in question_counts.items():
                print(f"      {qid}: {count} rows")
        
        print(f"\n🎉 Real state structure test passed!")
        
    except FileNotFoundError:
        print(f"❌ State file not found: {state_file}")
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_real_state_structure()