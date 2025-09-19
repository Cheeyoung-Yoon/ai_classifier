#!/usr/bin/env python3
"""
Direct Phase 2 test to debug configuration issues.
"""

import sys
import os
sys.path.append('/home/cyyoon/test_area/ai_text_classification/2.langgraph')

import numpy as np
from nodes.stage3_classification.phase2_secondary_labeling import Phase2SecondaryLabeling

def test_phase2_direct():
    """Test Phase 2 directly with proper configuration."""
    
    print("🔧 Direct Phase 2 Configuration Test")
    print("=" * 40)
    
    # Create test configuration
    test_config = {
        # Graph construction
        "edge_weights": {
            "alpha": 0.5,  # embedding similarity weight
            "beta": 0.3,   # co-context weight  
            "gamma": 0.2   # keyword overlap weight
        },
        "top_m_edges": 10,  # Keep top-m edges
        "edge_threshold": 0.1,  # Remove edges below threshold
        
        # Community detection
        "algorithm": "louvain",  # louvain/leiden
        "resolution": 1.0,  # Resolution parameter for cluster count control
        "random_seed": 42,
        "n_iterations": 10,  # For leiden
        
        # Constraints
        "must_links": [],  # [(group_id1, group_id2), ...]
        "cannot_links": [],  # [(group_id1, group_id2), ...]
        
        # Labeling mode
        "labeling_mode": "basic",  # human_in_loop/llm_assisted/basic
        
        # LLM configuration for label generation
        "llm_config": {
            "model": "gpt-3.5-turbo",
            "max_examples": 5,  # Max positive examples per label
            "generate_definitions": True,
            "generate_rules": True
        }
    }
    
    print(f"📋 Configuration keys: {list(test_config.keys())}")
    print(f"🔗 Edge threshold: {test_config['edge_threshold']}")
    
    # Initialize Phase 2
    try:
        phase2 = Phase2SecondaryLabeling(config=test_config)
        print(f"✅ Phase 2 initialized successfully")
        print(f"   Config edge_threshold: {phase2.config.get('edge_threshold', 'MISSING')}")
        
        # Create minimal test data
        test_groups = {
            "Q1": {
                "cluster_0": [0, 1, 2],
                "cluster_1": [3, 4, 5]
            }
        }
        
        test_prototypes = {
            "Q1": {
                "cluster_0": {
                    "text": "sample text 0",
                    "embedding": np.random.randn(64).tolist(),
                    "cluster_size": 3
                },
                "cluster_1": {
                    "text": "sample text 1", 
                    "embedding": np.random.randn(64).tolist(),
                    "cluster_size": 3
                }
            }
        }
        
        test_metadata = {
            "Q1": {
                "cluster_0": {
                    "keywords": ["word1", "word2"],
                    "avg_similarity": 0.8,
                    "member_indices": [0, 1, 2],
                    "texts": ["text0", "text1", "text2"]
                },
                "cluster_1": {
                    "keywords": ["word3", "word4"],
                    "avg_similarity": 0.7,
                    "member_indices": [3, 4, 5],
                    "texts": ["text3", "text4", "text5"]
                }
            }
        }
        
        print(f"\n🧪 Testing Phase 2 processing...")
        
        # Test processing
        results = phase2.process_phase1_groups(
            phase1_groups=test_groups,
            phase1_prototypes=test_prototypes, 
            phase1_metadata=test_metadata
        )
        
        print(f"✅ Phase 2 processing completed!")
        print(f"   Status: {results.get('status', 'unknown')}")
        print(f"   Labels: {results.get('n_labels', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 2 test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase2_direct()
    if success:
        print(f"\n🎉 Direct Phase 2 test passed!")
    else:
        print(f"\n💥 Direct Phase 2 test failed!")
        sys.exit(1)