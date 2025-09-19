# Trail 3 - Clean Stage 3 MCL Classification

Clean implementation of Stage 3 MCL clustering with just the 6 essential components.

## 📁 Structure

```
trail3/
├── mcl_pipeline.py     # 1. Clean MCL clustering algorithm
├── router.py          # 2. Simple mode validation router  
├── classification.py  # 3. Main classification node (all modes)
├── config.py          # 4. Simple configuration management
├── data_loader.py     # 5. State-based data loading from LangGraph
└── README.md          # 6. Documentation
```

## 🎯 Features

### 1. MCL Pipeline (`mcl_pipeline.py`)
- Clean MCL clustering implementation
- Functions: `estimate_clusters()`, `auto_train_mcl()`, `manual_train_mcl()`
- No external dependencies beyond numpy/sklearn

### 2. Router (`router.py`)
- Simple mode validation: `estimate`, `auto_train`, `manual_train`
- Basic parameter checking
- Returns routing decision in state

### 3. Classification Node (`classification.py`)
- Single main function: `stage3_classify()`
- Handles all three modes internally
- Clean state input/output

### 4. Configuration (`config.py`)
- Default parameters with validation
- State-based parameter merging
- Range validation for all parameters

### 5. Data Loader (`data_loader.py`)
- Reads data directly from LangGraph state (proper pattern)
- Supports both single and long format embeddings
- **ID Tracking**: Maintains unique IDs for each embedding (`respondent_id_column_name`)
- **Mapping Function**: `map_clusters_back_to_data()` maps cluster results back to original respondents
- Robust embedding parsing from various formats

### 6. Evaluation Utilities (`evaluation.py`)
- **NMI/ARI Scoring**: Normalized Mutual Information and Adjusted Rand Index
- **Cluster Quality Assessment**: Comprehensive evaluation metrics
- **Training Optimization**: Used in auto_train mode for parameter selection
- **Singleton Handling**: Robust evaluation with singleton cluster management

## 🚀 Usage

```python
# Basic usage in LangGraph
from trail3.router import stage3_router
from trail3.classification import stage3_classify

# State with data and optional evaluation
state = {
    "matched_questions": {
        "dataframe": [
            {
                "question": "연비는 어떠신가요?",
                "answer": "SUV 치고는 연비가 괜찮은 편입니다",
                "question_embed": [0.1, 0.2, ...],  # embedding array
                "answer_embed": [0.3, 0.4, ...]     # embedding array
            }
        ]
    },
    "stage3_mode": "auto_train",  # Uses evaluation for optimization
    "stage3_search_iterations": 20
}

# Route and classify with evaluation
routed_state = stage3_router(state)
result = stage3_classify(routed_state)

# Check evaluation results
if "stage3_best_evaluation" in result:
    eval_scores = result["stage3_best_evaluation"]
    print(f"NMI: {eval_scores.get('nmi', 'N/A'):.3f}")
    print(f"ARI: {eval_scores.get('ari', 'N/A'):.3f}")
```

## 📊 Evaluation Features

### NMI/ARI Integration
When true labels are available (e.g., from manual coding), the auto_train mode automatically:
- Computes **NMI (Normalized Mutual Information)** - measures information overlap
- Computes **ARI (Adjusted Rand Index)** - measures clustering similarity  
- Uses **composite score** (NMI + ARI) / 2 for parameter optimization
- Handles singleton clusters robustly
- Provides detailed evaluation metrics in results

### Evaluation Output
```python
# Example evaluation results in state
{
    "stage3_best_evaluation": {
        "nmi": 0.823,           # Normalized Mutual Information
        "ari": 0.756,           # Adjusted Rand Index  
        "nmi_adjusted": 0.798,  # Adjusted NMI
        "singleton_ratio": 0.12, # Proportion of singleton clusters
        "n_clusters_true": 15,   # True number of clusters
        "n_clusters_pred": 18    # Predicted number of clusters
    }
}
```
from trail3.classification import stage3_classify

# State with data
state = {
    "matched_questions": {
        "dataframe": [
            {
                "question": "연비는 어떠신가요?",
                "answer": "SUV 치고는 연비가 괜찮은 편입니다",
                "question_embed": [0.1, 0.2, ...],  # embedding array
                "answer_embed": [0.3, 0.4, ...]     # embedding array
            }
        ]
    },
    "stage3_mode": "estimate"  # or "auto_train" or "manual_train"
}

# Route and classify
routed_state = stage3_router(state)
result = stage3_classify(routed_state)
```

## 📊 Production Ready

**Core Components:**
- Input: Survey data with multiple embedding columns per respondent
- **ID Tracking**: Each embedding gets unique ID (e.g., `user_001_question_embed`)
- Processing: Fast MCL clustering with full traceability
- Output: Cluster labels + detailed mapping back to original respondents

## 🗺️ ID Mapping Example

```python
# Input: Survey responses with IDs
{
    "respondent_id": "user_001",
    "question": "연비는 어떠신가요?",
    "answer": "SUV 치고는 연비가 괜찮은 편입니다",
    "question_embed": [0.1, 0.2, ...],
    "answer_embed": [0.3, 0.4, ...]
}

# Output: Detailed cluster mapping
{
    "respondent_clusters": {
        "user_001": {
            "respondent_data": {...},  # Original survey data
            "embedding_clusters": {
                "question_embed": 5,   # Question clustered with group 5
                "answer_embed": 12     # Answer clustered with group 12  
            },
            "question_id": "fuel_economy"
        }
    },
    "embedding_to_cluster": {
        "user_001_question_embed": 5,
        "user_001_answer_embed": 12
    }
}
```

## 🧹 Clean Design Principles

1. **Single Responsibility**: Each file has one clear purpose
2. **State-Based**: Proper LangGraph pattern using state, not file paths
3. **Error Handling**: Comprehensive validation and error messages
4. **Simplicity**: Minimal complexity, maximum clarity
5. **Modularity**: Each component works independently
6. **Testability**: Easy to test and verify functionality

## 🎯 Ready For

- Real data integration from your Stage 2 results
- Production deployment in LangGraph pipelines  
- Extension with additional clustering algorithms
- Integration with your Korean survey classification system

Trail 3 is clean, production-ready, and focused on core functionality! 🚀