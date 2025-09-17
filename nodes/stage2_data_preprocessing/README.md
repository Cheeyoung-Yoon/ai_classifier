# Stage 2: Data Preprocessing

## Purpose
Stage 2 handles LLM-based data preprocessing with intelligent routing and content processing.

## Key Characteristics
- **LLM-Dependent**: Uses LLM models for text processing and analysis
- **Router-Based**: Intelligent routing based on question types and content
- **Content Processing**: Text cleaning, normalization, and preparation for classification

## Planned Components

### Router System
- Question type detection
- Content classification routing
- LLM model selection

### Preprocessing Nodes
- Text normalization
- Content cleaning
- Data validation
- Format standardization

## Current Files to Migrate (when ready)
Based on current `nodes/classifications/` structure, these files likely belong in stage2:
- Text preprocessing utilities
- Content classification logic
- LLM routing logic

## Dependencies
- **Input**: Clean data from Stage 1 (`question_data_match`, `open_columns`)
- **Output**: Preprocessed data ready for Stage 3 classification
- **External**: LLM router, text processing utilities

## Notes for Future Development
- Consider async processing for LLM calls
- Implement caching for repeated preprocessing operations
- Add validation checkpoints between preprocessing steps
- Design for extensibility with new preprocessing methods