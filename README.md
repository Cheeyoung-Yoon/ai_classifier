# AI Text Classification Pipeline

LangGraph workflow that ingests survey materials, aligns open-ended questions with response columns, and runs Stage 2 preprocessing (sentence/word handling, embeddings, CSV export). Designed for large Korean survey datasets.

---

## Directory Snapshot

```
2.langgraph/
├── graph/
│   ├── graph.py                # Pipeline wiring
│   └── state.py                # State schema & initializer
├── nodes/
│   ├── stage1_data_preparation/
│   │   ├── survey_loader.py
│   │   ├── data_loader.py
│   │   ├── survey_parser.py
│   │   ├── survey_context.py
│   │   ├── column_extractor.py
│   │   └── question_matcher.py
│   ├── stage2_data_preprocessing/
│   │   ├── stage2_main.py
│   │   ├── stage2_sentence_node.py
│   │   ├── stage2_word_node.py
│   │   ├── stage2_etc_node.py
│   │   └── prep_sentence.py
│   ├── stage2_next_question.py
│   └── shared/stage_tracker.py
├── router/stage2_router.py
├── io_layer/
│   ├── llm/client.py
│   └── embedding/embedding.py
├── config/prompt/prompt.config.yaml
└── tests/stage1_and_stage2_full_test.py
```

Auxiliary scripts for refinement (`0.refine/`) and ideas (`EDA/`, `train/`, …) sit alongside this directory but are not part of the main pipeline.

---

## Quick Start

```bash
cd 2.langgraph
python tests/stage1_and_stage2_full_test.py
```

Requirements:

- `OPENAI_API_KEY` configured (for `LLMClient` in `io_layer/llm/client.py`).
- Network access to OpenAI models (`gpt-4.1`, `gpt-4.1-nano`, etc.).
- `sentence_transformers` with `jhgan/ko-sroberta-multitask` available (download on first use).

The test triggers the full Stage 1 + Stage 2 flow for the `test` project and writes Stage 2 CSVs under `data/test/temp_data/stage2_results/`.

---

## Pipeline Flow

### Stage 1 – Data Preparation

1. **Pipeline Initialization** (`graph/graph.py` → `pipeline_initialization_node`)
   - `utils/project_manager.py` creates per-project directories (`data/<project>/raw`, `temp_data`, etc.).
   - Stage history manager set up (`utils.stage_history_manager` via `stage_tracker`).

2. **Survey & Data Loading**
   - `nodes/stage1_data_preparation/survey_loader.py`: loads raw survey using `FileLoader`.
   - `nodes/stage1_data_preparation/data_loader.py`: loads Excel, stores CSV path in `raw_dataframe_path`.

3. **Survey Parsing**
   - `survey_parser.py`: calls `tools.llm_router` with prompt `survey_parser_4.1` (see `config/prompt/prompt.config.yaml`) to extract open-ended questions into structured `parsed_survey`.

4. **Survey Context Summary**
   - `survey_context.py`: obtains overall context via `survey_context_summarizer`.

5. **Open Column Detection**
   - `column_extractor.py`: finds text columns using metadata or `DataHelper`.

6. **Question ↔ Column Matching**
   - `question_matcher.py`: routes to `question_data_matcher` prompt, fallback logic for failures.

7. **Data Integration & Memory Flush**
   - `nodes/survey_data_integrate` standardizes matches into `matched_questions`.
   - `stage1_memory_flush_node`, `memory_status_check_node`: drop heavy objects, log memory status.

Stage trackers in `nodes/shared/stage_tracker.py` persist summaries and cost metrics after each major step.

### Stage 2 – Data Preprocessing

Loop over each question discovered in Stage 1:

1. **Initialization** (`stage2_data_preprocessing/stage2_main.py`)
   - Sets `current_question_id`, `current_question_type`, `current_question_index`.

2. **Routing** (`router/stage2_router.py`)
   - Uses `QTYPE_MAP` to branch into `WORD`, `SENTENCE`, or `ETC`; exits when `stage2_processing_complete` is flagged.

3. **Sentence Questions** (`stage2_sentence_node.py`)
   - Resolves prompts (`sentence_grammar_check`, `sentence_<type>_split`, fallback to `sentence_only`).
   - Dual LLM flow: grammar correction (`LLMClient` with `gpt-4.1`) and analysis (`gpt-4.1-nano`).
   - Support for dependent questions via `prep_sentence.extract_question_choices`.
   - Parallel processing with `ThreadPoolExecutor`; tracks per-row USD cost.
   - Embeddings via `io_layer/embedding/embedding.py` for original/corrected text and atomic sentences.
   - Outputs CSV (`project_manager.get_stage2_csv_path`) with sentiment, matches, S/V/C keywords, embedding vectors.
   - Updates `matched_questions[question_id]['stage2_data']` and, if enabled, saves state log.

4. **Word Questions** (`stage2_word_node.py`)
   - Extracts raw text columns, generates embeddings, writes CSV identical naming scheme, updates metadata.

5. **ETC Questions** (`stage2_etc_node.py`)
   - Marks processing as complete without transformation (placeholder for future rules).

6. **Advance to Next Question** (`stage2_next_question.py`)
   - Moves through `matched_questions`, looping via router until all questions processed.

---

## Configuration

- **Prompts** (`config/prompt/prompt.config.yaml`): central registry for LLM prompts, including Stage 2 grammar and analysis branches.
- **LLM Models** (`config/llm/config_llm.py` via `LLMClient`): define model names, costs, invocation parameters.
- **Runtime Settings** (`config/config.py`): toggles like `SAVE_STATE_LOG`, base directories, API keys.

Update the YAML branches to tune instructions without touching node code.

---

## Outputs

- **Stage 2 CSVs**: `data/<project>/temp_data/stage2_results/stage2_<question_id>_<type>_<timestamp>.csv`
- **State & History**:
  - Latest pipeline state: `data/<project>/state.json`
  - Full history snapshots: `data/<project>/state_history/*state.json`
- **Logs**: Console prints per-node progress, cost, and file destinations.

Each CSV row includes ID, original/corrected responses, embeddings, atomic sentences, sentiment fields, and S/V/C keyword lists.

---

## Testing & Validation

- `tests/stage1_and_stage2_full_test.py`: end-to-end sanity check with debug output and CSV verification.
- For CI or offline runs, mock `LLMClient` and `VectorEmbedding` to bypass external calls.

---

## Troubleshooting

- **Missing prompts**: `stage2_sentence_node` prints fallbacks; ensure matching branch names in `prompt.config.yaml`.
- **No CSV outputs**: verify `matched_questions` includes question types and `project_manager` directories exist.
- **Embedding errors**: ensure `sentence_transformers` model downloads successfully and GPU/CPU environment supports it.
- **Cost tracking**: `total_llm_cost_usd` accumulates per question; inspect logs for per-row costs.

---

## Next Steps

- Add Stage 3 modeling (clustering, topic modeling) using generated embeddings.
- Extend router to handle new question types (`etc_pos_neg`, etc.).
- Introduce automated regression tests using mocked LLM responses to keep test runs deterministic.
