# Stage3 Integration TODO

## Stage Contracts
- Confirm Stage 2 → Stage 3 hand-off uses `state['matched_questions'][qid]['stage2_data']['csv_path']` values and document required columns (embeddings, labels if present).
- Extend `2.langgraph/graph/state.py` with Stage 3 fields: `stage3_mode`, `stage3_status`, `stage3_metrics`, `stage3_result_path`, `stage3_config_path`, `stage3_validation_path`, etc.

## Utilities & Data Prep
- Implement `2.langgraph/nodes/stage3_classification/data_loader.py` to discover Stage 2 CSV outputs via `ProjectDirectoryManager`, load embeddings/text, and assemble a reusable `Stage3Dataset` with optional gold labels.
- Extract KNN/CSLS graph helpers from `0.refine/auto_mcl_pipe.py` into `2.langgraph/nodes/stage3_classification/graph_builder.py` so both train/estimate reuse consistent clustering primitives.

## Mode Routing & Project Layout
- Add `stage3_mode_router` node selecting between train/estimate based on `state['stage3_mode']` (default `estimate`).
- Update `utils/project_manager.py` to create/access `temp_data/stage3_results` and `stage3_models`; add getters for saved configs, validation sets, and outputs.

## Train Mode Node
- Implement `stage3_train_node` that loads the dataset, constructs validation labels (persist as `stage3_models/validation.parquet`), defines hyperparameter search (random/grid), executes clustering runs using shared executor, scores with ARI/NMI/Adj-NMI, and logs per-trial diagnostics.
- Persist best config to `stage3_models/best_config.json`, save cluster assignments CSV, update state with best metrics and status.

## Estimate Mode Node
- Implement `stage3_estimate_node` that loads dataset plus best config, runs a single clustering pass, computes summary stats/optional unsupervised proxies, writes outputs under `temp_data/stage3_results`, and updates state.
- Handle missing config gracefully (fallback to defaults or error with guidance).

## Graph Integration
- Insert Stage 3 nodes after Stage 2 completion in `2.langgraph/graph/graph.py`: `stage3_start` tracker → `stage3_mode_router` → `stage3_train_node` / `stage3_estimate_node` → `stage3_completion` → `END`.
- Update `stage2_next_question.stage2_completion_router` and stage tracker utilities to recognize new Stage 3 checkpoints/history entries.

## Configuration & Documentation
- Create `2.langgraph/config/stage3_config.py` with defaults (search ranges, output locations, mode default) and expose via main entrypoints.
- Expand `2.langgraph/nodes/stage3_classification/README.md` detailing workflow, inputs/outputs, validation generation, and tuning/estimate usage.

## Validation & Testing
- Add tests in `2.langgraph/tests/stage3/` for dataset loader, train-mode scoring logic, and estimate-mode execution (mock embeddings for determinism).
- Draft manual validation checklist: run train on sample data, inspect saved config/metrics, rerun estimate with persisted config.
