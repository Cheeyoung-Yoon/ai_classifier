"""Utility script to run the Stage 1 & 2 pipeline and capture state snapshots per node.

This script compiles the LangGraph workflow, executes it on the bundled test data,
and writes the state returned after each node finishes into timestamped JSON files.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph.graph import create_workflow
from graph.state import initialize_project_state


PROJECT_NAME = "test"
SURVEY_FILENAME = "test.txt"
DATA_FILENAME = "-SUV_776부.xlsx"


def _serialize(value: Any) -> Any:
    """Best-effort conversion of LangGraph state values to JSON serialisable types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(v) for v in value]
    return str(value)


def _normalise_event(event: Any) -> Dict[str, Any]:
    """Normalise langgraph stream events into a dict with at least 'event' and 'state'."""
    if isinstance(event, dict):
        return event
    if isinstance(event, tuple) and len(event) == 2:
        event_type, payload = event
        if isinstance(payload, dict):
            payload = payload.copy()
            payload.setdefault("event", event_type)
            return payload
        return {"event": event_type, "payload": payload}
    return {"event": "unknown", "payload": event}


def stream_pipeline_states(base_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """Run the workflow via stream() and dump node states into JSON files."""
    workflow = create_workflow()
    app = workflow.compile()

    initial_state = initialize_project_state(
        project_name=PROJECT_NAME,
        survey_filename=SURVEY_FILENAME,
        data_filename=DATA_FILENAME,
        base_dir=str(base_dir),
        use_raw_dir=True,
    )

    log_dir.mkdir(parents=True, exist_ok=True)

    final_state: Dict[str, Any] | None = None
    visited_nodes: list[str] = []

    for index, event in enumerate(app.stream(initial_state, config={"recursion_limit": 200})):
        normalised = _normalise_event(event)
        event_type = normalised.get("event")
        if event_type != "on_node_end":
            # Only persist states after nodes finish to avoid partial updates.
            continue

        node_name = normalised.get("name", f"unknown_{index}")
        visited_nodes.append(str(node_name))
        state_snapshot = normalised.get("state", {})

        snapshot_path = log_dir / f"{index:02d}_{node_name}.json"
        with snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(_serialize(state_snapshot), f, ensure_ascii=False, indent=2)

        if node_name in {"__end__", "END"}:
            final_state = state_snapshot

    if final_state is None:
        raise RuntimeError("Pipeline stream did not yield a final END node state.")

    return {
        "final_state": final_state,
        "visited_nodes": visited_nodes,
    }


def ensure_stage2_outputs(final_state: Dict[str, Any]) -> None:
    """Validate that Stage 2 produced CSV outputs for every processed question."""
    matched_questions = final_state.get("matched_questions", {})
    if not isinstance(matched_questions, dict):
        raise RuntimeError("Final state does not contain matched_questions as a dict.")

    missing_outputs = []
    for question_id, payload in matched_questions.items():
        stage2_info = payload.get("stage2_data", {}) if isinstance(payload, dict) else {}
        csv_path = stage2_info.get("csv_path")
        if not csv_path or not Path(csv_path).exists():
            missing_outputs.append({
                "question_id": question_id,
                "csv_path": csv_path,
            })

    if missing_outputs:
        details = ", ".join(
            f"{item['question_id']} -> {item['csv_path']}" for item in missing_outputs
        )
        raise FileNotFoundError(
            f"Stage 2 outputs missing for: {details}. Run pipeline manually to inspect."
        )


def main() -> None:
    logs_root = Path(__file__).parent / "logs" / "pipeline_state"
    run_dir = logs_root / datetime.now().strftime("%Y%m%d_%H%M%S")

    result = stream_pipeline_states(base_dir=REPO_ROOT, log_dir=run_dir)

    final_state = result["final_state"]
    ensure_stage2_outputs(final_state)

    summary_path = run_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "project": PROJECT_NAME,
                "survey_file": SURVEY_FILENAME,
                "data_file": DATA_FILENAME,
                "visited_nodes": result["visited_nodes"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"State snapshots saved under: {run_dir}")


if __name__ == "__main__":
    main()
