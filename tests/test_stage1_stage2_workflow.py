"""Stage 1 & Stage 2 Graph-Aligned Test Runner
================================================

This module provides a test harness that mirrors the Stage 1 and Stage 2
execution flow defined in ``graph/graph.py``. It executes each node in the
same order as the LangGraph pipeline, logs start/end events, and snapshots
state after every node for inspection.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd

import importlib.machinery
import importlib.util
import types

# Ensure project root and langgraph namespace are discoverable for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_package(module_name: str, directory: Path) -> None:
    """Create a namespace package pointing to the given directory if needed."""
    if module_name in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(module_name, loader=None, is_package=True)
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(directory)]
    sys.modules[module_name] = module


def _load_module(alias: str, relative_path: str) -> None:
    """Load the module at relative_path and register it under alias."""
    if alias in sys.modules:
        return
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(alias, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)


def _bootstrap_langgraph_namespace() -> None:
    """Expose repo modules under the expected langgraph namespace."""
    _ensure_package("langgraph", PROJECT_ROOT)
    _ensure_package("langgraph.graph", PROJECT_ROOT / "graph")
    _ensure_package("langgraph.nodes", PROJECT_ROOT / "nodes")
    _ensure_package("langgraph.router", PROJECT_ROOT / "router")
    _ensure_package("langgraph.utils", PROJECT_ROOT / "utils")
    _ensure_package("langgraph.config", PROJECT_ROOT / "config")

    # Provide stub StateGraph/START/END for graph.graph import requirements
    graph_pkg = sys.modules["langgraph.graph"]
    if not hasattr(graph_pkg, "StateGraph"):
        class _StubStateGraph:
            def __init__(self, *_args, **_kwargs):
                pass

            def add_node(self, *_args, **_kwargs):
                return None

            def add_edge(self, *_args, **_kwargs):
                return None

            def add_conditional_edges(self, *_args, **_kwargs):
                return None

            def compile(self, *_args, **_kwargs):
                raise NotImplementedError(
                    "StateGraph is not available in this environment"
                )

        graph_pkg.StateGraph = _StubStateGraph
        graph_pkg.START = "__START__"
        graph_pkg.END = "__END__"

    # Load required modules into the namespace
    _load_module("langgraph.graph.state", "graph/state.py")
    _load_module("langgraph.graph.graph", "graph/graph.py")
    _load_module("langgraph.nodes.survey_data_integrate", "nodes/survey_data_integrate.py")
    _load_module("langgraph.nodes.state_flush_node", "nodes/state_flush_node.py")
    _load_module("langgraph.router.stage2_router", "router/stage2_router.py")
    _load_module("langgraph.utils.project_manager", "utils/project_manager.py")
    _load_module("langgraph.utils.stage_history_manager", "utils/stage_history_manager.py")

    # Stage 1 modules
    _load_module(
        "langgraph.nodes.stage1_data_preparation.survey_loader",
        "nodes/stage1_data_preparation/survey_loader.py",
    )
    _load_module(
        "langgraph.nodes.stage1_data_preparation.data_loader",
        "nodes/stage1_data_preparation/data_loader.py",
    )
    _load_module(
        "langgraph.nodes.stage1_data_preparation.survey_parser",
        "nodes/stage1_data_preparation/survey_parser.py",
    )
    _load_module(
        "langgraph.nodes.stage1_data_preparation.survey_context",
        "nodes/stage1_data_preparation/survey_context.py",
    )
    _load_module(
        "langgraph.nodes.stage1_data_preparation.column_extractor",
        "nodes/stage1_data_preparation/column_extractor.py",
    )
    _load_module(
        "langgraph.nodes.stage1_data_preparation.question_matcher",
        "nodes/stage1_data_preparation/question_matcher.py",
    )
    _load_module(
        "langgraph.nodes.stage1_data_preparation.memory_optimizer",
        "nodes/stage1_data_preparation/memory_optimizer.py",
    )
    _load_module("langgraph.nodes.shared.stage_tracker", "nodes/shared/stage_tracker.py")

    # Stage 2 modules
    _load_module(
        "langgraph.nodes.stage2_data_preprocessing.stage2_main",
        "nodes/stage2_data_preprocessing/stage2_main.py",
    )
    _load_module(
        "langgraph.nodes.stage2_data_preprocessing.stage2_word_node",
        "nodes/stage2_data_preprocessing/stage2_word_node.py",
    )
    _load_module(
        "langgraph.nodes.stage2_data_preprocessing.stage2_sentence_node",
        "nodes/stage2_data_preprocessing/stage2_sentence_node.py",
    )
    _load_module(
        "langgraph.nodes.stage2_data_preprocessing.stage2_etc_node",
        "nodes/stage2_data_preprocessing/stage2_etc_node.py",
    )
    _load_module("langgraph.nodes.stage2_next_question", "nodes/stage2_next_question.py")


_bootstrap_langgraph_namespace()

# Stage 1 / Stage 2 state helpers
from langgraph.graph.state import initialize_project_state
from langgraph.graph.graph import pipeline_initialization_node

# Stage 1 nodes
from langgraph.nodes.stage1_data_preparation.survey_loader import load_survey_node
from langgraph.nodes.stage1_data_preparation.data_loader import load_data_node
from langgraph.nodes.stage1_data_preparation.survey_parser import parse_survey_node
from langgraph.nodes.stage1_data_preparation.survey_context import survey_context_node
from langgraph.nodes.stage1_data_preparation.column_extractor import get_open_column_node
from langgraph.nodes.stage1_data_preparation.question_matcher import question_data_matcher_node
from langgraph.nodes.stage1_data_preparation.memory_optimizer import stage1_memory_flush_node
from langgraph.nodes.survey_data_integrate import survey_data_integrate_node

# Stage tracking helpers
from langgraph.nodes.shared.stage_tracker import (
    stage1_data_preparation_completion,
    stage1_memory_flush_completion,
    stage2_classification_start,
)

from langgraph.nodes.state_flush_node import memory_status_check_node

# Stage 2 nodes and routing
from langgraph.nodes.stage2_data_preprocessing.stage2_main import (
    stage2_data_preprocessing_node,
)
from langgraph.nodes.stage2_data_preprocessing.stage2_word_node import stage2_word_node
from langgraph.nodes.stage2_data_preprocessing.stage2_sentence_node import (
    stage2_sentence_node,
)
from langgraph.nodes.stage2_data_preprocessing.stage2_etc_node import stage2_etc_node
from langgraph.nodes.stage2_next_question import (
    stage2_next_question_node,
    stage2_completion_router,
)
from langgraph.router.stage2_router import stage2_type_router

# Project directory management
from langgraph.utils.project_manager import get_project_manager
from langgraph.utils.stage_history_manager import get_or_create_history_manager

logger = logging.getLogger("stage1_stage2_test_runner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


NodeFn = Callable[[Dict[str, Any]], Dict[str, Any]]


class Stage1Stage2GraphTestRunner:
    """Execute Stage 1 & Stage 2 nodes exactly as defined in the graph."""

    def __init__(
        self,
        project_name: str = "test",
        survey_filename: str = "test.txt",
        data_filename: str = "-SUV_776부.xlsx",
        snapshot_dirname: str = "stage1_stage2_snapshots",
    ) -> None:
        self.project_name = project_name
        self.survey_filename = survey_filename
        self.data_filename = data_filename

        # Project manager aligned with main graph configuration
        self.project_manager = get_project_manager(project_name, str(PROJECT_ROOT))
        self.project_dir = Path(self.project_manager.project_dir)

        # Snapshot directory (JSON + pickle exports per node)
        self.snapshot_dir = self.project_dir / snapshot_dirname
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Dedicated history manager (mirrors pipeline usage)
        self.history_manager = get_or_create_history_manager(project_name, project_name)

        self.node_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Run Stage 1 + Stage 2 nodes and return the final state."""
        logger.info("Starting Stage 1 & Stage 2 pipeline test")
        state = initialize_project_state(
            self.project_name, self.survey_filename, self.data_filename
        )
        self._snapshot_state(state, "00_initialized_state")

        # Stage 1 flow (matches graph.create_workflow sequence)
        stage1_nodes = [
            ("pipeline_init", pipeline_initialization_node),
            ("load_survey", load_survey_node),
            ("load_data", load_data_node),
            ("parse_survey", parse_survey_node),
            ("extract_survey_context", survey_context_node),
            ("get_open_columns", get_open_column_node),
            ("match_questions", question_data_matcher_node),
            ("survey_data_integrate", survey_data_integrate_node),
            ("stage1_completion", stage1_data_preparation_completion),
            ("stage1_memory_flush", stage1_memory_flush_node),
            ("stage1_flush_completion", stage1_memory_flush_completion),
            ("memory_status_check", memory_status_check_node),
            ("stage2_start", stage2_classification_start),
        ]

        for node_name, node_fn in stage1_nodes:
            state = self._run_node(node_name, node_fn, state)

        # Stage 2 loop
        state = self._run_node("stage2_main", stage2_data_preprocessing_node, state)

        while True:
            branch = stage2_type_router(state)
            logger.info("Stage2 router decision: %s", branch)

            if branch == "__END__":
                logger.info("Stage2 router signaled completion")
                break

            node_mapping = {
                "WORD": ("stage2_word", stage2_word_node),
                "SENTENCE": ("stage2_sentence", stage2_sentence_node),
                "ETC": ("stage2_etc", stage2_etc_node),
            }

            if branch not in node_mapping:
                logger.error("Unknown Stage2 branch: %s", branch)
                break

            node_name, node_fn = node_mapping[branch]
            state = self._run_node(node_name, node_fn, state)

            state = self._run_node("stage2_next_question", stage2_next_question_node, state)

            completion = stage2_completion_router(state)
            logger.info("Stage2 completion router: %s", completion)
            if completion == "CONTINUE":
                state = self._run_node("stage2_main", stage2_data_preprocessing_node, state)
                continue
            if completion == "COMPLETE":
                logger.info("Stage2 processing completed for all questions")
                break

        logger.info("Stage 1 & Stage 2 pipeline test finished")
        self._snapshot_state(state, "zz_final_state")
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_node(self, name: str, func: NodeFn, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single node with logging, timing, and snapshot."""
        logger.info("▶ Node start: %s", name)
        start_time = time.time()
        try:
            updated_state = func(state)
        except Exception:  # pragma: no cover - debugging aid
            logger.exception("Node failed: %s", name)
            raise

        elapsed = time.time() - start_time
        logger.info("✅ Node complete: %s (%.2fs)", name, elapsed)

        self.node_counter += 1
        snapshot_label = f"{self.node_counter:02d}_{name}"
        self._snapshot_state(updated_state, snapshot_label)
        return updated_state

    def _snapshot_state(self, state: Dict[str, Any], label: str) -> None:
        """Persist state in JSON (sanitized) and pickle for inspection."""
        json_path = self.snapshot_dir / f"{label}.json"
        pickle_path = self.snapshot_dir / f"{label}.pkl"

        sanitized = self._sanitize_state(state)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(sanitized, f, ensure_ascii=False, indent=2)

        with pickle_path.open("wb") as f:
            pickle.dump(state, f)

        logger.info("💾 State snapshot saved: %s", json_path.name)

    def _sanitize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert complex state objects into JSON-friendly representations."""
        def _convert(value: Any) -> Any:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, dict):
                return {str(k): _convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [_convert(v) for v in value]
            if isinstance(value, pd.DataFrame):
                return {
                    "__type__": "DataFrame",
                    "shape": list(value.shape),
                    "columns": list(value.columns),
                }
            if isinstance(value, pd.Series):
                return {
                    "__type__": "Series",
                    "shape": value.shape,
                    "index": list(value.index),
                }
            if hasattr(value, "to_dict"):
                try:
                    return value.to_dict()
                except Exception:
                    return str(type(value).__name__)
            return str(type(value).__name__)

        return {str(key): _convert(val) for key, val in state.items()}


def main() -> None:
    runner = Stage1Stage2GraphTestRunner()
    final_state = runner.run()
    logger.info("Final state keys: %s", sorted(final_state.keys()))


if __name__ == "__main__":
    main()
