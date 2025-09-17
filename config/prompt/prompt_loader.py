# config/prompt_loader.py
# LangGraph / LangChain friendly prompt loader with branch resolution
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, TypedDict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
import os
import yaml
from pydantic import BaseModel, Field, model_validator, field_validator
from string import Formatter

# Import schema registry
from config.schemas import SCHEMA_REGISTRY

# Optional LangChain imports (only required if you use the helpers)
try:
    from langchain_core.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.runnables import Runnable
    from langchain_core.output_parsers import StrOutputParser
except Exception:  # pragma: no cover - allow usage without LangChain installed
    ChatPromptTemplate = SystemMessagePromptTemplate = HumanMessagePromptTemplate = None
    SystemMessage = HumanMessage = None
    Runnable = None
    StrOutputParser = None

# ---------------------------------------------------------------------------
# Config path (env override supported)
# ---------------------------------------------------------------------------
_DEFAULT_CFG_PATH = Path(__file__).resolve().parent / "prompt.config.yaml"
ENV_PATH = os.getenv("PROMPT_CONFIG_PATH")
CONFIG_PATH = Path(ENV_PATH).expanduser().resolve() if ENV_PATH else _DEFAULT_CFG_PATH


# ---------------------------------------------------------------------------
# Pydantic models for config schema
# ---------------------------------------------------------------------------
class BranchCfg(BaseModel):
    model: Optional[str] = None
    system: Optional[str] = None
    user_template: Optional[str] = None
    schema: Optional[str] = None  # Schema name from SCHEMA_REGISTRY
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("system", "user_template", mode="before")
    @classmethod
    def _strip_text(cls, v):
        return v.strip() if isinstance(v, str) else v


class PromptConfig(BaseModel):
    version: str
    defaults: Dict[str, Any]
    branches: Dict[str, BranchCfg]

    @model_validator(mode="before")
    @classmethod
    def _normalize_defaults(cls, values):
        if isinstance(values, dict):
            defaults = values.get("defaults") or {}
            if "model" in defaults and isinstance(defaults["model"], str):
                defaults["model"] = defaults["model"].strip()
            values["defaults"] = defaults
        return values


# ---------------------------------------------------------------------------
# Types & utilities
# ---------------------------------------------------------------------------
class ResolvedPrompt(TypedDict, total=False):
    model: str
    system: str
    user_template: str
    schema: Optional[Any]  # Resolved Pydantic schema class
    params: Dict[str, Any]
    version: str
    branch: str


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _template_vars(tmpl: str) -> Set[str]:
    if not tmpl:
        return set()
    return {field for _, field, _, _ in Formatter().parse(tmpl) if field}


# ---------------------------------------------------------------------------
# Load / reload
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_prompt_config(path: Path = CONFIG_PATH) -> PromptConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PromptConfig(**data)


def reload_prompt_config() -> None:
    """Clear cache so next load reads from disk (hot-reload for dev)."""
    load_prompt_config.cache_clear()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Branch resolution (core API)
# ---------------------------------------------------------------------------
def resolve_branch(branch: str, *, path: Path = CONFIG_PATH) -> ResolvedPrompt:
    cfg = load_prompt_config(path)
    if branch not in cfg.branches:
        raise KeyError(
            f"Unknown branch '{branch}'. Available: {', '.join(sorted(cfg.branches.keys()))}"
        )

    b = cfg.branches[branch]

    # Top-level fallbacks
    model = b.model or cfg.defaults.get("model")
    system = b.system or cfg.defaults.get("system", "")
    user_template = b.user_template or cfg.defaults.get("user_template", "")
    
    # Schema resolution
    schema = None
    if b.schema:
        # Try to get schema from registry first
        if b.schema in SCHEMA_REGISTRY:
            schema = SCHEMA_REGISTRY[b.schema]
        else:
            # Fall back to auto-detecting by branch name
            if branch in SCHEMA_REGISTRY:
                schema = SCHEMA_REGISTRY[branch]
    elif branch in SCHEMA_REGISTRY:
        # Auto-detect schema by branch name if no explicit schema specified
        schema = SCHEMA_REGISTRY[branch]

    # params = defaults (non-top-level) ⊕ branch.params
    default_params = {
        k: v
        for k, v in cfg.defaults.items()
        if k not in {"model", "system", "user_template", "schema"}
    }
    params = _deep_merge(default_params, b.params or {})

    if not model:
        raise ValueError(
            f"Branch '{branch}' does not specify a model and no default model exists."
        )
    if not user_template:
        raise ValueError(
            f"Branch '{branch}' missing 'user_template' (neither branch nor defaults provide it)."
        )

    return ResolvedPrompt(
        version=cfg.version,
        branch=branch,
        model=model,
        system=system or "",
        user_template=user_template,
        schema=schema,
        params=params,
    )


# ---------------------------------------------------------------------------
# OpenAI-style payload (optional)
# ---------------------------------------------------------------------------
def get_prompt(branch: str, /, **vars: Any) -> Dict[str, Any]:
    """Render to an OpenAI-compatible payload: {model, messages, **params}."""
    rp = resolve_branch(branch)
    try:
        user_text = rp["user_template"].format(**vars)
    except KeyError as e:
        missing = e.args[0]
        raise KeyError(
            f"Template variable '{missing}' not provided for branch '{branch}'."
        ) from e

    messages = []
    if rp["system"]:
        messages.append({"role": "system", "content": rp["system"]})
    messages.append({"role": "user", "content": user_text})

    payload: Dict[str, Any] = {"model": rp["model"], "messages": messages}
    payload.update(rp["params"])
    return payload


# ---------------------------------------------------------------------------
# LangChain / LangGraph helpers
# ---------------------------------------------------------------------------

def required_vars(branch: str) -> Set[str]:
    """Return the set of template variables required by system + user templates."""
    rp = resolve_branch(branch)
    return _template_vars(rp["system"]) | _template_vars(rp["user_template"])


def to_lc_prompt(branch: str):
    """Build a ChatPromptTemplate (requires langchain).
    Returns ChatPromptTemplate.
    """
    if ChatPromptTemplate is None:
        raise ImportError("LangChain not installed; install langchain-core to use to_lc_prompt().")

    rp = resolve_branch(branch)

    parts = []
    if rp["system"]:
        parts.append(SystemMessagePromptTemplate.from_template(rp["system"]))
    parts.append(HumanMessagePromptTemplate.from_template(rp["user_template"]))

    prompt = ChatPromptTemplate.from_messages(parts)
    # Attach metadata for introspection/debugging
    prompt.metadata = {"params": rp["params"], "branch": rp["branch"], "version": rp["version"], "model": rp["model"]}
    return prompt


def make_lcel_chain(branch: str, llm: Any) -> Any:
    """Create an LCEL chain: ChatPromptTemplate | LLM | StrOutputParser.
    Applies common params (temperature, max_tokens, stop, etc.) if supported by your LLM.
    """
    if Runnable is None or StrOutputParser is None:
        raise ImportError("LangChain not installed; install langchain-core to use make_lcel_chain().")

    rp = resolve_branch(branch)
    prompt = to_lc_prompt(branch)

    # Bind supported params transparently
    supported_keys = {"temperature", "max_tokens", "stop", "top_p", "frequency_penalty", "presence_penalty"}
    bind_kwargs = {k: v for k, v in rp["params"].items() if k in supported_keys}
    llm_with = getattr(llm, "bind", None)
    if callable(llm_with):
        llm = llm.bind(model=rp["model"], **bind_kwargs)
    # If .bind not available, fall back to llm as-is (some providers set model in constructor)

    return prompt | llm | StrOutputParser()


def format_messages(branch: str, **vars):
    """Return LangChain message objects (SystemMessage/HumanMessage)."""
    if SystemMessage is None:
        raise ImportError("LangChain not installed; install langchain-core to use format_messages().")

    rp = resolve_branch(branch)
    try:
        user_text = rp["user_template"].format(**vars)
    except KeyError as e:
        missing = e.args[0]
        raise KeyError(
            f"Template variable '{missing}' not provided for branch '{branch}'."
        ) from e

    msgs: List[Any] = []
    if rp["system"]:
        msgs.append(SystemMessage(content=rp["system"]))
    msgs.append(HumanMessage(content=user_text))
    return msgs


essential_msg_keys = ("role", "content")


def format_dict_messages(branch: str, **vars):
    """Return list of dict messages: [{role, content}, ...] useful for messages-in-state patterns."""
    rp = resolve_branch(branch)
    try:
        user_text = rp["user_template"].format(**vars)
    except KeyError as e:
        missing = e.args[0]
        raise KeyError(
            f"Template variable '{missing}' not provided for branch '{branch}'."
        ) from e

    out: List[Dict[str, str]] = []
    if rp["system"]:
        out.append({"role": "system", "content": rp["system"]})
    out.append({"role": "user", "content": user_text})

    # light validation
    for m in out:
        for k in essential_msg_keys:
            if k not in m:
                raise ValueError(f"Message {m} missing key '{k}'")
    return out


# ---------------------------------------------------------------------------
# Optional tiny router (customize as needed)
# ---------------------------------------------------------------------------

def choose_branch_by_length(text: str, *, threshold: int = 3000) -> str:
    """Example router: long inputs → labeling_long_input, else grammar_fix."""
    return "labeling_long_input" if len(text) > threshold else "grammar_fix"


# Module-level cached config if someone still wants it
CFG = load_prompt_config()
