# %%

# %%
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Iterable, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

from config.llm.config_llm import MODEL_REGISTRY, ModelConfig  # 앞서 만든 레지스트리
from config.config import settings


SchemaType = Union[dict, type]  # JSON Schema dict 또는 Pydantic/BaseModel 타입
class CreditError(Exception):
    pass

class OpenAIError(Exception):
    pass

class LLMError(Exception):
    pass
@dataclass
class UsageLog:
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cost_usd: float
    raw_usage: Dict[str, Any]

def _calc_cost(cfg: ModelConfig, usage: Dict[str, Any]) -> float:
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    return (pt / 1_000_000.0) * cfg.input_per_1m + (ct / 1_000_000.0) * cfg.output_per_1m

def _extract_usage_and_model(resp, fallback_model: str):
    """
    LangChain AIMessage에서 토큰/모델 정보 추출 (버전 호환)
    우선순위:
    1) resp.usage_metadata = {"input_tokens":.., "output_tokens":.., "total_tokens":..}
    2) resp.response_metadata["usage"] = {"prompt_tokens":.., "completion_tokens":..}
    3) resp.response_metadata["token_usage"] (구버전/커스텀)
    """
    # 1) 권장: usage_metadata
    usage_meta = getattr(resp, "usage_metadata", None)
    if isinstance(usage_meta, dict) and usage_meta:
        pt = usage_meta.get("input_tokens", 0)
        ct = usage_meta.get("output_tokens", 0)
        model_name = (
            getattr(resp, "response_metadata", {}).get("model_name")
            or getattr(resp, "response_metadata", {}).get("model")
            or fallback_model
        )
        return {"prompt_tokens": pt, "completion_tokens": ct}, model_name

    # 2) response_metadata["usage"]
    meta = getattr(resp, "response_metadata", {}) or {}
    usage = meta.get("usage", {}) or meta.get("token_usage", {}) or {}
    if usage:
        model_name = meta.get("model_name") or meta.get("model") or fallback_model
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }, model_name

    # 3) 모두 실패하면 0으로
    model_name = meta.get("model_name") or meta.get("model") or fallback_model
    return {"prompt_tokens": 0, "completion_tokens": 0}, model_name


def _build_messages(
    system: Optional[str] = None,
    user: Optional[str] = None,
    extra_messages: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    if user:
        msgs.append({"role": "user", "content": user})
    if extra_messages:
        # 이미 role 포함된 메시지들을 그대로 추가
        msgs.extend(extra_messages)
    return msgs

class LLMClient:
    def __init__(self, model_key: str = "gpt-4-mini", api_key: str = settings.OPENAI_API_KEY, **overrides):
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_key: {model_key}")
        self.cfg: ModelConfig = MODEL_REGISTRY[model_key]
        base_kwargs = dict(self.cfg.kwargs)

        # model_kwargs 병합
        if "model_kwargs" in base_kwargs or "model_kwargs" in overrides:
            base_mkw = base_kwargs.pop("model_kwargs", {}) or {}
            over_mkw  = overrides.pop("model_kwargs", {}) or {}
            base_kwargs["model_kwargs"] = {**base_mkw, **over_mkw}

        self._base_kwargs = {**base_kwargs, **overrides}
        if api_key:
            self._base_kwargs["api_key"] = api_key
        self.llm = ChatOpenAI(model=self.cfg.name, **self._base_kwargs)

    def _maybe_structured_llm(self, schema: Optional[SchemaType]):
        if schema is None:
            return self.llm
        # LangChain은 Pydantic 모델/타입 또는 JSON Schema dict 모두 지원
        return self.llm.with_structured_output(schema=schema, include_raw=True)

    # -------- 단건 호출 --------
    def chat(
        self,
        *,
        system: Optional[str] = None,
        user: Optional[str] = None,
        extra_messages: Optional[List[Dict[str, Any]]] = None,
        schema: Optional[SchemaType] = None,
        return_raw: bool = False,
        **kwargs,
    ) -> Tuple[Any, UsageLog]:
        """
        system/user를 분리 입력. schema가 주어지면 with_structured_output 적용.
        return_raw=False면 구조화 출력 시 파싱된 객체(Pydantic 인스턴스/Dict)를 그대로 반환.
        """
        messages = _build_messages(system=system, user=user, extra_messages=extra_messages)

        llm = self._maybe_structured_llm(schema)
        t0 = time.time()
        resp = llm.invoke(messages, **kwargs)
        latency_ms = int((time.time() - t0) * 1000)

        usage, model_name = _extract_usage_and_model(resp, self.cfg.name)
        cost = _calc_cost(self.cfg, usage)

        log = UsageLog(
            model=model_name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            cost_usd=round(cost, 6),
            raw_usage=usage,
        )

        if schema is not None and not return_raw:
            # 구조화 출력: LangChain이 이미 파싱한 결과를 반환
            return resp, log
        else:
            # 일반 텍스트 출력
            content = getattr(resp, "content", None) or resp
            return content, log

    # -------- 배치 호출 --------
    def batch(
        self,
        batch_items: List[Dict[str, Any]],
        *,
        schema: Optional[SchemaType] = None,
        return_raw: bool = False,
        **kwargs,
    ) -> Tuple[List[Any], UsageLog]:
        """
        batch_items: 각 아이템은 {system: str|None, user: str|None, extra_messages: list|None}
        예: [{"system": "...", "user": "질문 A"}, {"system": "...", "user": "질문 B"}]
        schema가 주어지면 전체 배치에 동일한 구조화 출력 적용.
        """
        batch_messages: List[List[Dict[str, Any]]] = []
        for item in batch_items:
            msgs = _build_messages(
                system=item.get("system"),
                user=item.get("user"),
                extra_messages=item.get("extra_messages"),
            )
            batch_messages.append(msgs)

        llm = self._maybe_structured_llm(schema)
        t0 = time.time()
        resps = llm.batch(batch_messages, **kwargs)
        latency_ms = int((time.time() - t0) * 1000)

        total_pt = total_ct = 0
        outputs: List[Any] = []

        for r in resps:
            usage, _ = _extract_usage_and_model(r, self.cfg.name)
            total_pt += usage.get("prompt_tokens", 0)
            total_ct += usage.get("completion_tokens", 0)

            if schema is not None and not return_raw:
                outputs.append(r)  # 파싱된 구조 그대로 (Dict/Pydantic 등)
            else:
                outputs.append(getattr(r, "content", None) or r)

        usage_sum = {"prompt_tokens": total_pt, "completion_tokens": total_ct}
        cost = _calc_cost(self.cfg, usage_sum)

        log = UsageLog(
            model=self.cfg.name,
            prompt_tokens=total_pt,
            completion_tokens=total_ct,
            latency_ms=latency_ms,
            cost_usd=round(cost, 6),
            raw_usage=usage_sum,
        )
        return outputs, log

    # 모델 스위치
    def switch_model(self, model_key: str, **overrides):
        self.__init__(model_key=model_key, **overrides)
