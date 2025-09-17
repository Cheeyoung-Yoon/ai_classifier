# tools/llm_router.py
from __future__ import annotations
from typing import Any, Dict, Optional, Union
from functools import lru_cache

from io_layer.llm.client import LLMClient, CreditError, OpenAIError, LLMError
from config.llm.config_llm import MODEL_REGISTRY
from config.prompt.prompt_loader import PromptConfig, resolve_branch

SchemaType = Union[dict, type]


class LLMRouter:
    """
    - YAML 라우팅을 받아 브랜치별로 모델/프롬프트/파라미터를 적용
    - 모델 키(또는 실제 모델 이름)별로 LLMClient 캐시
    """
    def __init__(self, config_path: Optional[str] = None, api_key: Optional[str] = None):
        self.config_path = config_path
        self.api_key = api_key

    @lru_cache(maxsize=16)
    def _get_client(self, model_key: str) -> LLMClient:
        # MODEL_REGISTRY 키와 실제 모델명이 1:1이 아닐 수 있음
        # 우선 registry 키로 시도, 없으면 이름 그대로 사용
        key = model_key if model_key in MODEL_REGISTRY else model_key
        return LLMClient(model_key=key, api_key=self.api_key)

    def run(
        self,
        branch: str,
        variables: Dict[str, Any],
        *,
        schema: Optional[SchemaType] = None,
        override_model: Optional[str] = None,
        extra_messages: Optional[list[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        try:
            # resolve_branch를 사용하여 설정 해결
            resolved = resolve_branch(branch)
            
            model_key = override_model or resolved["model"]
            client = self._get_client(model_key)

            system = resolved["system"]
            user = resolved["user_template"].format(**variables)

            # defaults.params < branch.params < kwargs 순으로 병합
            # 모델별 파라미터 필터링
            base_params = dict(resolved.get("params", {}))
            base_params.update(kwargs or {})
            
            # GPT-5 모델은 특정 파라미터들을 지원하지 않음
            if model_key in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
                unsupported_params = ["temperature", "stop", "max_tokens"]
                removed_params = []
                for param in unsupported_params:
                    if param in base_params:
                        del base_params[param]
                        removed_params.append(param)
                if removed_params:
                    print(f"Removed unsupported parameters for GPT-5 model {model_key}: {removed_params}")

            # Chat 호출
            content, usage = client.chat(
                system=system,
                user=user,
                extra_messages=extra_messages,
                schema=schema,
                **base_params,
            )

            return {
                "result": content,
                "usage": usage.__dict__,   # 토큰/비용/지연
                "model": model_key,
                "branch": branch,
                "applied_params": base_params,
            }
            
        except CreditError as e:
            raise CreditError(f"Credit exhausted for branch '{branch}': {e}") from e
        except OpenAIError as e:
            raise OpenAIError(f"OpenAI error for branch '{branch}': {e}") from e
        except LLMError as e:
            raise LLMError(f"LLM error for branch '{branch}': {e}") from e
        except Exception as e:
            raise LLMError(f"Unexpected error for branch '{branch}': {e}") from e


# 글로벌 라우터 인스턴스
_router = None

def get_router():
    """글로벌 라우터 인스턴스 반환"""
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router

def call_llm_router(branch: str, **kwargs) -> Any:
    """
    LLM 라우터 호출을 위한 편의 함수
    
    Args:
        branch: 사용할 브랜치 이름
        **kwargs: LLM에 전달할 파라미터들
    
    Returns:
        LLM 응답 결과
    """
    router = get_router()
    # kwargs를 variables로 전달
    result = router.run(branch, variables=kwargs)
    return result["result"]
