# config_llm.py
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class ModelConfig:
    name: str
    input_per_1m: float   # USD per 1M prompt tokens
    output_per_1m: float  # USD per 1M completion tokens
    kwargs: dict          # temperature, reasoning 등 모델별 기본값

# 가격은 예시값(placeholder)입니다. 실제 단가는 운영 시 외부 설정으로 주입 권장.
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "gpt-5": ModelConfig(
        name="gpt-5",
        input_per_1m=1.25, output_per_1m=10.0,  # $125 per 1M input tokens, $1000 per 1M output tokens
        kwargs={"verbosity": "low", "reasoning_effort": "medium"}
    ),

    "gpt-5-mini": ModelConfig(
        name="gpt-5-mini",
        input_per_1m=0.25, output_per_1m=2.0,  # $25 per 1M input tokens, $200 per 1M output tokens
        kwargs={"verbosity": "low", "reasoning_effort": "medium"}
    ),
    "gpt-5-nano": ModelConfig(
        name="gpt-5-nano",
        input_per_1m=0.05, output_per_1m=0.4,  # $25 per 1M input tokens, $200 per 1M output tokens
        kwargs={"verbosity": "low", "reasoning_effort": "medium"}
    ),
    
    "gpt-4.1": ModelConfig(
        name="gpt-4.1",
        input_per_1m=2.0, output_per_1m=8.0,  # $250 per 1M input tokens, $1000 per 1M output tokens
        kwargs={"temperature": 0.2}
    ),
    "gpt-4.1-mini": ModelConfig(
        name="gpt-4.1-mini",
        input_per_1m=0.4, output_per_1m=1.6,  # $15 per 1M input tokens, $600 per 1M output tokens
        kwargs={"temperature": 0.2}
    ),

    "gpt-4.1-nano": ModelConfig(
        name="gpt-4.1-nano",
        input_per_1m=0.1, output_per_1m=0.40,  # $50 per 1M input tokens, $150 per 1M output tokens
        kwargs={"temperature": 0.2}
    ),
}
