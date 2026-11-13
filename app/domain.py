from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    GEMINI = "gemini"
    OPENAI_COMPATIBLE = "openai_compatible"


class TaskType(StrEnum):
    GENERAL = "general"
    REASONING = "reasoning"
    CODING = "coding"
    EXTRACTION = "extraction"
    CREATIVE = "creative"
    MULTIMODAL = "multimodal"


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    provider: Provider
    model_id: str
    capabilities: set[str] = Field(default_factory=set)
    context_window: int = Field(gt=0)
    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)
    expected_latency_ms: float | None = Field(default=None, gt=0)
    quality_score: float | None = Field(default=None, ge=0, le=1)
    available: bool = True
    deployment: dict[str, Any] = Field(default_factory=dict)
    fallback_group: str = "default"


class RoutingRequest(BaseModel):
    task_hint: TaskType | None = None
    quality_tier: str = "balanced"
    max_cost: float | None = Field(default=None, ge=0)
    max_latency_ms: float | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    allowed_providers: set[Provider] | None = None
    required_capabilities: set[str] = Field(default_factory=set)


class RoutingDecision(BaseModel):
    model: ModelSpec
    task: TaskType
    complexity: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: float | None
    reason: str
    fallback_models: list[str] = Field(default_factory=list)
