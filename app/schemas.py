from typing import Any

from pydantic import BaseModel, Field

from app.domain import ModelSpec, RoutingRequest


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False
    routing: RoutingRequest = Field(default_factory=RoutingRequest)


class RouteResponse(BaseModel):
    selected_model: ModelSpec
    task: str
    complexity: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: float | None
    reason: str
    fallbacks: list[str]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]
    routing: dict[str, Any]
