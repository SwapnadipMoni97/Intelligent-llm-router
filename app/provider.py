import asyncio
import random
import time
import uuid
from typing import Any

import litellm

from app.domain import ModelSpec, RoutingDecision
from app.observability import ERRORS, FALLBACKS, HEALTH, LATENCY, TOKENS, USAGE_COST


class ProviderFailure(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def provider_model(spec: ModelSpec) -> str:
    if "api_base" in spec.deployment:
        return spec.model_id
    if spec.provider.value == "bedrock" and not spec.model_id.startswith("bedrock/"):
        return f"bedrock/{spec.model_id}"
    if spec.provider.value == "gemini" and not spec.model_id.startswith("gemini/"):
        return f"gemini/{spec.model_id}"
    return spec.model_id


class LiteLLMProvider:
    def __init__(self, timeout: float, max_retries: int, retry_base: float, health):
        self.timeout, self.max_retries, self.retry_base, self.health = timeout, max_retries, retry_base, health

    async def complete(self, decision: RoutingDecision, messages: list[dict[str, Any]],
                       fallbacks: list[ModelSpec] | None = None,
                       temperature=None, max_tokens=None) -> dict[str, Any]:
        candidates = [decision.model] + (fallbacks or [])
        deadline = time.monotonic() + self.timeout
        last_error: Exception | None = None
        for index, spec in enumerate(candidates):
            if time.monotonic() >= deadline:
                break
            if index:
                FALLBACKS.labels(model=spec.name).inc()
            for attempt in range(self.max_retries + 1):
                start = time.perf_counter()
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    kwargs = dict(model=provider_model(spec), messages=messages, timeout=self.timeout,
                                  api_base=spec.deployment.get("api_base"),
                                  temperature=temperature, max_tokens=max_tokens)
                    kwargs = {k: v for k, v in kwargs.items() if v is not None}
                    result = await asyncio.wait_for(litellm.acompletion(**kwargs), timeout=remaining)
                    elapsed = time.perf_counter() - start
                    LATENCY.labels(model=spec.name).observe(elapsed)
                    self.health.success(spec.name)
                    HEALTH.labels(model=spec.name).set(1)
                    usage = getattr(result, "usage", None)
                    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(usage, "completion_tokens", 0) or 0
                    TOKENS.labels(model=spec.name, direction="input").inc(input_tokens)
                    TOKENS.labels(model=spec.name, direction="output").inc(output_tokens)
                    if spec.input_price_per_million is not None and spec.output_price_per_million is not None:
                        actual = (input_tokens * spec.input_price_per_million + output_tokens * spec.output_price_per_million) / 1e6
                        USAGE_COST.labels(model=spec.name).inc(actual)
                    return {"result": result, "model": spec, "latency": elapsed}
                except Exception as exc:
                    last_error = exc
                    self.health.failure(spec.name)
                    HEALTH.labels(model=spec.name).set(1 if self.health.is_available(spec.name) else 0)
                    kind = type(exc).__name__
                    ERRORS.labels(model=spec.name, kind=kind[:64]).inc()
                    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in type(exc).__name__.lower():
                        raise ProviderFailure("Provider request timed out") from exc
                    retryable = _retryable(exc)
                    if retryable and attempt < self.max_retries:
                        delay = self.retry_base * (2 ** attempt) * random.uniform(0.75, 1.25)
                        await asyncio.sleep(min(delay, max(0, deadline - time.monotonic())))
                        continue
                    break
        # Avoid returning provider exception text, which can contain sensitive details.
        raise ProviderFailure("All eligible providers failed", retryable=False) from last_error


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is not None:
        return int(status) == 429
    name = type(exc).__name__.lower()
    return any(term in name for term in ("ratelimit", "serviceunavailable", "internalserver",
                                         "connecterror", "connectionerror", "connecttimeout"))


def new_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex
