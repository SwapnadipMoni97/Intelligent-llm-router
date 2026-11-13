from types import SimpleNamespace

import pytest

from app.domain import RoutingRequest
from app.provider import LiteLLMProvider
from app.resilience import CircuitBreakers
from app.routing import RoutingEngine, STRATEGIES


class Registry:
    def __init__(self, models): self.models = models
    def list_models(self): return self.models


@pytest.mark.asyncio
async def test_falls_back_to_next_provider(monkeypatch, models):
    import app.provider as provider_module
    calls = []
    async def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) == 1:
            raise RuntimeError("temporary provider outage")
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
                               model_dump=lambda: {"id": "id", "choices": [],
                                                   "usage": {"prompt_tokens": 2, "completion_tokens": 3}})
    monkeypatch.setattr(provider_module.litellm, "acompletion", fake_completion)
    health = CircuitBreakers(threshold=1)
    decision = RoutingEngine(Registry(models), STRATEGIES["intelligent"](), health).route(
        [{"role": "user", "content": "hello"}], RoutingRequest())
    fallbacks = [m for m in models if m.name in decision.fallback_models]
    client = LiteLLMProvider(1, 0, 0.01, health)
    result = await client.complete(decision, [{"role": "user", "content": "hello"}], fallbacks)
    assert result["model"].name != decision.model.name
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_does_not_retry_non_retryable_error(monkeypatch, models):
    import app.provider as provider_module
    calls = 0
    async def fake_completion(**kwargs):
        nonlocal calls
        calls += 1
        error = RuntimeError("bad request")
        error.status_code = 400
        raise error
    monkeypatch.setattr(provider_module.litellm, "acompletion", fake_completion)
    health = CircuitBreakers()
    decision = RoutingEngine(Registry(models[:1]), STRATEGIES["intelligent"](), health).route(
        [{"role": "user", "content": "hello"}], RoutingRequest())
    client = LiteLLMProvider(1, 3, 0.01, health)
    with pytest.raises(provider_module.ProviderFailure):
        await client.complete(decision, [{"role": "user", "content": "hello"}], [])
    assert calls == 1
