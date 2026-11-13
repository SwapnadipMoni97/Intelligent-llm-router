import logging
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import get_settings
from app.domain import RoutingRequest
from app.observability import END_TO_END, EST_COST, REQUESTS, SELECTIONS
from app.provider import LiteLLMProvider, ProviderFailure, new_completion_id
from app.registry import YamlModelRegistry
from app.resilience import CircuitBreakers, InMemoryRateLimiter
from app.routing import (NoEligibleModel, RoutingEngine, STRATEGIES)
from app.schemas import ChatCompletionRequest, ChatCompletionResponse, RouteResponse

settings = get_settings()
if settings.structured_logs:
    structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"),
                                    structlog.processors.add_log_level,
                                    structlog.processors.JSONRenderer()],
                        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
log = structlog.get_logger()
registry = YamlModelRegistry(settings.model_registry_path)
health = CircuitBreakers(settings.circuit_failure_threshold, settings.circuit_recovery_seconds)
limiter = InMemoryRateLimiter(settings.rate_limit_requests_per_minute)
provider = LiteLLMProvider(settings.request_timeout_seconds, settings.max_retries,
                           settings.retry_base_seconds, health)
strategy_cls = STRATEGIES.get(settings.routing_strategy)
if strategy_cls is None:
    raise RuntimeError(f"Unknown ROUTING_STRATEGY: {settings.routing_strategy}")
engine = RoutingEngine(registry, strategy_cls(), health)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)


async def enforce_limit(request: Request):
    # Put the service behind an authenticated gateway; do not trust a caller-supplied tenant header.
    key = request.client.host if request.client else "unknown"
    if not await limiter.allow(key):
        raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "60"})


def make_decision(messages, options: RoutingRequest):
    try:
        return engine.route(messages, options)
    except NoEligibleModel as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/health")
async def health_check():
    try:
        models = registry.list_models()
        return {"status": "ok", "configured_models": len(models),
                "available_models": sum(health.is_available(m.name) and m.available for m in models)}
    except Exception:
        return JSONResponse(status_code=503, content={"status": "degraded"})


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [m.model_dump(exclude={"deployment"})
                                         for m in registry.list_models()]}


@app.post("/v1/route", response_model=RouteResponse)
async def route_only(body: ChatCompletionRequest, request: Request):
    await enforce_limit(request)
    options = body.routing.model_copy(update={"max_output_tokens": body.max_tokens})
    decision = make_decision([m.model_dump() for m in body.messages], options)
    if decision.estimated_cost is not None:
        EST_COST.labels(model=decision.model.name).inc(decision.estimated_cost)
    log.info("routing_decision", model=decision.model.name, task=decision.task.value,
             reason=decision.reason, estimated_cost=decision.estimated_cost)
    return RouteResponse(selected_model=decision.model.model_copy(update={"deployment": {}}), task=decision.task.value,
                         complexity=decision.complexity,
                         estimated_input_tokens=decision.estimated_input_tokens,
                         estimated_output_tokens=decision.estimated_output_tokens,
                         estimated_cost=decision.estimated_cost, reason=decision.reason,
                         fallbacks=decision.fallback_models)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(body: ChatCompletionRequest, request: Request):
    await enforce_limit(request)
    if body.stream:
        raise HTTPException(501, "Streaming is not enabled in this initial implementation")
    messages = [m.model_dump() for m in body.messages]
    options = body.routing.model_copy(update={"max_output_tokens": body.max_tokens})
    decision = make_decision(messages, options)
    SELECTIONS.labels(model=decision.model.name, strategy=settings.routing_strategy).inc()
    if decision.estimated_cost is not None:
        EST_COST.labels(model=decision.model.name).inc(decision.estimated_cost)
    candidates = {m.name: m for m in registry.list_models()}
    fallback_specs = [candidates[n] for n in decision.fallback_models if n in candidates
                      and health.is_available(n)]
    started = time.perf_counter()
    try:
        completed = await provider.complete(decision, messages, fallback_specs,
                                            temperature=body.temperature, max_tokens=body.max_tokens)
        result = completed["result"]
        chosen = completed["model"]
        # LiteLLM objects are Pydantic-compatible; return only the standard completion fields.
        raw = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        usage = raw.get("usage") or {}
        REQUESTS.labels(status="success", task=decision.task.value).inc()
        elapsed_ms = (time.perf_counter() - started) * 1000
        END_TO_END.labels(model=chosen.name).observe(elapsed_ms / 1000)
        log.info("model_request_complete", model=chosen.name, task=decision.task.value,
                 latency_ms=round(elapsed_ms, 2), input_tokens=usage.get("prompt_tokens", 0),
                 output_tokens=usage.get("completion_tokens", 0), fallback=chosen.name != decision.model.name)
        return ChatCompletionResponse(id=raw.get("id") or new_completion_id(),
                                      model=chosen.name, choices=raw.get("choices", []), usage=usage,
                                      routing={"selected_model": chosen.name, "task": decision.task.value,
                                               "reason": decision.reason,
                                               "fallback": chosen.name != decision.model.name,
                                               "estimated_cost": decision.estimated_cost,
                                               "latency_ms": round(elapsed_ms, 2)})
    except ProviderFailure as exc:
        REQUESTS.labels(status="error", task=decision.task.value).inc()
        log.warning("provider_request_failed", selected_model=decision.model.name)
        raise HTTPException(502, str(exc)) from exc


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
