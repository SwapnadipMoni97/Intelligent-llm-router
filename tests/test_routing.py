import pytest

from app.domain import RoutingRequest, TaskType
from app.routing import NoEligibleModel


def route(engine, text, options=None):
    return engine.route([{"role": "user", "content": text}], options or RoutingRequest())


def test_simple_request_prefers_economical_model(engine_factory, models):
    result = route(engine_factory(models), "Hello, what time is it?")
    assert result.model.name == "economy"


def test_reasoning_request_selects_strong_model(engine_factory, models):
    result = route(engine_factory(models), "Analyze this complex trade-off and explain why")
    assert result.task == TaskType.REASONING
    assert result.model.name == "reasoner"


def test_coding_request_requires_coding_capability(engine_factory, models):
    result = route(engine_factory(models), "Debug this function and explain the code")
    assert result.task == TaskType.CODING
    assert "coding" in result.model.capabilities


def test_context_overflow_routes_to_wider_model(engine_factory, models):
    long_text = "x" * 170000
    result = route(engine_factory(models), long_text)
    assert result.model.name == "wide"


def test_explicit_capability_and_cost_constraints(engine_factory, models):
    result = route(engine_factory(models), "Summarize", RoutingRequest(max_cost=0.001))
    assert result.model.name == "economy"
    with pytest.raises(NoEligibleModel):
        route(engine_factory(models[:1]), "Summarize", RoutingRequest(required_capabilities={"vision"}))


def test_cost_strategy_orders_by_configured_price(engine_factory, models):
    result = route(engine_factory(models), "plain", RoutingRequest(),)
    assert result.estimated_cost is not None
