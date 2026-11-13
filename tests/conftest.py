import pytest

from app.domain import ModelSpec, Provider
from app.resilience import CircuitBreakers
from app.routing import RoutingEngine, STRATEGIES


class MemoryRegistry:
    def __init__(self, models):
        self.models = models

    def list_models(self):
        return self.models


@pytest.fixture
def models():
    return [
        ModelSpec(name="economy", provider=Provider.OPENAI, model_id="mock/economy",
                  capabilities={"general", "structured_output"}, context_window=5000,
                  input_price_per_million=0.1, output_price_per_million=0.2,
                  expected_latency_ms=300, quality_score=0.55),
        ModelSpec(name="reasoner", provider=Provider.ANTHROPIC, model_id="mock/reasoner",
                  capabilities={"general", "reasoning", "coding"}, context_window=50000,
                  input_price_per_million=2, output_price_per_million=8,
                  expected_latency_ms=900, quality_score=0.95),
        ModelSpec(name="wide", provider=Provider.GEMINI, model_id="mock/wide",
                  capabilities={"general", "reasoning", "coding", "vision"}, context_window=100000,
                  input_price_per_million=1, output_price_per_million=3,
                  expected_latency_ms=500, quality_score=0.85),
    ]


@pytest.fixture
def engine_factory():
    def make(models, strategy="intelligent", health=None):
        return RoutingEngine(MemoryRegistry(models), STRATEGIES[strategy](), health or CircuitBreakers())
    return make
