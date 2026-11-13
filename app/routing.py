import re
from abc import ABC, abstractmethod

from app.domain import ModelSpec, RoutingDecision, RoutingRequest, TaskType

_CODE = re.compile(r"```|\b(function|class|debug|compile|stack trace|algorithm|code)\b", re.I)
_REASONING = re.compile(r"\b(analy[sz]e|reason|prove|compare|trade-?off|step by step|complex)\b", re.I)
_EXTRACT = re.compile(r"\b(extract|summari[sz]e|classify|convert|json|structured)\b", re.I)


def estimate_tokens(text: str) -> int:
    # Conservative generic estimate; use a provider tokenizer when one is available.
    return max(1, (len(text) + 2) // 3)


def classify(text: str, hint: TaskType | None = None) -> TaskType:
    if hint:
        return hint
    if re.search(r"\b(image_url|image|audio_url|video_url)\b", text, re.I):
        return TaskType.MULTIMODAL
    if _CODE.search(text):
        return TaskType.CODING
    if _REASONING.search(text):
        return TaskType.REASONING
    if _EXTRACT.search(text):
        return TaskType.EXTRACTION
    return TaskType.GENERAL


def complexity(text: str, task: TaskType) -> float:
    score = min(1.0, len(text) / 12000)
    if task == TaskType.REASONING:
        score += 0.35
    elif task == TaskType.CODING:
        score += 0.2
    return min(1.0, score)


class RoutingStrategy(ABC):
    @abstractmethod
    def choose(self, candidates: list[ModelSpec], task: TaskType,
               complexity_score: float) -> ModelSpec: ...

    def rank(self, candidates: list[ModelSpec], task: TaskType,
             complexity_score: float) -> list[ModelSpec]:
        return sorted(candidates, key=lambda m: m.name)


class RuleBasedRouter(RoutingStrategy):
    def rank(self, candidates, task, complexity_score):
        cap = {TaskType.CODING: "coding", TaskType.REASONING: "reasoning",
               TaskType.MULTIMODAL: "vision", TaskType.EXTRACTION: "structured_output"}.get(task)
        if cap:
            candidates = [m for m in candidates if cap in m.capabilities]
        return sorted(candidates, key=lambda m: (
            (m.quality_score or 0.0) < complexity_score * 0.7,
            m.expected_latency_ms or float("inf"),
            m.input_price_per_million is None,
            m.input_price_per_million or float("inf"), m.name))

    def choose(self, candidates, task, complexity_score):
        return self.rank(candidates, task, complexity_score)[0]


class CostOptimizedRouter(RuleBasedRouter):
    def rank(self, candidates, task, complexity_score):
        candidates = super().rank(candidates, task, complexity_score)
        return sorted(candidates, key=lambda m: (
            m.input_price_per_million is None or m.output_price_per_million is None,
            (m.input_price_per_million or 0) + (m.output_price_per_million or 0),
            m.expected_latency_ms or float("inf"), m.name))


class LatencyOptimizedRouter(RuleBasedRouter):
    def rank(self, candidates, task, complexity_score):
        candidates = super().rank(candidates, task, complexity_score)
        return sorted(candidates, key=lambda m: (m.expected_latency_ms is None,
                                                  m.expected_latency_ms or float("inf"), m.name))


class QualityOptimizedRouter(RuleBasedRouter):
    def rank(self, candidates, task, complexity_score):
        candidates = super().rank(candidates, task, complexity_score)
        return sorted(candidates, key=lambda m: (m.quality_score is None,
                                                  -(m.quality_score or 0.0), m.name))


class IntelligentRouter(RuleBasedRouter):
    """Constrained cost/latency/quality ranking; tune weights with measured outcomes."""
    def rank(self, candidates, task, complexity_score):
        base = super().rank(candidates, task, complexity_score)
        if task == TaskType.REASONING or complexity_score >= 0.7:
            return QualityOptimizedRouter().rank(base, task, complexity_score)
        def score(m):
            quality_penalty = max(0.0, complexity_score * 0.75 - (m.quality_score or 0.0)) * 100
            cost = ((m.input_price_per_million or 0) + (m.output_price_per_million or 0))
            latency = m.expected_latency_ms or 10000
            return quality_penalty + cost + latency / 10000, m.name
        return sorted(base, key=score)


STRATEGIES: dict[str, type[RoutingStrategy]] = {
    "rule_based": RuleBasedRouter, "cost": CostOptimizedRouter,
    "latency": LatencyOptimizedRouter, "quality": QualityOptimizedRouter,
    "intelligent": IntelligentRouter,
}


class RoutingEngine:
    def __init__(self, registry, strategy: RoutingStrategy, health):
        self.registry, self.strategy, self.health = registry, strategy, health

    def route(self, messages: list[dict], options: RoutingRequest) -> RoutingDecision:
        text = "\n".join(str(m.get("content", "")) for m in messages)
        task = classify(text, options.task_hint)
        score = complexity(text, task)
        input_tokens = estimate_tokens(text)
        output_tokens = options.max_output_tokens or (1024 if score > 0.55 else 512)
        candidates = [m for m in self.registry.list_models() if m.available
                      and self.health.is_available(m.name)
                      and (options.allowed_providers is None or m.provider in options.allowed_providers)
                      and options.required_capabilities.issubset(m.capabilities)
                      and input_tokens + output_tokens <= m.context_window]
        if options.max_cost is not None:
            candidates = [m for m in candidates
                          if m.input_price_per_million is not None and m.output_price_per_million is not None
                          and (input_tokens * m.input_price_per_million
                               + output_tokens * m.output_price_per_million) / 1_000_000 <= options.max_cost]
        if options.max_latency_ms is not None:
            candidates = [m for m in candidates if m.expected_latency_ms is not None
                          and m.expected_latency_ms <= options.max_latency_ms]
        if not candidates:
            raise NoEligibleModel("No available model satisfies capability, policy, and context constraints")
        ranked = self.strategy.rank(candidates, task, score)
        selected = ranked[0]
        cost = None
        if selected.input_price_per_million is not None and selected.output_price_per_million is not None:
            cost = (input_tokens * selected.input_price_per_million
                    + output_tokens * selected.output_price_per_million) / 1_000_000
        fallbacks = [m.name for m in ranked if m.name != selected.name]
        return RoutingDecision(model=selected, task=task, complexity=score,
                               estimated_input_tokens=input_tokens, estimated_output_tokens=output_tokens,
                               estimated_cost=cost, reason=f"{task.value}; strategy={type(self.strategy).__name__}",
                               fallback_models=fallbacks)


class NoEligibleModel(Exception):
    pass
