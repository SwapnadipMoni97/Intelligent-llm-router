from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("router_requests_total", "Router requests", ["status", "task"])
SELECTIONS = Counter("router_selections_total", "Selected model counts", ["model", "strategy"])
FALLBACKS = Counter("router_fallbacks_total", "Provider fallback attempts", ["model"])
ERRORS = Counter("router_provider_errors_total", "Provider errors", ["model", "kind"])
LATENCY = Histogram("router_provider_latency_seconds", "Provider call latency", ["model"])
END_TO_END = Histogram("router_end_to_end_latency_seconds", "End-to-end response latency", ["model"])
TTFT = Histogram("router_time_to_first_token_seconds", "Time to first token", ["model"])
TOKENS = Counter("router_tokens_total", "Tokens reported by providers", ["model", "direction"])
EST_COST = Counter("router_estimated_cost_usd_total", "Estimated cost in USD", ["model"])
USAGE_COST = Counter("router_usage_cost_usd_total", "Usage-based cost estimate from provider token counts", ["model"])
QUALITY = Histogram("router_quality_evaluation_score", "External quality evaluation scores", ["model", "evaluator"])
HEALTH = Gauge("router_model_available", "Model circuit availability", ["model"])
