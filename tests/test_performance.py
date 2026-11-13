import time

from app.domain import RoutingRequest


def test_router_throughput_smoke(engine_factory, models):
    engine = engine_factory(models)
    messages = [{"role": "user", "content": "Explain this simple question"}]
    start = time.perf_counter()
    for _ in range(1000):
        engine.route(messages, RoutingRequest())
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"1000 local routing decisions took {elapsed:.2f}s"
