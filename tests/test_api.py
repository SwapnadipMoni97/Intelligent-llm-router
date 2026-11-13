from fastapi.testclient import TestClient

import app.main as main
from app.domain import RoutingRequest
from app.routing import RoutingEngine, STRATEGIES


class Registry:
    def __init__(self, models): self.models = models
    def list_models(self): return self.models


def test_route_endpoint_returns_decision(monkeypatch, models):
    main.registry = Registry(models)
    main.engine = RoutingEngine(main.registry, STRATEGIES["intelligent"](), main.health)
    with TestClient(main.app) as client:
        response = client.post("/v1/route", json={"messages": [{"role": "user", "content": "hello"}]})
    assert response.status_code == 200
    assert response.json()["selected_model"]["name"] == "economy"


def test_streaming_is_rejected_explicitly():
    with TestClient(main.app) as client:
        response = client.post("/v1/chat/completions", json={"stream": True,
                             "messages": [{"role": "user", "content": "hello"}]})
    assert response.status_code == 501
