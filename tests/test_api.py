import pytest
from fastapi.testclient import TestClient

import app.server as server


@pytest.fixture()
def client(router, monkeypatch):
    monkeypatch.setattr(server, "load_router", lambda: server.STATE.update(router=router, error=None))
    with TestClient(server.app) as c:
        yield c


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["uses_paid_api"] is False


def test_route_returns_team_and_reasons(client):
    res = client.post("/route", json={"request_id": "SR1", "request_text": "invoice not received for air fryer",
                                      "channel": "email", "product_family": "Air Fryer",
                                      "warranty_status": "in_warranty"})
    assert res.status_code == 200
    body = res.json()
    assert body["team"] == "Billing"
    assert body["request_id"] == "SR1"
    assert body["reasons"]
    assert {"confidence", "confidence_level", "needs_clarification", "alternatives", "route_type"} <= body.keys()


def test_only_request_text_is_required(client):
    assert client.post("/route", json={"request_text": "how to clean air fryer"}).status_code == 200


@pytest.mark.parametrize("payload", [{}, {"request_text": ""}, {"request_text": "x", "channel": "fax"},
                                     {"request_text": "x" * 2001}])
def test_bad_input_is_rejected_politely(client, payload):
    res = client.post("/route", json=payload)
    assert res.status_code == 422
    assert "detail" in res.json()


def test_screen_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "/route" in res.text


def test_service_starts_without_model_or_data(monkeypatch):
    """No model file and no data pack: the service must still start and explain what to do."""
    def missing():
        server.STATE.update(router=None, error="The router is not trained yet and the data pack was not found. "
                                               "Copy the Kestrel data pack into data/ and run: python train.py")
    monkeypatch.setattr(server, "load_router", missing)
    with TestClient(server.app) as c:
        assert c.get("/health").json()["status"] == "not_ready"
        res = c.post("/route", json={"request_text": "fan not working"})
        assert res.status_code == 503
        assert "python train.py" in res.json()["detail"]
        assert c.get("/").status_code == 200


def test_real_load_router_reports_missing_data(monkeypatch, tmp_path):
    """Exercise the real loader with an empty model path and an empty data folder."""
    import kestrel.data as kd
    monkeypatch.setattr(server, "DEFAULT_MODEL_PATH", tmp_path / "none.joblib")
    monkeypatch.setattr(kd, "DATA_DIR", tmp_path)
    orig = kd.load_train
    monkeypatch.setattr(kd, "load_train", lambda data_dir=tmp_path: orig(tmp_path))
    server.STATE.update(router=None, error=None)
    server.load_router()
    assert server.STATE["router"] is None
    assert "data pack was not found" in server.STATE["error"]
