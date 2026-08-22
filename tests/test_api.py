import os

os.environ["DEMO_MODE"] = "true"
os.environ["MODEL_PATH"] = "rl/checkpoints/best_model.pt"

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.hazard_service import get_hazard_store


@pytest.fixture(autouse=True)
def clear_hazards():
    store = get_hazard_store()
    for h in list(store.list_active()):
        store.deactivate(h.id)
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["demo_mode"] is True


def test_root_endpoint_has_disclaimer(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "disclaimer" in resp.json()


def test_geocode_mock_provider(client):
    resp = client.post("/api/geocode", json={"address": "Erode, Tamil Nadu"})
    assert resp.status_code == 200
    body = resp.json()
    assert -90 <= body["latitude"] <= 90
    assert -180 <= body["longitude"] <= 180


def test_route_endpoint(client):
    resp = client.post("/api/route", json={
        "origin": {"latitude": 11.341, "longitude": 77.717},
        "destination": {"latitude": 11.350, "longitude": 77.725},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_distance_m"] > 0
    assert body["source"] == "mock_provider"


def test_invalid_gps_rejected(client):
    resp = client.post("/api/route", json={
        "origin": {"latitude": 999, "longitude": 77.717},
        "destination": {"latitude": 11.350, "longitude": 77.725},
    })
    assert resp.status_code == 422


def test_hazard_crud(client):
    create_resp = client.post("/api/hazards", json={
        "id": "HZ-TEST", "location": {"latitude": 11.34, "longitude": 77.71},
        "radius_m": 300, "severity": 0.5, "hazard_type": "flood", "hard_constraint": False,
    })
    assert create_resp.status_code == 200
    list_resp = client.get("/api/hazards")
    assert list_resp.status_code == 200
    ids = [h["id"] for h in list_resp.json()]
    assert "HZ-TEST" in ids


def test_proximity_check_no_hazards(client):
    resp = client.post("/api/proximity/check", json={"location": {"latitude": 11.34, "longitude": 77.71}})
    assert resp.status_code == 200
    assert resp.json()["nearby"] is False


def test_proximity_check_near_hazard(client):
    client.post("/api/hazards", json={
        "id": "HZ-NEAR", "location": {"latitude": 11.34, "longitude": 77.71},
        "radius_m": 300, "severity": 0.8, "hazard_type": "wildfire", "hard_constraint": False,
    })
    resp = client.post("/api/proximity/check", json={"location": {"latitude": 11.3401, "longitude": 77.7101}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["nearby"] is True
    assert body["hazard_id"] == "HZ-NEAR"


def test_rl_decision_safe_route(client):
    resp = client.post("/api/rl/decision", json={
        "current_location": {"latitude": 11.341, "longitude": 77.717},
        "destination": {"latitude": 11.350, "longitude": 77.725},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("SAFE_ROUTE", "NO_SAFE_ROUTE")
    assert body["decision_source"] in ("RL", "SAFETY_HEURISTIC", "UNAVAILABLE")


def test_rl_decision_no_safe_route_when_standing_in_hard_hazard(client):
    client.post("/api/hazards", json={
        "id": "HZ-HARD", "location": {"latitude": 11.341, "longitude": 77.717},
        "radius_m": 200, "severity": 1.0, "hazard_type": "wildfire", "hard_constraint": True,
    })
    resp = client.post("/api/rl/decision", json={
        "current_location": {"latitude": 11.341, "longitude": 77.717},
        "destination": {"latitude": 11.350, "longitude": 77.725},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "NO_SAFE_ROUTE"
    assert body["requires_human_intervention"] is True


def test_alerts_test_endpoint(client):
    resp = client.post("/api/alerts/test")
    assert resp.status_code == 200
    assert resp.json()["severity"] == "HIGH"


def test_scenarios_endpoint(client):
    resp = client.get("/api/scenarios")
    assert resp.status_code == 200
    assert set(resp.json()["difficulties"]) == {"EASY", "MEDIUM", "HARD", "EXTREME"}


def test_docs_available(client):
    resp = client.get("/docs")
    assert resp.status_code == 200
