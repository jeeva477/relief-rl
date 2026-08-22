"""Smoke tests for the simulation API + WebSocket (Phases 4, 23, 24)."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app.main import app
from backend.app.services import sim_service

client = TestClient(app)


def _start_session(policy: str = "heuristic", difficulty: str = "EASY") -> dict:
    resp = client.post("/api/sim/start", json={
        "policy": policy,
        "difficulty": difficulty,
        "disaster": "flood",
        "seed": 7,
        "grid_size": 10,
        "max_steps": 100,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_start_session_and_get_state() -> None:
    state = _start_session()
    assert state["status"] in ("running", "idle")
    assert state["difficulty"] == "EASY"
    assert state["seed"] == 7
    sid = state["session_id"]
    got = client.get(f"/api/sim/state?session_id={sid}")
    assert got.status_code == 200
    assert got.json()["session_id"] == sid


def test_invalid_session_returns_404() -> None:
    assert client.get("/api/sim/state?session_id=doesnotexist").status_code == 404
    assert client.post("/api/sim/step?session_id=doesnotexist").status_code == 404
    assert client.get("/api/sim/history?session_id=doesnotexist").status_code == 404


def test_step_produces_frame_with_reward_penalty_and_events() -> None:
    state = _start_session()
    sid = state["session_id"]
    resp = client.post(f"/api/sim/step?session_id={sid}")
    assert resp.status_code == 200
    body = resp.json()
    frame = body["frame"]
    assert frame["step"] >= 1
    assert frame["session_id"] == sid
    assert frame["reward"] is not None
    assert frame["penalty"] >= 0
    assert isinstance(frame["reward_breakdown"], dict)
    assert "agent" in frame and "goal" in frame
    assert "hazards" in frame and "blocked_cells" in frame and "incidents" in frame
    assert isinstance(frame["valid_mask"], list) and len(frame["valid_mask"]) == 8
    assert frame["action"]["name"] in {"STAY", "NORTH", "EAST", "SOUTH", "WEST", "REROUTE", "DISPATCH", "PRIORITIZE"}
    assert isinstance(body["summary"], dict)


def test_run_session_to_completion_and_replay() -> None:
    state = _start_session(policy="shortest", difficulty="EASY")
    sid = state["session_id"]
    guard = 0
    done = False
    while guard < 500:
        resp = client.post(f"/api/sim/step?session_id={sid}")
        assert resp.status_code == 200
        body = resp.json()
        if body["summary"]["status"] == "done":
            done = True
            break
        guard += 1
    assert done, "session should terminate within 500 steps on EASY"
    history = client.get(f"/api/sim/history?session_id={sid}")
    assert history.status_code == 200
    frames = history.json()["frames"]
    assert len(frames) > 1
    assert frames[-1]["status"] == "done"


def test_events_log_persists_across_steps_and_is_not_cleared() -> None:
    """Regression test for the events-log bug: session.events must be a
    permanent, append-only log for the episode. Reading it via /history
    (or draining the per-call `events` field) must never erase history that
    a later request/report/replay needs."""
    state = _start_session(policy="heuristic", difficulty="EASY")
    sid = state["session_id"]

    seen_event_ids: set[str] = set()
    for _ in range(10):
        resp = client.post(f"/api/sim/step?session_id={sid}")
        assert resp.status_code == 200
        for ev in resp.json()["events"]:
            seen_event_ids.add(ev["event_id"])
        if resp.json()["summary"]["status"] == "done":
            break

    history = client.get(f"/api/sim/history?session_id={sid}")
    assert history.status_code == 200
    logged_events = history.json()["events"]

    # Every event ever delivered to a caller must still be present in the
    # permanent log -- this fails if any code path resets session.events.
    logged_ids = {ev["event_id"] for ev in logged_events}
    assert seen_event_ids.issubset(logged_ids), (
        "events delivered via /step are missing from the permanent events "
        "log -- session.events was cleared somewhere instead of only "
        "draining the transient _pending_events buffer"
    )
    for ev in logged_events:
        assert {"event_id", "step", "simulation_time", "type", "severity", "message"} <= ev.keys()


def test_reset_starts_new_episode() -> None:
    state = _start_session()
    sid = state["session_id"]
    client.post(f"/api/sim/step?session_id={sid}")
    reset = client.post(f"/api/sim/reset?session_id={sid}")
    assert reset.status_code == 200
    assert reset.json()["step"] == 0
    assert reset.json()["status"] in ("running", "idle")


def test_learning_trend_reflects_real_training_log() -> None:
    trend = client.get("/api/sim/learning")
    assert trend.status_code == 200
    data = trend.json()
    assert "available" in data
    if data["available"]:
        assert data["episodes"]
        assert "reward" in data["episodes"][0]
        assert "penalty" in data["episodes"][0]


def test_model_status_endpoint() -> None:
    status = client.get("/api/sim/model-status")
    assert status.status_code == 200
    body = status.json()
    assert "available" in body and "compatible" in body and "fallback_policy" in body


def test_training_status_endpoint() -> None:
    status = client.get("/api/sim/train/status")
    assert status.status_code == 200
    assert "running" in status.json()


def test_explain_returns_rule_based_analysis() -> None:
    state = _start_session()
    sid = state["session_id"]
    for _ in range(3):
        client.post(f"/api/sim/step?session_id={sid}")
    resp = client.post(f"/api/sim/explain?session_id={sid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert "sections" in body
    assert "MISSION SUMMARY" in body["sections"]


def test_before_after_evaluation_runs() -> None:
    resp = client.post("/api/sim/evaluate/before-after?episodes=3&seed=7&difficulty=EASY")
    assert resp.status_code == 200
    body = resp.json()
    assert "policies" in body
    for name, metrics in body["policies"].items():
        assert "mean_reward" in metrics
        assert "success_rate" in metrics


def test_websocket_streams_frames_and_mission_result() -> None:
    with client.websocket_connect("/ws/sim") as ws:
        ws.send_json({"type": "start", "payload": {
            "policy": "shortest", "difficulty": "EASY", "disaster": "flood",
            "seed": 3, "grid_size": 10, "max_steps": 100, "speed": 4.0,
        }})
        first = ws.receive_json()
        assert first["type"] == "connected"
        started = ws.receive_json()
        assert started["type"] == "simulation_started"
        sid = started["payload"]["session_id"]
        assert sid

        seen_frame = False
        seen_event = False
        mission_done = False
        for _ in range(400):
            msg = ws.receive_json()
            t = msg["type"]
            if t == "frame":
                seen_frame = True
            elif t == "event":
                seen_event = True
            elif t in ("mission_complete", "mission_failed"):
                mission_done = True
                break
            elif t == "error":
                pytest.fail(f"server error: {msg}")
        assert seen_frame
        assert mission_done

        # Replay via REST after the run.
        history = client.get(f"/api/sim/history?session_id={sid}")
        assert history.status_code == 200
        assert len(history.json()["frames"]) > 1


def test_websocket_ping_pong_and_reconnect() -> None:
    with client.websocket_connect("/ws/sim?session_id=pingtest") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json({"type": "ping", "payload": {"t": 12345}})
        msg = ws.receive_json()
        assert msg["type"] == "pong"
        assert msg["payload"]["t"] == 12345

    # Reconnect should succeed without leaking connections.
    with client.websocket_connect("/ws/sim?session_id=pingtest") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json({"type": "ping", "payload": {"t": 1}})
        assert ws.receive_json()["type"] == "pong"


def test_websocket_unknown_message_gets_error() -> None:
    with client.websocket_connect("/ws/sim?session_id=unknown_test") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json({"type": "bogus"})
        msg = ws.receive_json()
        assert msg["type"] == "error"