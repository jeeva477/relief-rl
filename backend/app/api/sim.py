"""
REST + WebSocket API for the Relief-RL 3D simulator.

The Gymnasium environment remains the source of truth; these endpoints
expose thin projections of it (sessions, frames, training, evaluation,
learning trends, explanation) and stream live updates over /ws/sim.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from backend.app.schemas.sim import (
    EvalRunRequest,
    ReplayResponse,
    SimStartRequest,
    SimState,
    TrainStartRequest,
    TrainStatus,
)
from backend.app.services import hf_explanation_service, sim_service
from backend.app.services.sim_service import SimSession, manager

router = APIRouter()


# ----------------------------------------------------------------------
# Session lifecycle (REST)
# ----------------------------------------------------------------------

@router.post("/api/sim/start", response_model=SimState)
async def sim_start(req: SimStartRequest) -> SimState:
    session = manager.create(req)
    state = manager.state(session.session_id)
    if state is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="session could not be created")
    return state


@router.get("/api/sim/state", response_model=SimState)
async def sim_state(session_id: str) -> SimState:
    state = manager.state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown session {session_id}")
    return state


@router.post("/api/sim/step")
async def sim_step(session_id: str):
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session {session_id}")
    frame = await manager.step(session_id)
    if frame is None:
        raise HTTPException(status_code=409, detail="Session is already complete.")
    # Drain only the transient delivery buffer -- `session.events` is the
    # permanent per-episode log and must never be cleared (it backs
    # /api/sim/history, the post-game report, and replay).
    pending, session._pending_events = session._pending_events, []
    return {"frame": frame.model_dump(), "events": pending, "summary": session.summary()}


@router.post("/api/sim/reset", response_model=SimState)
async def sim_reset(session_id: str, seed: int | None = None) -> SimState:
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session {session_id}")
    await manager.stop_autostep(session_id)
    session.reset_episode(seed=seed)
    state = manager.state(session_id)
    if state is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="session state unavailable")
    return state


@router.post("/api/sim/pause", response_model=SimState)
async def sim_pause(session_id: str) -> SimState:
    """Temporary halt (REST fallback). Unlike /stop, the session can be
    resumed and no end_reason/result is set."""
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session {session_id}")
    await manager.stop_autostep(session_id)
    state = manager.state(session_id)
    if state is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="session state unavailable")
    return state


@router.post("/api/sim/resume", response_model=SimState)
async def sim_resume(session_id: str) -> SimState:
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session {session_id}")
    if session.status == "done":
        raise HTTPException(status_code=409, detail="Session is complete; it cannot be resumed. Use /reset.")
    await manager.start_autostep(session_id)
    state = manager.state(session_id)
    if state is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="session state unavailable")
    return state


@router.post("/api/sim/stop")
async def sim_stop(session_id: str):
    """User-initiated mission stop (REST fallback for clients not on the
    WebSocket). Freezes the session's real current state and marks it
    end_reason='user_stopped' -- never fabricates a completion/failure."""
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session {session_id}")
    stopped = await manager.stop_mission(session_id)
    if stopped is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="session could not be stopped")
    return stopped.summary()


@router.get("/api/sim/history", response_model=ReplayResponse)
async def sim_history(session_id: str) -> ReplayResponse:
    """Full replay data for a session: every recorded frame AND the
    permanent event log (never cleared mid-mission -- see SimSession.events).
    Backs the post-game report, the event timeline, and JSON/CSV export."""
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session {session_id}")
    return ReplayResponse(
        session_id=session.session_id, policy=session.policy,
        frames=session.frames, events=session.events,
    )


@router.post("/api/sim/explain")
async def sim_explain(session_id: str, use_hf: bool = True):
    """Post-game / mid-mission explanation.

    Always includes the existing rule-based explanation (computed only from
    recorded episode data -- see `sim_service.explain_episode`). When
    `use_hf=true` (default) and HF_ENABLED is configured with a token, this
    also asks Hugging Face to narrate the SAME structured data as an
    explanation-only layer; on any failure/timeout/disabled config it falls
    back to the rule-based text and reports SOURCE: RULE-BASED FALLBACK
    rather than fabricating an AI narrative. Hugging Face never controls
    the policy, routes, vehicles, or rewards -- see hf_explanation_service.py.
    """
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session {session_id}")
    rule_based = session.explanation()
    if not use_hf:
        result = dict(rule_based)
        result.setdefault("source", "RULE-BASED FALLBACK")
        result.setdefault("narrative", None)
        return result
    return await hf_explanation_service.explain_with_hf(rule_based)


# ----------------------------------------------------------------------
# Training / evaluation / model / learning
# ----------------------------------------------------------------------

@router.post("/api/sim/train", response_model=TrainStatus)
async def sim_train(req: TrainStartRequest) -> TrainStatus:
    return sim_service.start_training(req)


@router.get("/api/sim/train/status", response_model=TrainStatus)
async def sim_train_status() -> TrainStatus:
    return sim_service.training_status()


@router.post("/api/sim/train/stop", response_model=TrainStatus)
async def sim_train_stop() -> TrainStatus:
    return sim_service.stop_training()


@router.post("/api/sim/evaluate")
async def sim_evaluate(req: EvalRunRequest):
    return sim_service.start_evaluation(req)


@router.get("/api/sim/evaluate")
async def sim_evaluate_status():
    return sim_service.evaluation_status()


@router.post("/api/sim/evaluate/before-after")
async def sim_before_after(episodes: int = 20, seed: int = 7, difficulty: str = "MEDIUM", disaster: str = "any"):
    if episodes < 1 or episodes > 100:
        raise HTTPException(status_code=400, detail="episodes must be in [1, 100]")
    return sim_service.run_before_after(episodes, seed, difficulty, disaster)


@router.get("/api/sim/model-status")
async def sim_model_status(algo: str = "ppo"):
    return sim_service.model_status(algo=algo)


@router.get("/api/sim/compare/ppo-vs-qrdqn")
async def sim_ppo_vs_qrdqn(
    live: bool = False,
    episodes: int = 30,
    seed: int = 7,
    difficulty: str = "MEDIUM",
    disaster: str = "any",
):
    """Real PPO vs QR-DQN comparison for the research UI (PpoVsDqn panel).

    Defaults to the precomputed rl/checkpoints/ppo_vs_qrdqn.json for speed;
    pass live=true to re-run both checkpoints on fresh seeded scenarios.
    """
    if episodes < 1 or episodes > 200:
        raise HTTPException(status_code=400, detail="episodes must be in [1, 200]")
    return sim_service.ppo_vs_qrdqn_comparison(
        live=live, episodes=episodes, seed=seed, difficulty=difficulty, disaster=disaster
    )


@router.get("/api/sim/learning")
async def sim_learning():
    return sim_service.learning_trend()


# ----------------------------------------------------------------------
# WebSocket: /ws/sim
# ----------------------------------------------------------------------

async def _handle_message(websocket: WebSocket, session_id: str, message: dict) -> str:
    """Handle one client message. Returns the (possibly new) session id."""
    session: SimSession | None = manager.get(session_id) if session_id and session_id != "pending" else None
    kind = message.get("type")

    if kind == "ping":
        await websocket.send_json({"type": "pong", "payload": {"t": message.get("payload", {}).get("t")}})
        return session_id

    if kind == "start":
        req = SimStartRequest(**message.get("payload", {}))
        if session is None:
            session = manager.create(req)
            session_id = session.session_id
        else:
            session.speed = req.speed
            if req.disaster and req.disaster != "any":
                session._reset_options = {"scenario_config": {
                    "disaster_type": req.disaster,
                    "difficulty": req.difficulty,
                    "grid_size": req.grid_size,
                    "max_steps": req.max_steps}}
            session.reset_episode(seed=req.seed)
        await manager.start_autostep(session.session_id)
        state = manager.state(session.session_id)
        await websocket.send_json({"type": "simulation_started",
                                   "payload": state.model_dump() if state else None})
        return session_id

    if session is None:
        await websocket.send_json({"type": "error", "payload": {"message": "No active session. Send {type:'start'} first."}})
        return session_id

    if kind == "pause":
        await manager.stop_autostep(session_id)
        await websocket.send_json({"type": "state", "payload": {"status": "paused"}})
    elif kind == "resume":
        await manager.start_autostep(session_id)
        await websocket.send_json({"type": "state", "payload": {"status": "running"}})
    elif kind == "step":
        await manager.stop_autostep(session_id)
        frame = await manager.step(session_id)
        if frame is not None:
            await websocket.send_json({"type": "frame", "payload": frame.model_dump()})
            pending, session._pending_events = session._pending_events, []
            for ev in pending:
                await websocket.send_json({"type": "event", "payload": ev})
        if session.status == "done":
            await websocket.send_json(
                {"type": "mission_complete" if (frame and frame.success) else "mission_failed",
                 "payload": session.summary()})
    elif kind == "reset":
        await manager.stop_autostep(session_id)
        payload = message.get("payload", {}) or {}
        session.reset_episode(seed=payload.get("seed"))
        state = manager.state(session_id)
        await websocket.send_json({"type": "state", "payload": state.model_dump() if state else None})
    elif kind == "stop":
        stopped = await manager.stop_mission(session_id)
        if stopped is not None:
            pending, stopped._pending_events = stopped._pending_events, []
            for ev in pending:
                await websocket.send_json({"type": "event", "payload": ev})
            await websocket.send_json({"type": "mission_stopped", "payload": stopped.summary()})
    elif kind == "set_speed":
        manager.set_speed(session_id, float(message.get("payload", {}).get("speed", 1.0)))
        await websocket.send_json({"type": "state", "payload": {"speed": session.speed}})
    else:
        await websocket.send_json({"type": "error", "payload": {"message": f"Unknown message type {kind!r}"}})
    return session_id


@router.websocket("/ws/sim")
async def ws_sim(websocket: WebSocket, session_id: str | None = None):
    """Streams sim events/frames for a session.

    Client commands (JSON): start / pause / resume / step / reset / set_speed / ping
    Server events: connected, simulation_started, state, frame, event,
                   mission_complete, mission_failed, pong, error
    """
    try:
        await websocket.accept()
        if session_id is None:
            session_id = websocket.query_params.get("session_id")
        if not session_id:
            session_id = "pending"
        await manager.connect(session_id, websocket)
        await websocket.send_json({"type": "connected", "payload": {"session_id": session_id}})
        while True:
            message = await websocket.receive_json()
            new_id = await _handle_message(websocket, session_id, message)
            if new_id != session_id:
                manager.disconnect(session_id)
                session_id = new_id
                await manager.connect(session_id, websocket)
    except WebSocketDisconnect:
        pass
    finally:
        if session_id and session_id != "pending":
            await manager.stop_autostep(session_id)
            manager.disconnect(session_id)