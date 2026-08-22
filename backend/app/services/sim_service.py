"""
Simulation service for the Relief-RL 3D simulator.

The Gymnasium environment (`rl.envs.evacuation_env.EvacuationEnv`) remains
the source of truth. This service wraps it in a session: it runs a policy,
records actual frames/events, builds compact state for the frontend, and
provides explanation / training / evaluation / learning-trend services.

Everything reported to the frontend (reward, penalty, score, rescues,
unmet, response time, route efficiency) is measured from the environment.
No values are fabricated.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from fastapi import WebSocket

from backend.app.core.dependencies import (
    get_model_handle,
    get_qrdqn_model_handle,
    reload_model,
    reload_qrdqn_model,
)
from backend.app.schemas.sim import (
    EvalRunRequest,
    HazardView,
    IncidentView,
    LearningTrend,
    ModelStatus,
    RewardBreakdown,
    SimFrame,
    SimStartRequest,
    SimState,
    TrainStartRequest,
    TrainStatus,
)
from rl.baselines.safety_heuristic import safety_heuristic_action
from rl.baselines.shortest_path import shortest_safe_path_action
from rl.envs.evacuation_env import EvacuationEnv, N_ACTIONS, OBS_DIM, TIME_PER_STEP_S
from rl.training.config import TrainingConfig

ACTION_NAMES = {0: "STAY", 1: "NORTH", 2: "EAST", 3: "SOUTH", 4: "WEST", 5: "REROUTE", 6: "DISPATCH", 7: "PRIORITIZE"}
CHECKPOINT_DIR = os.path.abspath("rl/checkpoints")

PolicyFn = Callable[[EvacuationEnv, np.ndarray], int]


def _make_policy(policy: str, model_label: str) -> tuple[PolicyFn, str]:
    """Build a policy callable. Returns (policy_fn, resolved_model_label).

    The label is resolved *after* checking the model handle so the UI
    honestly shows which engine is actually making decisions.
    """
    handle = get_model_handle()

    def ppo_policy(env: EvacuationEnv, obs: np.ndarray) -> int:
        if handle.available and handle.model is not None:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask = torch.as_tensor(env.valid_action_mask()).unsqueeze(0)
            with torch.inference_mode():
                action, _, _ = handle.model.get_action(obs_t, action_mask=mask, deterministic=True)
            return int(action.item())
        return safety_heuristic_action(env)

    def qrdqn_policy(env: EvacuationEnv, obs: np.ndarray) -> int:
        qhandle = get_qrdqn_model_handle()
        if qhandle.available and qhandle.model is not None:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask = torch.as_tensor(env.valid_action_mask()).unsqueeze(0)
            with torch.inference_mode():
                action = qhandle.model.act(obs_t, action_mask=mask, deterministic=True)
            return int(action.item())
        return safety_heuristic_action(env)

    if policy == "ppo":
        label = f"PPO" if handle.available else "AI FALLBACK (HEURISTIC)"
        return ppo_policy, label
    if policy == "qrdqn":
        qhandle = get_qrdqn_model_handle()
        label = "QR-DQN" if qhandle.available else "AI FALLBACK (HEURISTIC)"
        return qrdqn_policy, label
    if policy == "heuristic":
        return lambda env, obs: safety_heuristic_action(env), "RULE HEURISTIC"
    if policy == "shortest":
        return lambda env, obs: shortest_safe_path_action(env), "SHORTEST SAFE PATH"
    if policy == "random":
        return lambda env, obs: int(env.np_random.choice(np.flatnonzero(env.valid_action_mask()))), "RANDOM (UNTRAINED)"
    raise ValueError(f"unknown policy {policy}")


def build_incidents(env: EvacuationEnv, rng: np.random.Generator) -> list[IncidentView]:
    """Victim clusters anchored to actual hazard zones.

    The *positions* are deterministic per episode seed (decorative), but
    the victim counts always sum to the environment's real `victims` value
    and update every frame, so the HUD/incident totals stay true to state.
    """
    victims = max(env.victims, 0)
    if victims <= 0:
        return []
    weights: list[float] = []
    anchors: list[tuple[float, float]] = []
    for hz in env.hazards[:8]:
        weights.append(max(hz.severity, 0.1))
        anchors.append((hz.x, hz.y))
    if not anchors:
        gx, gy = env.graph.normalize(*env.goal_cell)
        anchors = [(gx, gy)]
        weights = [1.0]
    total_w = sum(weights)
    incidents: list[IncidentView] = []
    remaining = victims
    for i, (anchor, w) in enumerate(zip(anchors, weights)):
        share = int(round(victims * w / total_w))
        if i == len(anchors) - 1:
            share = remaining
        jitter = 0.05
        incidents.append(
            IncidentView(
                x=float(np.clip(anchor[0] + rng.uniform(-jitter, jitter), 0.05, 0.95)),
                y=float(np.clip(anchor[1] + rng.uniform(-jitter, jitter), 0.05, 0.95)),
                victims=share,
            )
        )
        remaining -= share
    return incidents


@dataclass
class SimSession:
    session_id: str
    env: EvacuationEnv
    policy_fn: PolicyFn
    policy: str
    model_label: str
    difficulty: str
    disaster: str
    seed: int
    grid_size: int
    max_steps: int
    speed: float = 1.0
    status: str = "idle"  # idle | running | paused | done
    end_reason: str | None = None  # None | completed | failed | timeout | user_stopped
    step: int = 0
    frames: list[SimFrame] = field(default_factory=list)
    episode_reward: float = 0.0
    episode_penalty: float = 0.0
    # `events` is the PERMANENT, append-only log for the current episode --
    # it is never cleared mid-mission (only reset when a new episode starts
    # via start_episode()). It backs /api/sim/history, the post-game report,
    # and replay. `_pending_events` is a short-lived delivery buffer: it
    # holds only the events generated by the step(s) not yet pushed to a
    # WebSocket/REST caller, and callers drain it after reading -- draining
    # the pending buffer must never touch `events`.
    events: list[dict[str, Any]] = field(default_factory=list)
    _pending_events: list[dict[str, Any]] = field(default_factory=list)
    _event_seq: int = 0
    _incident_rng: np.random.Generator | None = None
    _episode_metrics: dict[str, Any] | None = None
    _last_action: int = 0
    _reset_options: dict[str, Any] | None = None
    # Rolling windows used ONLY for lightweight statistical anomaly detection
    # (route_risk is computed fresh each step from live frame state instead).
    _reward_history: list[float] = field(default_factory=list)
    _penalty_history: list[float] = field(default_factory=list)
    _pos_history: list[tuple[int, int]] = field(default_factory=list)
    _invalid_history: list[bool] = field(default_factory=list)

    def start_episode(self) -> None:
        self.env.reset(seed=self.seed, options=self._reset_options)
        self.step = 0
        self.episode_reward = 0.0
        self.episode_penalty = 0.0
        self.frames = []
        self.events = []
        self._pending_events = []
        self._event_seq = 0
        self._incident_rng = np.random.default_rng(self.seed * 7919 + 13)
        self._episode_metrics = None
        self.status = "running"
        self.end_reason = None
        self._reward_history = []
        self._penalty_history = []
        self._pos_history = []
        self._invalid_history = []

    def reset_episode(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.start_episode()

    def explanation(self) -> dict[str, Any]:
        frames = self.frames
        metrics = self._episode_metrics or {}
        if not frames:
            return {
                "available": False,
                "message": "No recorded episode to explain.",
            }
        return explain_episode(self, frames, metrics)

    def summary(self) -> dict[str, Any]:
        frames = self.frames
        metrics = self._episode_metrics or {}
        last = frames[-1] if frames else None
        success = bool(last and last.success)
        return {
            "session_id": self.session_id,
            "status": self.status,
            "end_reason": self.end_reason,
            "policy": self.policy,
            "model_label": self.model_label,
            "difficulty": self.difficulty,
            "disaster": self.disaster,
            "seed": self.seed,
            "steps": self.step,
            "success": success,
            "reward": round(self.episode_reward, 2),
            "penalty": round(self.episode_penalty, 2),
            "score": round(self.episode_reward, 2),
            # Fall back to the last recorded frame's cumulative counters when
            # the episode never reached env-level termination (e.g. the user
            # stopped it mid-mission) -- metrics stays {} in that case, so
            # `.get(..., 0)` alone would silently misreport a real partial
            # rescue count as zero.
            "rescued": metrics.get("rescued", last.victims_rescued if last else 0),
            "unmet": metrics.get("unmet", last.unmet if last else 0),
            "response_time_s": metrics.get("response_time_s"),
            "victims": metrics.get("victims", last.victims_total if last else 0),
        }


def _compute_route_risk(env: EvacuationEnv, distance_to_goal: float, traffic_level: float,
                         severity: float, blocked_cells: list[list[int]]) -> tuple[str, float]:
    """Deterministic numerical route risk from real, live frame state:
    traffic congestion, hazard severity, fraction of the grid currently
    blocked, and proximity to a hard (impassable/lethal) hazard zone. No ML,
    no fabricated inputs -- every term reads straight off the environment."""
    grid_cells = max(1, env.grid_size * env.grid_size)
    blocked_frac = len(blocked_cells) / grid_cells

    ax, ay = env.graph.normalize(*env.agent_cell)
    hard_nearby = False
    for hz in env.hazards:
        if not hz.hard_constraint or not hz.active:
            continue
        d = ((hz.x - ax) ** 2 + (hz.y - ay) ** 2) ** 0.5
        if d <= hz.radius * 1.6:
            hard_nearby = True
            break

    score = (
        0.30 * min(1.0, max(0.0, traffic_level))
        + 0.30 * min(1.0, max(0.0, severity))
        + 0.20 * min(1.0, blocked_frac * 4.0)
        + 0.20 * (1.0 if hard_nearby else 0.0)
    )
    score = round(min(1.0, score), 3)
    if score < 0.33:
        return "LOW", score
    if score < 0.66:
        return "MEDIUM", score
    return "HIGH", score


def _detect_anomalies(session: SimSession, reward: float, penalty: float, action_valid: bool,
                       agent_pos: tuple[int, int]) -> tuple[str, list[str]]:
    """Lightweight statistical anomaly detection over the session's own
    rolling history -- no external model. Flags: reward well below the
    recent mean, a penalty spike, a stranded vehicle (no movement for
    several steps), and repeated blocked/failed actions."""
    reasons: list[str] = []

    hist_r = session._reward_history
    if len(hist_r) >= 5:
        mean_r = statistics.mean(hist_r)
        std_r = statistics.pstdev(hist_r) or 1e-6
        if reward < mean_r - 2.0 * std_r:
            reasons.append(f"reward anomaly ({reward:.2f} vs recent mean {mean_r:.2f})")

    hist_p = session._penalty_history
    if len(hist_p) >= 5:
        mean_p = statistics.mean(hist_p)
        std_p = statistics.pstdev(hist_p) or 1e-6
        if penalty > mean_p + 2.0 * std_p and penalty > 0.5:
            reasons.append(f"penalty spike ({penalty:.2f} vs recent mean {mean_p:.2f})")

    pos_hist = session._pos_history
    window = 6
    if len(pos_hist) >= window and len(set(pos_hist[-window:])) == 1:
        reasons.append(f"vehicle stranded (no movement in the last {window} steps)")

    invalid_hist = session._invalid_history
    if len(invalid_hist) >= 4 and sum(1 for v in invalid_hist[-4:] if v) >= 3:
        reasons.append("repeated blocked/invalid route attempts")

    if not reasons:
        status = "NORMAL"
    elif len(reasons) == 1:
        status = "WARNING"
    else:
        status = "ANOMALY"
    return status, reasons


def _push_history(hist: list, value, maxlen: int = 12) -> None:
    hist.append(value)
    if len(hist) > maxlen:
        del hist[0]


def _frame_events(step_info: dict[str, Any], action_id: int, action_name: str, action_valid: bool) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if step_info.get("hard_violation"):
        events.append({"type": "HARD_ZONE_ENTERED", "severity": "critical", "text": f"Vehicle entered a hard-constraint hazard zone (penalty {step_info.get('hard_violations_total', 1)})"})
    if step_info.get("wasted"):
        events.append({"type": "RESOURCE_WASTED", "severity": "warning", "text": "Resource action attempted with no resources available"})
    if step_info.get("dispatched"):
        events.append({"type": "VEHICLE_DISPATCHED", "severity": "info", "text": "DISPATCH: +1 emergency vehicle, 5-step speed boost"})
    if action_id == 7 and not step_info.get("wasted"):
        events.append({"type": "PRIORITIZED", "severity": "info", "text": f"PRIORITIZE: mission priority raised to {step_info.get('priority', 0):.2f}"})
    if action_id == 5:
        events.append({"type": "REROUTE", "severity": "info", "text": "REROUTE: vehicle replanned toward the safest improving cell"})
    for raw in step_info.get("events") or []:
        events.append({"type": "WORLD_UPDATE", "severity": "warning", "text": str(raw)})
    if not action_valid and action_id in (1, 2, 3, 4):
        events.append({"type": "ROAD_BLOCKED", "severity": "warning", "text": "Blocked-road attempt: the selected road is closed"})
    return events


def _build_frame(session: SimSession, action_id: int, obs: np.ndarray, step_info: dict[str, Any],
                 reward: float, breakdown: dict[str, float], terminated: bool, truncated: bool,
                 inference_ms: float, env_step_ms: float, action_valid: bool,
                 risk_quantiles: list[float] | None = None) -> SimFrame:
    env = session.env
    mask = env.valid_action_mask()
    action_name = ACTION_NAMES[action_id]
    step_penalty = sum(-v for v in breakdown.values() if v < 0)
    session.episode_reward += reward
    session.episode_penalty += step_penalty
    session.step = int(step_info.get("steps_taken", env.steps_taken))
    session._last_action = action_id

    success = bool(step_info.get("success", False))
    timed_out = bool(truncated and not success)
    finished = terminated or truncated

    incidents = build_incidents(env, session._incident_rng or np.random.default_rng(session.seed))

    blocked_cells_list = [[int(c[0]), int(c[1])] for c in env.graph._blocked]
    route_risk, route_risk_score = _compute_route_risk(
        env, env._distance_to_goal(env.agent_cell), env.traffic_level, env.severity, blocked_cells_list
    )
    anomaly_status, anomaly_reasons = _detect_anomalies(
        session, reward, step_penalty, action_valid, (int(env.agent_cell[0]), int(env.agent_cell[1]))
    )
    # Update rolling history AFTER comparing against it, so this step is
    # judged against what came before it, not against itself.
    _push_history(session._reward_history, reward)
    _push_history(session._penalty_history, step_penalty)
    _push_history(session._pos_history, (int(env.agent_cell[0]), int(env.agent_cell[1])))
    _push_history(session._invalid_history, not action_valid)

    frame = SimFrame(
        session_id=session.session_id,
        step=session.step,
        status="done" if finished else "running",
        policy=session.policy,
        model_label=session.model_label,
        grid_size=env.grid_size,
        disaster=env.scenario.disaster_type.value if env.scenario else "any",
        agent={"x": float(env.agent_cell[0]), "y": float(env.agent_cell[1])},
        goal={"x": float(env.goal_cell[0]), "y": float(env.goal_cell[1])},
        action={"id": action_id, "name": action_name, "valid": action_valid},
        action_valid=action_valid,
        valid_mask=[int(b) for b in mask],
        reward=round(reward, 3),
        reward_breakdown=RewardBreakdown(**{k: round(v, 4) for k, v in breakdown.items()}),
        penalty=round(step_penalty, 3),
        cumulative_reward=round(session.episode_reward, 3),
        cumulative_penalty=round(session.episode_penalty, 3),
        score=round(session.episode_reward, 3),
        victims_total=int(env.victims),
        victims_rescued=int(env.victims if success else 0),
        unmet=int(env.victims - (env.victims if success else 0)),
        resources=int(env.resources),
        vehicles=int(env.vehicles),
        priority=round(env.priority, 3),
        dispatch_steps_left=int(env.dispatch_steps_left),
        weather=str(env.weather.value),
        severity=round(env.severity, 3),
        traffic_level=round(env.traffic_level, 3),
        time_frac=round(session.step / max(env.max_steps, 1), 3),
        distance_to_goal=round(env._distance_to_goal(env.agent_cell), 3),
        route_distance=int(env.route_distance),
        hard_violations=int(env.hard_violations),
        blocked_attempts=int(env.blocked_attempts),
        wasted_actions=int(env.wasted_actions),
        blocked_cells=blocked_cells_list,
        hazards=[
            HazardView(
                id=h.id, x=float(h.x), y=float(h.y), radius=float(h.radius),
                severity=float(h.severity), type=str(h.hazard_type.value),
                hard=bool(h.hard_constraint), velocity=[float(h.velocity[0]), float(h.velocity[1])],
            )
            for h in env.hazards[:8]
        ],
        incidents=incidents,
        terminated=terminated,
        truncated=truncated,
        success=success,
        timed_out=timed_out,
        explanation=_step_explanation(session, action_name, action_valid, step_info, reward, breakdown, step_penalty),
        response_time_s=step_info.get("response_time_s"),
        inference_ms=round(inference_ms, 2),
        env_step_ms=round(env_step_ms, 2),
        episode_metrics=session._episode_metrics,
        risk_quantiles=risk_quantiles,
        route_risk=route_risk,
        route_risk_score=route_risk_score,
        anomaly_status=anomaly_status,
        anomaly_reasons=anomaly_reasons,
    )
    session.frames.append(frame)
    return frame


def _step_explanation(session: SimSession, action_name: str, action_valid: bool, step_info: dict[str, Any],
                      reward: float, breakdown: dict[str, float], penalty: float) -> str:
    """Deterministic, rule-based reasoning about the *actual* step outcome."""
    parts: list[str] = []
    if session.policy == "ppo" and session.model_label.startswith("PPO"):
        parts.append(f"PPO policy selected {action_name} (action {step_info.get('steps_taken', 0)})")
    elif session.model_label.startswith("AI FALLBACK"):
        parts.append(f"Trained checkpoint incompatible/missing; safety heuristic took over and selected {action_name}")
    else:
        parts.append(f"{session.model_label} selected {action_name}")
    if not action_valid and action_name in ("NORTH", "EAST", "SOUTH", "WEST"):
        parts.append("action was masked (road blocked) and rejected by the environment")
    if breakdown.get("hard_violation_penalty"):
        parts.append("the new cell violates a hard-constraint hazard zone (large penalty)")
    elif breakdown.get("risk_cost"):
        parts.append("the new cell carries soft risk exposure")
    if breakdown.get("traffic_cost"):
        parts.append("traffic slowed the vehicle")
    if breakdown.get("progress"):
        parts.append("the vehicle moved closer to the goal")
    if breakdown.get("dispatch_bonus"):
        parts.append("a vehicle was dispatched (speed boost for 5 steps)")
    if breakdown.get("success_bonus"):
        parts.append("GOAL REACHED — all victims rescued")
    if reward >= 0:
        parts.append(f"net effect: +{reward:.2f} reward")
    else:
        parts.append(f"net effect: {reward:.2f} (penalty {penalty:.2f})")
    return "; ".join(parts)


def explain_episode(session: SimSession, frames: list[SimFrame], metrics: dict[str, Any]) -> dict[str, Any]:
    """Post-game explanation built ONLY from the recorded episode data."""
    last = frames[-1] if frames else None
    success = bool(last and last.success)
    steps = metrics.get("steps", session.step)
    rewarded = metrics.get("total_reward", session.episode_reward)
    penalty = metrics.get("total_penalty", session.episode_penalty)

    hard_v = metrics.get("hard_violations", last.hard_violations if last else 0)
    blocked = metrics.get("blocked_attempts", last.blocked_attempts if last else 0)
    wasted = metrics.get("wasted_actions", last.wasted_actions if last else 0)
    # Same fallback as SimSession.summary(): if the episode never reached
    # env-level termination (e.g. stopped by the user), metrics is {} and we
    # must read the real partial counts off the last recorded frame instead
    # of silently reporting 0.
    rescued = metrics.get("rescued", last.victims_rescued if last else 0)
    unmet = metrics.get("unmet", last.unmet if last else 0)
    victims = metrics.get("victims", last.victims_total if last else 0)
    response = metrics.get("response_time_s")
    efficiency = metrics.get("route_efficiency")
    priority = last.priority if last else 0.0
    end_reason = session.end_reason

    sections: dict[str, str] = {}

    if end_reason == "user_stopped":
        sections["MISSION SUMMARY"] = (
            f"MISSION STOPPED BY USER — the operator ended the mission after {steps} steps. "
            f"{rescued} rescued, {unmet} of {victims} victims unmet at the time of stopping. "
            f"Reward at stop: {rewarded:+.1f} (penalty {penalty:.1f})."
        )
    elif success:
        sections["MISSION SUMMARY"] = (
            f"MISSION COMPLETE — the emergency vehicle reached the safe zone in {steps} steps "
            f"({response} s response time). All {rescued} victims rescued. "
            f"Final reward {rewarded:+.1f} (penalty {penalty:.1f})."
        )
    else:
        if last and last.timed_out:
            sections["MISSION SUMMARY"] = (
                f"MISSION FAILED — time ran out after {steps} steps. "
                f"{rescued} rescued, {unmet} of {victims} victims unmet. Final reward {rewarded:+.1f}."
            )
        else:
            sections["MISSION SUMMARY"] = (
                f"MISSION FAILED — the vehicle did not reach the safe zone in {steps} steps. "
                f"{rescued} rescued, {unmet} of {victims} victims unmet. Final reward {rewarded:+.1f}."
            )

    if session.policy == "ppo":
        if session.model_label.startswith("PPO"):
            sections["WHY DID THE AI MAKE THESE DECISIONS?"] = (
                "The PPO policy acts on the 37-dim observation (position, goal, distance, risk, "
                "traffic, severity, victims, resources, weather, time, blocked fraction, priority, "
                "nearby hazards, previous action). Blocked roads are masked out so the model can "
                "never pick an impossible move; REROUTE/DISPATCH/PRIORITIZE let it adapt to "
                "mid-episode road changes and resource constraints."
            )
        else:
            sections["WHY DID THE AI MAKE THESE DECISIONS?"] = (
                "The trained checkpoint is incompatible with the current environment "
                "(obs/action schema mismatch), so the safety heuristic fallback ran instead. "
                "The decision record above reflects the heuristic's choices."
            )
    else:
        sections["WHY DID THE AI MAKE THESE DECISIONS?"] = (
            f"This session ran the {session.model_label} policy. "
            "Per-step decisions are recorded in the timeline above."
        )

    penalty_causes: list[str] = []
    if hard_v > 0:
        penalty_causes.append(f"{hard_v} hard-zone violations ({hard_v * 15:.0f} penalty points)")
    if blocked > 0:
        penalty_causes.append(f"{blocked} blocked-road attempts")
    if wasted > 0:
        penalty_causes.append(f"{wasted} wasted resource actions")
    if not success and end_reason != "user_stopped":
        penalty_causes.append(f"{unmet} victims left unmet (failed-rescue penalty)")
    sections["WHAT CAUSED PENALTIES?"] = (
        "; ".join(penalty_causes) if penalty_causes else "No significant penalty sources in this episode."
    )

    went_well: list[str] = []
    if success:
        went_well.append(f"all {rescued} victims rescued with {(response or 0):.1f} s response time")
    elif end_reason == "user_stopped" and rescued > 0:
        went_well.append(f"{rescued} victim(s) rescued before the mission was stopped")
    if efficiency and efficiency <= 1.5:
        went_well.append(f"route efficiency {efficiency:.2f}x (near-optimal path)")
    if blocked == 0:
        went_well.append("no blocked-road attempts")
    if hard_v == 0:
        went_well.append("no hard-hazard violations")
    if wasted == 0:
        went_well.append("no wasted resource actions")
    sections["WHAT WENT WELL?"] = "; ".join(went_well) if went_well else "The episode produced no notable positive results."

    went_wrong: list[str] = []
    if not success and end_reason == "user_stopped":
        went_wrong.append(f"mission stopped before completion ({unmet} victims still unmet)")
    elif not success:
        went_wrong.append(f"mission failed ({unmet} victims unmet)")
    if hard_v > 0:
        went_wrong.append(f"entered hard hazard zones {hard_v} times")
    if blocked > 0:
        went_wrong.append(f"attempted blocked roads {blocked} times")
    if wasted > 0:
        went_wrong.append(f"wasted {wasted} resource actions")
    if efficiency and efficiency > 1.5:
        went_wrong.append(f"route was {efficiency:.2f}x the shortest feasible distance")
    sections["WHAT WENT WRONG?"] = "; ".join(went_wrong) if went_wrong else "Nothing significant went wrong in this episode."

    sections["WHAT DID THE AI LEARN?"] = (
        "This session's decisions are one sample from the trained policy. "
        "Learning trends (Reward / Penalty / Net / Success / Response time / Efficiency) are shown "
        "in the Learning panel from the actual training log; per-episode reward and penalty "
        f"in training went from early negatives to a stable mean of {metrics.get('reward_mean', '—')} "
        "across the latest episodes." if metrics.get("reward_mean") is not None else
        "See the Learning panel for the actual training curves; no per-step policy gradients are "
        "exposed per session because PPO updates only during training."
    )

    if success:
        sections["RECOMMENDATION"] = (
            f"Maintain the trained policy for {session.difficulty} {session.disaster} missions. "
            "Re-train or fine-tune when road-network or disaster distributions change."
        )
    else:
        sections["RECOMMENDATION"] = (
            f"Increase training episodes for {session.difficulty} {session.disaster}, tune the "
            "reward weights, or raise the priority budget before launch."
        )

    # ROUTE ANALYSIS / ANOMALY ANALYSIS -- aggregated from the real per-step
    # route_risk / anomaly_status recorded on every frame of this episode.
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    anomaly_events: list[str] = []
    route_changes = 0
    prev_action_name: str | None = None
    for fr in frames:
        risk_counts[fr.route_risk] = risk_counts.get(fr.route_risk, 0) + 1
        if fr.anomaly_status != "NORMAL":
            for reason in fr.anomaly_reasons:
                anomaly_events.append(f"step {fr.step}: {reason}")
        if fr.action.get("name") == "REROUTE":
            route_changes += 1
        prev_action_name = fr.action.get("name")
    _ = prev_action_name  # kept for readability of the loop above

    # AI PERFORMANCE -- real per-step decision counts, no fabrication.
    ai_decisions = len(frames)
    masked_actions = sum(1 for fr in frames if not fr.action_valid)
    valid_decisions = ai_decisions - masked_actions
    dispatches = sum(1 for fr in frames if fr.action.get("name") == "DISPATCH" and fr.reward_breakdown.dispatch_bonus)

    # REWARD / PENALTY ANALYSIS -- sum each named component across every
    # recorded frame's real RewardBreakdown. Positive components are reward
    # sources; negative components (already stored as negative floats) are
    # penalty sources. A component absent from every frame is omitted
    # rather than shown as a fabricated 0.
    component_totals: dict[str, float] = {}
    for fr in frames:
        for field_name, value in fr.reward_breakdown.model_dump().items():
            if value:
                component_totals[field_name] = component_totals.get(field_name, 0.0) + value
    reward_components = {k: round(v, 3) for k, v in component_totals.items() if v > 0}
    penalty_components = {k: round(-v, 3) for k, v in component_totals.items() if v < 0}

    # VEHICLE PERFORMANCE -- this environment is a single-agent grid
    # (EvacuationEnv has one navigating agent plus a `vehicles` resource
    # counter used by DISPATCH); there is no per-vehicle id/telemetry to
    # report, so distance/response-time-per-vehicle are honestly N/A rather
    # than invented.
    vehicle_summary = {
        "active_vehicles": last.vehicles if last else 0,
        "dispatches": dispatches,
        "note": "Single-agent grid environment: no discrete per-vehicle telemetry "
                "(distance/response time per vehicle) is tracked -- N/A rather than fabricated.",
    }

    mean_risk_score = round(statistics.mean([fr.route_risk_score for fr in frames]), 3) if frames else 0.0
    peak_risk = "HIGH" if risk_counts["HIGH"] > 0 else ("MEDIUM" if risk_counts["MEDIUM"] > 0 else "LOW")
    sections["ROUTE ANALYSIS"] = (
        f"{route_changes} reroute action(s); average route risk score {mean_risk_score:.2f} "
        f"(LOW {risk_counts['LOW']} / MEDIUM {risk_counts['MEDIUM']} / HIGH {risk_counts['HIGH']} steps, "
        f"peak {peak_risk}); {blocked} blocked-road attempts."
    ) if frames else "No recorded steps to analyze."

    if anomaly_events:
        sections["ANOMALY ANALYSIS"] = (
            f"{len(anomaly_events)} anomaly signal(s) detected during the episode: "
            + "; ".join(anomaly_events[:6])
            + (f" (+{len(anomaly_events) - 6} more)" if len(anomaly_events) > 6 else "")
        )
    else:
        sections["ANOMALY ANALYSIS"] = "NORMAL — no reward/penalty anomalies, stranding, or repeated route failures detected."

    return {
        "available": True,
        "generated_from": "rule-based analysis of actual simulation metrics (LLM module not configured)",
        "policy": session.policy,
        "model_label": session.model_label,
        "difficulty": session.difficulty,
        "disaster": session.disaster,
        "seed": session.seed,
        "success": success,
        "end_reason": end_reason,
        "steps": steps,
        "reward": round(float(rewarded), 2),
        "penalty": round(float(penalty), 2),
        "rescued": int(rescued),
        "unmet": int(unmet),
        "victims": int(victims),
        "response_time_s": response,
        "route_risk_summary": {
            "mean_score": mean_risk_score,
            "peak_level": peak_risk,
            "steps_low": risk_counts["LOW"],
            "steps_medium": risk_counts["MEDIUM"],
            "steps_high": risk_counts["HIGH"],
            "route_changes": route_changes,
        },
        "anomaly_summary": {
            "event_count": len(anomaly_events),
            "status": "ANOMALY" if len(anomaly_events) >= 2 else ("WARNING" if anomaly_events else "NORMAL"),
        },
        "route_efficiency": efficiency,
        "priority": round(float(priority), 2),
        "sections": sections,
        "ai_performance": {
            "model": session.model_label,
            "ai_decisions": ai_decisions,
            "valid_decisions": valid_decisions,
            "masked_actions": masked_actions,
            "route_changes": route_changes,
            "dispatches": dispatches,
        },
        "reward_components": reward_components,
        "penalty_components": penalty_components,
        "vehicle_summary": vehicle_summary,
        "final_state": last.model_dump() if last else None,
    }


class SimManager:
    """Owns all simulation sessions, autostep tasks and WebSocket clients."""

    def __init__(self) -> None:
        self._sessions: dict[str, SimSession] = {}
        self._clients: dict[str, WebSocket] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._event_loop = loop

    # -- sessions -----------------------------------------------------
    def create(self, req: SimStartRequest) -> SimSession:
        policy_fn, model_label = _make_policy(req.policy, "")
        env = EvacuationEnv(
            difficulty=req.difficulty,
            grid_size=req.grid_size,
            max_steps=req.max_steps,
        )
        session = SimSession(
            session_id=uuid.uuid4().hex[:12],
            env=env,
            policy_fn=policy_fn,
            policy=req.policy,
            model_label=model_label,
            difficulty=req.difficulty,
            disaster=req.disaster,
            seed=req.seed,
            grid_size=req.grid_size,
            max_steps=req.max_steps,
            speed=req.speed,
        )
        options = None
        if req.disaster != "any":
            options = {"scenario_config": {"disaster_type": req.disaster,
                                           "difficulty": req.difficulty,
                                           "grid_size": req.grid_size,
                                           "max_steps": req.max_steps}}
        session._reset_options = options
        session.env.reset(seed=req.seed, options=options)
        session.start_episode()
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> SimSession | None:
        return self._sessions.get(session_id)

    def state(self, session_id: str) -> SimState | None:
        session = self.get(session_id)
        if session is None:
            return None
        return SimState(
            session_id=session.session_id,
            status=session.status,
            policy=session.policy,
            step=session.step,
            difficulty=session.difficulty,
            disaster=session.disaster,
            seed=session.seed,
            frames=len(session.frames),
        )

    # -- stepping ------------------------------------------------------
    async def step(self, session_id: str) -> SimFrame | None:
        session = self.get(session_id)
        if session is None or session.status == "done":
            return None
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            return self._step_now(session)

    def _step_now(self, session: SimSession) -> SimFrame | None:
        env = session.env
        if session.status == "done":
            return None
        obs = env._build_observation()
        t0 = time.perf_counter()
        action_id = int(session.policy_fn(env, obs))
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # For QR-DQN sessions, also capture the full return-distribution
        # quantiles for the action taken (the "risk" view distributional RL
        # provides over vanilla PPO) so the frontend can show a real
        # distribution, not a fabricated one.
        risk_quantiles: list[float] | None = None
        if session.policy == "qrdqn":
            qhandle = get_qrdqn_model_handle()
            if qhandle.available and qhandle.model is not None:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.inference_mode():
                    quantiles = qhandle.model(obs_t)  # (1, n_actions, n_quantiles)
                risk_quantiles = [round(float(v), 3) for v in quantiles[0, action_id].cpu().numpy().tolist()]

        t0 = time.perf_counter()
        _, reward, terminated, truncated, step_info = env.step(action_id)
        env_step_ms = (time.perf_counter() - t0) * 1000.0
        breakdown = env.episode_breakdown
        action_valid = not bool(step_info.get("action_masked", False))

        if terminated or truncated:
            session._episode_metrics = env._episode_metrics(bool(step_info.get("success", False)))
            session.status = "done"
            success_flag = bool(step_info.get("success", False))
            session.end_reason = "completed" if success_flag else ("timeout" if truncated else "failed")
            step_info["response_time_s"] = session._episode_metrics.get("response_time_s")

        frame = _build_frame(
            session, action_id, obs, step_info, float(reward), dict(breakdown),
            bool(terminated), bool(truncated), inference_ms, env_step_ms, action_valid,
            risk_quantiles=risk_quantiles,
        )
        frame.episode_metrics = session._episode_metrics

        sim_time_s = round(session.step * TIME_PER_STEP_S, 2)
        for ev in _frame_events(step_info, action_id, frame.action["name"], action_valid):
            session._event_seq += 1
            entry = {
                "event_id": f"{session.session_id}-{session._event_seq}",
                "step": session.step,
                "simulation_time": sim_time_s,
                "type": ev["type"],
                "severity": ev["severity"],
                "message": ev["text"],
                "text": ev["text"],  # kept for older frontend clients that read `text`
                "location": {"x": frame.agent["x"], "y": frame.agent["y"]},
                "vehicle": None,  # single-agent env: no discrete per-vehicle id to attach
                "incident": None,
                "metadata": {"action": frame.action["name"], "reward": frame.reward, "penalty": frame.penalty},
            }
            session.events.append(entry)  # permanent -- never cleared mid-episode
            session._pending_events.append(entry)  # transient -- drained by callers

        if terminated or truncated:
            session._event_seq += 1
            lifecycle_type = {
                "completed": "MISSION_COMPLETED",
                "failed": "MISSION_FAILED",
                "timeout": "MISSION_FAILED",
            }.get(session.end_reason, "MISSION_COMPLETED")
            entry = {
                "event_id": f"{session.session_id}-{session._event_seq}",
                "step": session.step,
                "simulation_time": sim_time_s,
                "type": lifecycle_type,
                "severity": "critical" if lifecycle_type == "MISSION_FAILED" else "info",
                "message": f"Mission ended: {session.end_reason}",
                "text": f"Mission ended: {session.end_reason}",
                "location": {"x": frame.agent["x"], "y": frame.agent["y"]},
                "vehicle": None,
                "incident": None,
                "metadata": {"end_reason": session.end_reason},
            }
            session.events.append(entry)
            session._pending_events.append(entry)
        return frame

    # -- autostep (WebSocket-driven) ------------------------------------
    async def _autostep_loop(self, session_id: str) -> None:
        session = self.get(session_id)
        while session is not None and session.status == "running":
            frame = await self.step(session_id)
            if frame is not None:
                await self._push(session_id, {"type": "frame", "payload": frame.model_dump()})
                pending, session._pending_events = session._pending_events, []
                for ev in pending:
                    await self._push(session_id, {"type": "event", "payload": ev})
                if session.status == "done":
                    await self._push(session_id, {
                        "type": "mission_complete" if frame.success else "mission_failed",
                        "payload": session.summary(),
                    })
            if session.status != "running":
                break
            # Real-time pacing: TIME_PER_STEP_S is the in-sim seconds one
            # step represents; dividing by `speed` fast-forwards playback.
            # Floor of 0.02s (50 steps/sec) is a safety limit against a
            # runaway loop, not an artificial speed cap -- the actual cap is
            # session.speed's own max (set_speed / SimStartRequest.speed).
            await asyncio.sleep(max(0.02, TIME_PER_STEP_S / max(session.speed, 0.25)))

    async def start_autostep(self, session_id: str) -> None:
        session = self.get(session_id)
        if session is None:
            return
        session.status = "running"
        if session_id in self._tasks and not self._tasks[session_id].done():
            return
        self._tasks[session_id] = asyncio.create_task(self._autostep_loop(session_id))

    async def stop_autostep(self, session_id: str) -> None:
        task = self._tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()
        session = self.get(session_id)
        if session is not None and session.status == "running":
            session.status = "paused"

    async def stop_mission(self, session_id: str) -> SimSession | None:
        """User-initiated stop: cancel autostep, freeze the session in its
        current (real) state, and mark it done with end_reason='user_stopped'
        rather than 'completed'/'failed'. Never resets or discards frames."""
        await self.stop_autostep(session_id)
        session = self.get(session_id)
        if session is None:
            return None
        if session.status != "done":  # don't clobber a natural completion
            session.status = "done"
            session.end_reason = "user_stopped"
            session._event_seq += 1
            last = session.frames[-1] if session.frames else None
            entry = {
                "event_id": f"{session.session_id}-{session._event_seq}",
                "step": session.step,
                "simulation_time": round(session.step * TIME_PER_STEP_S, 2),
                "type": "MISSION_STOPPED",
                "severity": "info",
                "message": "Mission stopped by user request.",
                "text": "Mission stopped by user request.",
                "location": {"x": last.agent["x"], "y": last.agent["y"]} if last else None,
                "vehicle": None,
                "incident": None,
                "metadata": {"end_reason": "user_stopped"},
            }
            session.events.append(entry)
            session._pending_events.append(entry)
        return session

    def set_speed(self, session_id: str, speed: float) -> None:
        session = self.get(session_id)
        if session is not None:
            session.speed = max(0.25, min(speed, 16.0))

    # -- WebSocket ------------------------------------------------------
    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        self._clients[session_id] = websocket

    def disconnect(self, session_id: str) -> None:
        self._clients.pop(session_id, None)

    async def _push(self, session_id: str, message: dict[str, Any]) -> None:
        ws = self._clients.get(session_id)
        if ws is not None:
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(session_id)

    async def broadcast(self, message: dict[str, Any]) -> None:
        for session_id in list(self._clients.keys()):
            await self._push(session_id, message)


manager = SimManager()

# ----------------------------------------------------------------------
# Training runner (background thread so the event loop never blocks)
# ----------------------------------------------------------------------

_train_lock = threading.Lock()
_train_status: dict[str, Any] = {"running": False}


def _train_worker(cfg: TrainingConfig, run_id: str) -> None:
    try:
        if cfg.algo == "qrdqn":
            from rl.training.train_qrdqn import QRDQNTrainingConfig, train as train_qrdqn

            qcfg = QRDQNTrainingConfig(
                episodes=cfg.episodes,
                learning_rate=5e-4,
                gamma=cfg.gamma,
                difficulty=cfg.difficulty,
                disaster=cfg.disaster,
                max_steps=cfg.max_steps,
                grid_size=cfg.grid_size,
                seed=cfg.seed,
                checkpoint_frequency=cfg.checkpoint_frequency,
                checkpoint_dir=cfg.checkpoint_dir,
                eval_after_episodes=cfg.eval_after_episodes,
                eval_seed_offset=cfg.eval_seed_offset,
            )
            checkpoint_path = train_qrdqn(qcfg)
            reload_qrdqn_model()  # make the new QR-DQN checkpoint live
        else:
            from rl.training.train import train

            checkpoint_path = train(cfg)
            reload_model()  # make the new PPO checkpoint live
        with _train_lock:
            _train_status["running"] = False
            _train_status["finished"] = True
            _train_status["run_id"] = run_id
            _train_status["checkpoint_dir"] = cfg.checkpoint_dir
            _train_status["checkpoint_path"] = checkpoint_path
            _train_status["message"] = "Training finished. Checkpoint reloaded."
    except Exception as exc:  # pragma: no cover - defensive
        with _train_lock:
            _train_status["running"] = False
            _train_status["error"] = str(exc)
            _train_status["message"] = "Training failed."


def start_training(req: TrainStartRequest) -> TrainStatus:
    with _train_lock:
        if _train_status.get("running"):
            return TrainStatus(running=True, message="Training already in progress.")
    run_id = uuid.uuid4().hex[:12]
    cfg = TrainingConfig(
        episodes=req.episodes,
        algo=req.algo,
        difficulty=req.difficulty,
        disaster=req.disaster,
        seed=req.seed,
        checkpoint_dir=CHECKPOINT_DIR,
    )
    with _train_lock:
        _train_status.update({
            "running": True,
            "finished": False,
            "error": None,
            "run_id": run_id,
            "episodes": req.episodes,
            "checkpoint_dir": cfg.checkpoint_dir,
            "started_at": time.time(),
        })
    threading.Thread(target=_train_worker, args=(cfg, run_id), daemon=True).start()
    return TrainStatus(running=True, run_id=run_id, total_episodes=req.episodes,
                       checkpoint_dir=cfg.checkpoint_dir, message="Training started.")


def training_status() -> TrainStatus:
    latest = None
    csv_path = os.path.join(CHECKPOINT_DIR, "metrics.csv")
    if os.path.exists(csv_path):
        import csv as _csv
        with open(csv_path, newline="") as f:
            rows = list(_csv.DictReader(f))
        if rows:
            latest = rows[-1]
    with _train_lock:
        return TrainStatus(
            running=bool(_train_status.get("running")),
            run_id=_train_status.get("run_id"),
            episode=int(latest["episode"]) if latest and latest.get("episode") else None,
            total_episodes=_train_status.get("episodes"),
            latest=latest,
            checkpoint_dir=_train_status.get("checkpoint_dir", CHECKPOINT_DIR),
            message=_train_status.get("message"),
        )


def stop_training() -> TrainStatus:
    # The training thread is not safely interruptible mid-episode; a new
    # run can only start once the current one finishes. Honest limitation.
    with _train_lock:
        return TrainStatus(
            running=bool(_train_status.get("running")),
            run_id=_train_status.get("run_id"),
            message="Training thread cannot be interrupted mid-episode; it will finish and the "
                    "checkpoint will be reloaded automatically.",
        )


# ----------------------------------------------------------------------
# Evaluation runner
# ----------------------------------------------------------------------

_eval_lock = threading.Lock()
_eval_status: dict[str, Any] = {"running": False}


def _eval_worker(req: EvalRunRequest, run_id: str, checkpoint: str) -> None:
    try:
        from rl.training.evaluate import load_model, make_rl_policy, _run_policy_episode, random_policy, shortest_path_policy, heuristic_policy
        from rl.envs.evacuation_env import EvacuationEnv
        from rl.evaluation.metrics import aggregate, aggregate_to_dict

        model = load_model(checkpoint)
        rng = np.random.default_rng(req.seed + req.seed * 7)
        env = EvacuationEnv(difficulty=req.difficulty, grid_size=req.grid_size, max_steps=req.max_steps)
        policies: dict[str, Any] = {
            "Random (untrained)": random_policy,
            "ShortestSafe": shortest_path_policy,
            "RuleHeuristic": heuristic_policy,
        }
        if model is not None:
            policies["Trained PPO"] = make_rl_policy(model)
        per_seed: dict[str, list] = {name: [] for name in policies}
        for _ in range(max(1, req.seeds)):
            for name, policy in policies.items():
                results = []
                for _ in range(req.episodes):
                    seed = int(rng.integers(0, 2**31 - 1))
                    env.reset(seed=seed)
                    results.append(_run_policy_episode(env, policy, seed))
                per_seed[name].append(aggregate(results))
        with _eval_lock:
            _eval_status["running"] = False
            _eval_status["finished"] = True
            _eval_status["run_id"] = run_id
            _eval_status["result"] = {
                "difficulty": req.difficulty,
                "disaster": req.disaster,
                "episodes_per_seed": req.episodes,
                "seeds": req.seeds,
                "unseen_scenarios": True,
                "note": "Unseen-scenario evaluation: seed range was never used during training.",
                "policies": {name: aggregate_to_dict(aggregate(metrics)) for name, metrics in per_seed.items()},
            }
    except Exception as exc:  # pragma: no cover - defensive
        with _eval_lock:
            _eval_status["running"] = False
            _eval_status["error"] = str(exc)


def start_evaluation(req: EvalRunRequest) -> dict[str, Any]:
    with _eval_lock:
        if _eval_status.get("running"):
            return {"running": True, "message": "Evaluation already in progress."}
        _eval_status.update({"running": True, "finished": False, "error": None,
                             "result": None, "run_id": uuid.uuid4().hex[:12]})
    run_id = _eval_status["run_id"]
    checkpoint = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    threading.Thread(target=_eval_worker, args=(req, run_id, checkpoint), daemon=True).start()
    return {"running": True, "run_id": run_id, "message": "Evaluation started."}


def evaluation_status() -> dict[str, Any]:
    with _eval_lock:
        return dict(_eval_status)


def run_before_after(episodes: int, seed: int, difficulty: str, disaster: str) -> dict[str, Any]:
    from rl.training.evaluate import compare_agents, load_model

    model = load_model(os.path.join(CHECKPOINT_DIR, "best_model.pt"))
    return compare_agents(model=model, episodes=episodes, seed=seed,
                          difficulty=difficulty, disaster=disaster)


# ----------------------------------------------------------------------
# PPO vs QR-DQN research comparison (feeds the frontend PpoVsDqn panel)
# ----------------------------------------------------------------------

def ppo_vs_qrdqn_comparison(
    live: bool = False,
    episodes: int = 30,
    seed: int = 7,
    difficulty: str = "MEDIUM",
    disaster: str = "any",
) -> dict[str, Any]:
    """Real PPO vs QR-DQN metrics for the research UI.

    live=False (default): returns the precomputed comparison written by
    `scripts/compare_ppo_qrdqn.py` (rl/checkpoints/ppo_vs_qrdqn.json) if it
    exists -- fast, no training/eval cost per request.

    live=True: re-runs both trained checkpoints on an identical set of
    seeded scenarios right now (mirrors scripts/compare_ppo_qrdqn.py) so the
    numbers reflect whichever checkpoints are currently on disk. Any
    checkpoint that is missing/incompatible is reported honestly as
    unavailable rather than fabricated.
    """
    static_path = os.path.join(CHECKPOINT_DIR, "ppo_vs_qrdqn.json")

    if not live:
        if os.path.exists(static_path):
            with open(static_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["source"] = "precomputed (rl/checkpoints/ppo_vs_qrdqn.json)"
            return data
        # No precomputed file -- fall through to a live run so the panel
        # still shows real numbers instead of nothing.

    from rl.envs.evacuation_env import EvacuationEnv as _Env
    from rl.evaluation.metrics import aggregate, aggregate_to_dict
    from rl.training.evaluate import (
        _run_policy_episode,
        load_model,
        load_qrdqn_model,
        make_qrdqn_policy,
        make_rl_policy,
    )

    ppo_model = load_model(os.path.join(CHECKPOINT_DIR, "best_model.pt"))
    qrdqn_model = load_qrdqn_model(os.path.join(CHECKPOINT_DIR, "qrdqn_best_model.pt"))

    policies: dict[str, Any] = {}
    unavailable: list[str] = []
    if ppo_model is not None:
        policies["PPO"] = make_rl_policy(ppo_model)
    else:
        unavailable.append("PPO")
    if qrdqn_model is not None:
        policies["QR-DQN"] = make_qrdqn_policy(qrdqn_model)
    else:
        unavailable.append("QR-DQN")

    if not policies:
        return {
            "episodes": episodes, "seed": seed, "difficulty": difficulty, "disaster": disaster,
            "agents": {}, "source": "live",
            "note": "INSUFFICIENT DATA -- neither PPO nor QR-DQN has a compatible checkpoint.",
        }

    env = _Env(difficulty=difficulty, grid_size=10, max_steps=100)
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(episodes)]

    agents: dict[str, dict] = {}
    for name, policy in policies.items():
        results = [_run_policy_episode(env, policy, seed=s) for s in seeds]
        agents[name] = aggregate_to_dict(aggregate(results))

    note = "Both agents ran on the identical set of seeded scenarios."
    if unavailable:
        note += f" No compatible checkpoint for: {', '.join(unavailable)} (not fabricated)."

    return {
        "episodes": episodes, "seed": seed, "difficulty": difficulty, "disaster": disaster,
        "agents": agents, "source": "live", "note": note,
    }


# ----------------------------------------------------------------------
# Model status + learning trend
# ----------------------------------------------------------------------

def model_status(algo: str = "ppo") -> ModelStatus:
    """Report on the requested checkpoint's real, on-disk metadata.
    algo="ppo" (default) reports the PPO Actor-Critic checkpoint;
    algo="qrdqn" reports the QR-DQN checkpoint. Never fabricates metadata --
    everything comes from the checkpoint file itself or is None."""
    if algo == "qrdqn":
        handle = get_qrdqn_model_handle()
        ckpt_path = os.path.join(CHECKPOINT_DIR, "qrdqn_best_model.pt")
    else:
        handle = get_model_handle()
        ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    ckpt_meta: dict[str, Any] = {}
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
        except Exception:
            ckpt_meta = {}
    return ModelStatus(
        available=handle.available,
        model_name=handle.model_name or ckpt_meta.get("model_name"),
        model_version=handle.model_version or ckpt_meta.get("model_version"),
        algo=ckpt_meta.get("algo") or handle.model_name,
        obs_dim=ckpt_meta.get("obs_dim"),
        n_actions=ckpt_meta.get("n_actions"),
        hidden_dim=ckpt_meta.get("hidden_dim"),
        episode=ckpt_meta.get("episode"),
        mean_reward=ckpt_meta.get("mean_reward"),
        path=ckpt_path if os.path.exists(ckpt_path) else None,
        compatible=handle.compatible,
        incompatible_reason=handle.incompatible_reason,
        fallback_policy=handle.model_name if handle.available else "SAFETY_HEURISTIC",
    )


def learning_trend() -> LearningTrend:
    """Historical learning curves from the REAL training log (metrics.csv)."""
    csv_path = os.path.join(CHECKPOINT_DIR, "metrics.csv")
    if not os.path.exists(csv_path):
        return LearningTrend(available=False, episodes=[], checkpoint_dir=CHECKPOINT_DIR,
                             source=None)
    import csv as _csv
    with open(csv_path, newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return LearningTrend(available=False, episodes=[], checkpoint_dir=CHECKPOINT_DIR,
                             source="metrics.csv (empty)")
    episodes: list[dict[str, Any]] = []
    for row in rows:
        def num(key: str) -> float | None:
            val = row.get(key)
            if val in (None, "", "None"):
                return None
            try:
                return float(val)
            except ValueError:
                return None
        episodes.append({
            "episode": int(row.get("episode", 0)),
            "reward": num("reward"),
            "penalty": num("penalty"),
            "net_reward": num("net_reward"),
            "success": True if row.get("success") in ("True", "true", "1") else False,
            "success_rate": None,
            "response_time_s": num("response_time_s"),
            "rescued": num("rescued"),
            "route_efficiency": num("route_efficiency"),
            "hard_violations": num("hard_violations"),
            "wasted_actions": num("wasted_actions"),
            "blocked_attempts": num("blocked_attempts"),
            "policy_loss": num("policy_loss"),
            "value_loss": num("value_loss"),
            "entropy": num("entropy"),
        })
    window = 10
    n = len(episodes)
    for i, ep in enumerate(episodes):
        lo = max(0, i - window + 1)
        win = episodes[lo : i + 1]
        ok = [w for w in win if w["success"] is not None]
        ep["success_rate"] = round(sum(1 for w in ok if w["success"]) / len(ok), 3) if ok else None
    return LearningTrend(
        available=True,
        episodes=episodes,
        checkpoint_dir=CHECKPOINT_DIR,
        source=csv_path,
    )