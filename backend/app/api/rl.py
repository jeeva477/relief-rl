from __future__ import annotations

import torch
from fastapi import APIRouter

from backend.app.core.dependencies import get_hazards, get_maps_provider, get_model_handle
from backend.app.schemas.rl import RLDecisionRequest, RLDecisionResponse
from backend.app.services.geofence_service import get_risk_level, haversine_distance_m
from backend.app.services.route_safety import analyze_route_safety
from backend.app.services.safety_validator import point_violates_hard_hazard
from backend.app.services.state_builder import build_observation
from rl.envs.evacuation_env import EvacuationEnv

router = APIRouter()

_ACTION_NAMES = {0: "STAY", 1: "NORTH", 2: "EAST", 3: "SOUTH", 4: "WEST"}


@router.post("/api/rl/decision", response_model=RLDecisionResponse)
async def rl_decision(req: RLDecisionRequest):
    hazards = get_hazards().list_active()

    violated = point_violates_hard_hazard(
        req.current_location.latitude,
        req.current_location.longitude,
        hazards,
    )
    if violated is not None:
        return RLDecisionResponse(
            status="NO_SAFE_ROUTE",
            decision_source="UNAVAILABLE",
            requires_human_intervention=True,
            message=(
                f"Current location is inside a hard-constraint hazard zone ({violated.id}). "
                "Immediate evacuation required; automated routing cannot certify safety here."
            ),
        )

    provider = get_maps_provider()
    route = None
    traffic_factor = 0.0
    try:
        route = await provider.compute_route(req.current_location, req.destination)
        traffic_factor = max((s.traffic_factor for s in route.segments), default=0.0)
    except Exception:
        route = None

    # Build the same fixed-size observation used by the trained Actor-Critic.
    obs = build_observation(
        req.current_location,
        req.destination,
        hazards,
        traffic_factor,
    )

    remaining_distance_m = haversine_distance_m(
        req.current_location.latitude,
        req.current_location.longitude,
        req.destination.latitude,
        req.destination.longitude,
    )

    # A live/mock route is a candidate, so validate the actual returned
    # polyline before exposing it as a certified safe route.
    if route is not None:
        hazard_level, hard_violation, reason = analyze_route_safety(route, hazards)
        if hard_violation:
            return RLDecisionResponse(
                status="NO_SAFE_ROUTE",
                decision_source="UNAVAILABLE",
                requires_human_intervention=True,
                hazard_level=round(hazard_level, 3),
                route=[segment.model_dump() for segment in route.segments],
                message=reason or "Route failed the hard safety validator.",
            )
    else:
        hazard_level = float(obs[5])

    handle = get_model_handle()
    if handle.available:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            action, _, _ = handle.model.get_action(obs_t, deterministic=True)
        action_id = int(action.item())
        decision_source = "RL"
        model_name, model_version = handle.model_name, handle.model_version
    else:
        action_id = 2 if obs[2] >= obs[0] else 4
        decision_source = "SAFETY_HEURISTIC"
        model_name, model_version = None, None

    if hazard_level >= 0.85:
        return RLDecisionResponse(
            status="NO_SAFE_ROUTE",
            decision_source=decision_source,
            requires_human_intervention=True,
            hazard_level=round(hazard_level, 3),
            message="No candidate route satisfies the hard safety threshold near the current location.",
        )

    safety_score = max(0.0, 1.0 - hazard_level)
    if hazard_level < 0.15:
        risk_level = "LOW"
    elif hazard_level < 0.4:
        risk_level = "MODERATE"
    elif hazard_level < 0.85:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    estimated_time_s = remaining_distance_m / 11.0 * (1.0 + traffic_factor)

    return RLDecisionResponse(
        status="SAFE_ROUTE",
        action=_ACTION_NAMES[action_id],
        safety_score=round(safety_score, 3),
        risk_level=risk_level,
        remaining_distance_m=round(remaining_distance_m, 1),
        estimated_time_s=round(estimated_time_s, 1),
        hazard_level=round(hazard_level, 3),
        route=[seg.model_dump() for seg in route.segments] if route is not None else [],
        decision_source=decision_source,
        model_name=model_name,
        model_version=model_version,
    )
