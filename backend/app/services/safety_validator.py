"""
Deterministic hard-safety validation layer (Section 15).

This is the FINAL authority on whether a candidate action/route is
allowed to be presented to the user. The RL policy is only ever an
*optimizer* proposing candidates; this module is what actually decides
safety, and it is intentionally simple, non-learned, and auditable.

Pipeline:
    candidate action
        -> is the target road segment valid / not blocked?
        -> does it enter a confirmed hard-constraint hazard zone?
        -> is it otherwise a prohibited/impossible transition?
        -> ALLOWED / REJECTED

If every candidate the RL (or a baseline) proposes is rejected, the
validator reports NO_SAFE_ROUTE with requires_human_intervention=True.
It never silently substitutes a dangerous action.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.schemas.hazard import HazardOut
from backend.app.services.geofence_service import CircleHazardZone, distance_to_hazard, is_inside_hazard


@dataclass
class CandidateRoute:
    id: str
    distance_m: float
    duration_s: float
    hazard_exposure: float  # aggregate soft risk score, 0..1
    passes_through_hard_hazard: bool
    blocked: bool = False


@dataclass
class ValidationResult:
    allowed: bool
    reason: str | None = None
    safety_score: float = 1.0  # 1.0 = fully safe, 0.0 = rejected


def validate_candidate(route: CandidateRoute) -> ValidationResult:
    """Deterministically validate a single candidate route/action."""
    if route.blocked:
        return ValidationResult(allowed=False, reason="Road segment is blocked/impassable.", safety_score=0.0)
    if route.passes_through_hard_hazard:
        return ValidationResult(
            allowed=False,
            reason="Route intersects a confirmed hard-constraint hazard zone.",
            safety_score=0.0,
        )
    # Soft hazard exposure still factors into a safety score even for
    # allowed routes, so the caller can rank multiple allowed candidates.
    safety_score = max(0.0, 1.0 - route.hazard_exposure)
    return ValidationResult(allowed=True, reason=None, safety_score=safety_score)


def select_safest_route(candidates: list[CandidateRoute]) -> tuple[CandidateRoute | None, ValidationResult]:
    """
    Validate all candidates and return the safest ALLOWED one (highest
    safety_score, tie-broken by shortest duration). Returns (None, result)
    with requires_human_intervention semantics left to the caller if no
    candidate is allowed.
    """
    best: tuple[CandidateRoute, ValidationResult] | None = None
    for route in candidates:
        result = validate_candidate(route)
        if not result.allowed:
            continue
        if best is None or (
            result.safety_score > best[1].safety_score
            or (result.safety_score == best[1].safety_score and route.duration_s < best[0].duration_s)
        ):
            best = (route, result)

    if best is None:
        return None, ValidationResult(allowed=False, reason="No candidate route satisfies hard safety constraints.")
    return best


def hazards_to_hard_zones(hazards: list[HazardOut]) -> list[CircleHazardZone]:
    return [
        CircleHazardZone(
            center_lat=h.location.latitude,
            center_lon=h.location.longitude,
            radius_m=h.radius_m,
            severity=h.severity,
            hazard_id=h.id,
            hard_constraint=h.hard_constraint,
        )
        for h in hazards
    ]


def point_violates_hard_hazard(lat: float, lon: float, hazards: list[HazardOut]) -> HazardOut | None:
    """Returns the first hard-constraint hazard the point falls inside, if any."""
    for h in hazards:
        if not h.hard_constraint or not h.active:
            continue
        zone = CircleHazardZone(h.location.latitude, h.location.longitude, h.radius_m, h.severity, h.id, True)
        if is_inside_hazard(lat, lon, zone):
            return h
    return None
