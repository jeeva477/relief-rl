from backend.app.schemas.hazard import HazardOut, HazardType
from backend.app.schemas.location import LatLng
from backend.app.services.safety_validator import (
    CandidateRoute,
    point_violates_hard_hazard,
    select_safest_route,
    validate_candidate,
)


def test_blocked_route_rejected():
    route = CandidateRoute(id="r1", distance_m=100, duration_s=30, hazard_exposure=0.0,
                            passes_through_hard_hazard=False, blocked=True)
    result = validate_candidate(route)
    assert not result.allowed
    assert result.safety_score == 0.0


def test_hard_hazard_route_rejected():
    route = CandidateRoute(id="r1", distance_m=100, duration_s=30, hazard_exposure=0.1,
                            passes_through_hard_hazard=True)
    result = validate_candidate(route)
    assert not result.allowed


def test_safe_route_allowed_with_score():
    route = CandidateRoute(id="r1", distance_m=100, duration_s=30, hazard_exposure=0.2,
                            passes_through_hard_hazard=False)
    result = validate_candidate(route)
    assert result.allowed
    assert result.safety_score == 0.8


def test_select_safest_route_picks_highest_score():
    routes = [
        CandidateRoute(id="risky", distance_m=100, duration_s=20, hazard_exposure=0.6, passes_through_hard_hazard=False),
        CandidateRoute(id="safe", distance_m=150, duration_s=40, hazard_exposure=0.1, passes_through_hard_hazard=False),
        CandidateRoute(id="blocked", distance_m=50, duration_s=10, hazard_exposure=0.0, passes_through_hard_hazard=True),
    ]
    best, result = select_safest_route(routes)
    assert best.id == "safe"
    assert result.allowed


def test_select_safest_route_returns_none_when_all_unsafe():
    routes = [
        CandidateRoute(id="a", distance_m=100, duration_s=20, hazard_exposure=0.0, passes_through_hard_hazard=True),
        CandidateRoute(id="b", distance_m=50, duration_s=10, hazard_exposure=0.0, blocked=True,
                        passes_through_hard_hazard=False),
    ]
    best, result = select_safest_route(routes)
    assert best is None
    assert not result.allowed


def test_point_violates_hard_hazard_detects_containment():
    hazard = HazardOut(
        id="h1", location=LatLng(latitude=11.34, longitude=77.71), radius_m=500, severity=0.9,
        hazard_type=HazardType.WILDFIRE, hard_constraint=True, active=True,
    )
    violated = point_violates_hard_hazard(11.34, 77.71, [hazard])
    assert violated is not None
    assert violated.id == "h1"


def test_point_violates_hard_hazard_ignores_soft_hazards():
    hazard = HazardOut(
        id="h1", location=LatLng(latitude=11.34, longitude=77.71), radius_m=500, severity=0.9,
        hazard_type=HazardType.WILDFIRE, hard_constraint=False, active=True,
    )
    violated = point_violates_hard_hazard(11.34, 77.71, [hazard])
    assert violated is None
