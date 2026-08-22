import math

from backend.app.services.geofence_service import (
    CircleHazardZone,
    distance_to_hazard,
    get_risk_level,
    haversine_distance_m,
    is_inside_hazard,
)


def test_haversine_zero_distance_for_identical_points():
    assert haversine_distance_m(11.34, 77.71, 11.34, 77.71) == 0.0


def test_haversine_known_distance_approx():
    # London to Paris is approximately 344 km.
    d = haversine_distance_m(51.5074, -0.1278, 48.8566, 2.3522)
    assert 330_000 < d < 360_000


def test_is_inside_hazard_true_at_center():
    zone = CircleHazardZone(center_lat=11.34, center_lon=77.71, radius_m=500, severity=0.5, hazard_id="h1")
    assert is_inside_hazard(11.34, 77.71, zone)


def test_is_inside_hazard_false_far_away():
    zone = CircleHazardZone(center_lat=11.34, center_lon=77.71, radius_m=500, severity=0.5, hazard_id="h1")
    assert not is_inside_hazard(11.50, 77.90, zone)


def test_distance_to_hazard_negative_when_inside():
    zone = CircleHazardZone(center_lat=11.34, center_lon=77.71, radius_m=1000, severity=0.5, hazard_id="h1")
    d = distance_to_hazard(11.34, 77.71, zone)
    assert d < 0


def test_risk_level_classification():
    assert get_risk_level(distance_m=-10, severity=0.5) == "CRITICAL"
    assert get_risk_level(distance_m=100, severity=0.7, near_threshold_m=500) == "HIGH"
    assert get_risk_level(distance_m=100, severity=0.2, near_threshold_m=500) == "MODERATE"
    assert get_risk_level(distance_m=10_000, severity=0.9, near_threshold_m=500) == "LOW"
