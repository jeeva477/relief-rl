"""
Geofencing utilities (Section 20).

Implements correct great-circle (Haversine) distance for point-to-point
and point-to-circle-hazard checks. Polygon hazard support uses Shapely
when available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from shapely.geometry import Point, Polygon
    _SHAPELY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAPELY_AVAILABLE = False

EARTH_RADIUS_M = 6_371_000.0


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


@dataclass
class CircleHazardZone:
    center_lat: float
    center_lon: float
    radius_m: float
    severity: float
    hazard_id: str
    hard_constraint: bool = False


def distance_to_hazard(lat: float, lon: float, hazard: CircleHazardZone) -> float:
    """Distance (m) from a point to the edge of a circular hazard zone (negative if inside)."""
    center_distance = haversine_distance_m(lat, lon, hazard.center_lat, hazard.center_lon)
    return center_distance - hazard.radius_m


def is_inside_hazard(lat: float, lon: float, hazard: CircleHazardZone) -> bool:
    return distance_to_hazard(lat, lon, hazard) <= 0


def is_inside_polygon_hazard(lat: float, lon: float, polygon_coords: list[tuple[float, float]]) -> bool:
    """polygon_coords: list of (lat, lon) tuples defining the hazard polygon."""
    if not _SHAPELY_AVAILABLE:
        raise RuntimeError("Shapely is required for polygon hazard checks; install shapely.")
    poly = Polygon([(lon, lat) for lat, lon in polygon_coords])
    return poly.contains(Point(lon, lat))


def get_risk_level(distance_m: float, severity: float, near_threshold_m: float = 500.0) -> str:
    """
    Coarse risk classification used for alerts and API responses.
    Distance <= 0 means inside the hazard -> CRITICAL regardless of severity.
    """
    if distance_m <= 0:
        return "CRITICAL"
    if distance_m <= near_threshold_m and severity >= 0.6:
        return "HIGH"
    if distance_m <= near_threshold_m:
        return "MODERATE"
    return "LOW"
