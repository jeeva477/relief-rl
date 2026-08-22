"""Route-level hazard analysis bridging Maps output and the safety layer."""

from __future__ import annotations

from backend.app.schemas.hazard import HazardOut
from backend.app.schemas.route import RouteResponse
from backend.app.services.geofence_service import haversine_distance_m


def analyze_route_safety(route: RouteResponse, hazards: list[HazardOut]) -> tuple[float, bool, str | None]:
    """Return (soft_risk, hard_violation, reason) for a Maps route.

    Every returned polyline coordinate is checked. A hard hazard is a veto;
    soft exposure is bounded and can be used as a routing signal.
    """
    if not route.segments:
        return 1.0, True, "Route contains no traversable segments."

    hard_hits: list[str] = []
    soft_risk = 0.0

    for segment in route.segments:
        points = segment.coordinates or [segment.start, segment.end]
        for point in points:
            for hazard in hazards:
                if not hazard.active:
                    continue
                distance = haversine_distance_m(
                    point.latitude, point.longitude,
                    hazard.location.latitude, hazard.location.longitude,
                )
                if distance <= hazard.radius_m:
                    if hazard.hard_constraint:
                        hard_hits.append(hazard.id)
                    else:
                        soft_risk = max(soft_risk, hazard.severity)
                elif distance <= hazard.radius_m * 2.0:
                    falloff = 1.0 - ((distance - hazard.radius_m) / hazard.radius_m)
                    soft_risk = max(soft_risk, hazard.severity * falloff)

    if hard_hits:
        ids = ", ".join(sorted(set(hard_hits)))
        return 1.0, True, f"Route intersects hard-constraint hazard zone(s): {ids}."

    traffic_risk = max((s.traffic_factor for s in route.segments), default=0.0)
    soft_risk = min(1.0, max(soft_risk, 0.6 * soft_risk + 0.4 * traffic_risk))
    return float(soft_risk), False, None
