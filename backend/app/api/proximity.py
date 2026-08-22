from __future__ import annotations

import time

from fastapi import APIRouter

from backend.app.config import get_settings
from backend.app.core.dependencies import get_hazards
from backend.app.schemas.alerts import ProximityCheckRequest, ProximityCheckResponse
from backend.app.services.geofence_service import get_risk_level, haversine_distance_m

router = APIRouter()

# hazard_id -> last alert timestamp, for cooldown/deduplication (Section 21).
_last_alert_ts: dict[str, float] = {}


def _recommended_action(severity_label: str) -> str:
    return {
        "CRITICAL": "EVACUATE",
        "HIGH": "EVACUATE",
        "MODERATE": "MONITOR",
        "LOW": "NONE",
    }[severity_label]


@router.post("/api/proximity/check", response_model=ProximityCheckResponse)
def check_proximity(req: ProximityCheckRequest):
    settings = get_settings()
    hazards = get_hazards().list_active()

    nearest = None
    nearest_distance = float("inf")
    for h in hazards:
        d = haversine_distance_m(req.location.latitude, req.location.longitude,
                                  h.location.latitude, h.location.longitude) - h.radius_m
        if d < nearest_distance:
            nearest_distance = d
            nearest = h

    if nearest is None:
        return ProximityCheckResponse(nearby=False)

    severity_label = get_risk_level(nearest_distance, nearest.severity, settings.alert_distance_high_m)
    is_nearby = nearest_distance <= settings.alert_distance_moderate_m

    if not is_nearby:
        return ProximityCheckResponse(nearby=False)

    now = time.time()
    last = _last_alert_ts.get(nearest.id, 0.0)
    if now - last < settings.alert_cooldown_s:
        # Deduplicated: still report the state truthfully, but callers can
        # use this to decide not to re-notify the user.
        pass
    _last_alert_ts[nearest.id] = now

    return ProximityCheckResponse(
        nearby=True,
        severity=severity_label,
        distance_m=max(0.0, nearest_distance),
        hazard_id=nearest.id,
        recommended_action=_recommended_action(severity_label),
    )
