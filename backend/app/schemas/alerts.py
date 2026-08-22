from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from backend.app.schemas.location import LatLng


class ProximityCheckRequest(BaseModel):
    location: LatLng


class ProximityCheckResponse(BaseModel):
    nearby: bool
    severity: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"] | None = None
    distance_m: float | None = None
    hazard_id: str | None = None
    recommended_action: str | None = None
