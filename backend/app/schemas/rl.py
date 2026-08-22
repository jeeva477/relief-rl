from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from backend.app.schemas.location import LatLng


class RLDecisionRequest(BaseModel):
    current_location: LatLng
    destination: LatLng


class RLDecisionResponse(BaseModel):
    status: Literal["SAFE_ROUTE", "NO_SAFE_ROUTE", "SERVICE_DEGRADED"]
    action: str | None = None
    safety_score: float | None = None
    risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"] | None = None
    remaining_distance_m: float | None = None
    estimated_time_s: float | None = None
    hazard_level: float | None = None
    route: list[dict] = []
    decision_source: Literal["RL", "SAFETY_HEURISTIC", "UNAVAILABLE"] = "UNAVAILABLE"
    model_name: str | None = None
    model_version: str | None = None
    requires_human_intervention: bool = False
    message: str | None = None
