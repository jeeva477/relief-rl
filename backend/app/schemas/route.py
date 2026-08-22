from __future__ import annotations

from pydantic import BaseModel

from backend.app.schemas.location import LatLng


class RouteRequest(BaseModel):
    origin: LatLng
    destination: LatLng


class RouteSegment(BaseModel):
    """
    Internal, normalized representation of a route leg. The RL/state-builder
    code depends only on this schema, never on Google's raw response shape,
    so the system is testable without live Google API access (Section 17).
    """
    start: LatLng
    end: LatLng
    distance_m: float
    duration_s: float
    traffic_factor: float  # 0.0 (free flow) .. 1.0 (gridlock)
    coordinates: list[LatLng] = []
    risk: float = 0.0
    blocked: bool = False


class RouteResponse(BaseModel):
    segments: list[RouteSegment]
    total_distance_m: float
    total_duration_s: float
    source: str  # "google_routes_api" | "mock_provider"
