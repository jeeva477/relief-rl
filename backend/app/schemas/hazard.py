from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from backend.app.schemas.location import LatLng


class HazardType(str, Enum):
    FLOOD = "flood"
    WILDFIRE = "wildfire"
    EARTHQUAKE = "earthquake"
    CHEMICAL_LEAK = "chemical_leak"
    LANDSLIDE = "landslide"
    STORM = "storm"
    GENERIC = "generic_emergency_zone"


class HazardIn(BaseModel):
    id: str
    location: LatLng
    radius_m: float = Field(..., gt=0)
    severity: float = Field(..., ge=0.0, le=1.0)
    hazard_type: HazardType = HazardType.GENERIC
    hard_constraint: bool = False
    source: str = "manual"  # manual | government_feed | external_api | simulation


class HazardOut(HazardIn):
    active: bool = True
