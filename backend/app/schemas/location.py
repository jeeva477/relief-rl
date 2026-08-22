from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class LatLng(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)

    @field_validator("latitude", "longitude")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v:  # NaN check
            raise ValueError("coordinate must be a finite number")
        return v
