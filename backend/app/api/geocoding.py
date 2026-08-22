from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.core.dependencies import get_maps_provider
from backend.app.schemas.location import LatLng

router = APIRouter()


class GeocodeRequest(BaseModel):
    address: str


class ReverseGeocodeRequest(BaseModel):
    location: LatLng


@router.post("/api/geocode", response_model=LatLng)
async def geocode(req: GeocodeRequest):
    provider = get_maps_provider()
    try:
        return await provider.geocode(req.address)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Geocoding failed: {exc}")


@router.post("/api/reverse-geocode")
async def reverse_geocode(req: ReverseGeocodeRequest):
    provider = get_maps_provider()
    try:
        address = await provider.reverse_geocode(req.location)
        return {"address": address}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Reverse geocoding failed: {exc}")
