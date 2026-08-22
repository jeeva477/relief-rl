from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.core.dependencies import get_maps_provider
from backend.app.schemas.location import LatLng
from backend.app.schemas.route import RouteRequest, RouteResponse

router = APIRouter()


class RouteMatrixRequest(BaseModel):
    origins: list[LatLng]
    destinations: list[LatLng]


@router.post("/api/route", response_model=RouteResponse)
async def compute_route(req: RouteRequest):
    provider = get_maps_provider()
    try:
        return await provider.compute_route(req.origin, req.destination)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Route computation failed: {exc}")


@router.post("/api/route-matrix")
async def compute_route_matrix(req: RouteMatrixRequest):
    provider = get_maps_provider()
    try:
        entries = await provider.compute_route_matrix(req.origins, req.destinations)
        return {"entries": [vars(e) for e in entries]}
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Route matrix computation failed: {exc}")
