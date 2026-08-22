from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from backend.app.auth import require_admin
from backend.app.core.dependencies import get_hazards
from backend.app.schemas.hazard import HazardIn, HazardOut

router = APIRouter()

@router.get("/api/hazards", response_model=list[HazardOut])
def list_hazards():
    return get_hazards().list_active()

@router.get("/api/admin/hazards", response_model=list[HazardOut], dependencies=[Depends(require_admin)])
def list_all_hazards():
    return get_hazards().list_all()

@router.post("/api/hazards", response_model=HazardOut)
def create_hazard(hazard: HazardIn):
    return get_hazards().add(HazardOut(**hazard.model_dump(), active=True))

@router.post("/api/admin/hazards", response_model=HazardOut, dependencies=[Depends(require_admin)])
def create_admin_hazard(hazard: HazardIn):
    return get_hazards().add(HazardOut(**hazard.model_dump(), active=True))

@router.put("/api/admin/hazards/{hazard_id}", response_model=HazardOut, dependencies=[Depends(require_admin)])
def update_hazard(hazard_id: str, hazard: HazardIn):
    updated = get_hazards().update(hazard_id, HazardOut(**hazard.model_dump(), active=True))
    if updated is None: raise HTTPException(status_code=404, detail="Hazard not found")
    return updated

@router.post("/api/admin/hazards/{hazard_id}/deactivate", dependencies=[Depends(require_admin)])
def deactivate_hazard(hazard_id: str):
    if not get_hazards().deactivate(hazard_id): raise HTTPException(status_code=404, detail="Hazard not found")
    return {"id": hazard_id, "active": False}

@router.delete("/api/admin/hazards/{hazard_id}", dependencies=[Depends(require_admin)])
def delete_hazard(hazard_id: str):
    if not get_hazards().delete(hazard_id): raise HTTPException(status_code=404, detail="Hazard not found")
    return {"id": hazard_id, "deleted": True}
