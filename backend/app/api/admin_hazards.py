from __future__ import annotations
from fastapi import APIRouter, HTTPException
from backend.app.core.dependencies import get_hazards
from backend.app.schemas.hazard import HazardIn, HazardOut

router = APIRouter()

@router.get('/api/admin/hazards', response_model=list[HazardOut])
def list_all_hazards():
    return get_hazards().list_all()

@router.post('/api/admin/hazards', response_model=HazardOut)
def create_admin_hazard(hazard: HazardIn):
    return get_hazards().add(HazardOut(**hazard.model_dump(), active=True))

@router.put('/api/admin/hazards/{hazard_id}', response_model=HazardOut)
def update_admin_hazard(hazard_id: str, hazard: HazardIn):
    if hazard.id != hazard_id:
        raise HTTPException(status_code=400, detail='Hazard ID in path and body must match')
    result = get_hazards().update(hazard_id, HazardOut(**hazard.model_dump(), active=True))
    if result is None: raise HTTPException(status_code=404, detail='Hazard not found')
    return result

@router.post('/api/admin/hazards/{hazard_id}/deactivate', response_model=dict)
def deactivate_admin_hazard(hazard_id: str):
    if not get_hazards().deactivate(hazard_id): raise HTTPException(status_code=404, detail='Hazard not found')
    return {'success': True, 'id': hazard_id}

@router.delete('/api/admin/hazards/{hazard_id}', response_model=dict)
def delete_admin_hazard(hazard_id: str):
    if not get_hazards().delete(hazard_id): raise HTTPException(status_code=404, detail='Hazard not found')
    return {'success': True, 'id': hazard_id}
