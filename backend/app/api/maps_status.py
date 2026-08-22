from fastapi import APIRouter
from backend.app.config import get_settings

router = APIRouter(prefix="/api/maps", tags=["maps"])

@router.get("/status")
def maps_status():
    settings = get_settings()
    return {
        "configured": bool(settings.google_maps_api_key),
        "mode": "demo" if settings.demo_mode or not settings.google_maps_api_key else "live",
        "provider": "google_maps" if settings.google_maps_api_key and not settings.demo_mode else "mock",
        "key_exposed": False,
    }
