from __future__ import annotations

from fastapi import APIRouter

from backend.app.config import get_settings
from backend.app.core.dependencies import get_model_handle

router = APIRouter()


@router.get("/health")
def health():
    settings = get_settings()
    handle = get_model_handle()
    return {
        "status": "ok",
        "app_env": settings.app_env,
        "demo_mode": settings.demo_mode,
        "model_available": handle.available,
        "model_name": handle.model_name if handle.available else None,
    }
