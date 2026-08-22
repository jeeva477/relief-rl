from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.app.config import get_settings
from backend.app.core.dependencies import get_model_handle
from backend.app.db import create_engine_and_session

router = APIRouter()


@router.get("/ready")
def ready():
    """Readiness probe for Docker/Cloud Run/load balancers."""
    settings = get_settings()
    checks: dict[str, str] = {"api": "ok"}
    ready_ok = True

    if settings.database_url:
        try:
            engine, _ = create_engine_and_session(settings.database_url)
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            engine.dispose()
            checks["database"] = "ok"
        except Exception as exc:
            checks["database"] = f"error: {type(exc).__name__}"
            ready_ok = False
    else:
        checks["database"] = "not_configured"

    handle = get_model_handle()
    checks["model"] = "ok" if handle.available else "fallback_mode"

    payload = {
        "status": "ready" if ready_ok else "not_ready",
        "checks": checks,
        "demo_mode": settings.demo_mode,
    }
    return JSONResponse(
        status_code=status.HTTP_200_OK if ready_ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload,
    )
