from __future__ import annotations
import time, uuid
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from backend.app.api import alerts, auth, geocoding, hazards, health, maps_status, proximity, readiness, rl, routes, sim
from backend.app.config import get_settings
from backend.app.core.logging import configure_logging, get_logger
settings = get_settings(); configure_logging(settings.log_level); logger = get_logger("relief.api")
app = FastAPI(title="Relief-RL API", description="Adaptive Reinforcement Learning for Disaster Evacuation and Safe Route Optimization. Research/educational prototype -- not a replacement for official emergency services or evacuation orders.", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",")], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id=str(uuid.uuid4()); start=time.time()
    try: response=await call_next(request)
    except Exception:
        logger.exception(f"[{request_id}] Unhandled error on {request.method} {request.url.path}")
        return JSONResponse(status_code=500, content={"status":"ERROR","message":"Internal server error.","request_id":request_id})
    logger.info(f"[{request_id}] {request.method} {request.url.path} -> {response.status_code} ({(time.time()-start)*1000:.1f}ms)"); response.headers["X-Request-ID"]=request_id; return response
app.include_router(auth.router, tags=["auth"]); app.include_router(maps_status.router, tags=["maps"])
app.include_router(health.router, tags=["health"]); app.include_router(readiness.router, tags=["health"])
app.include_router(geocoding.router, tags=["geocoding"]); app.include_router(routes.router, tags=["routes"]); app.include_router(hazards.router, tags=["hazards"])
app.include_router(proximity.router, tags=["proximity"]); app.include_router(rl.router, tags=["rl"]); app.include_router(alerts.router, tags=["alerts"])
app.include_router(sim.router, tags=["sim"])
_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

@app.get("/")
def root(request: Request):
    accept = request.headers.get("accept", "")
    if _DIST.exists() and "text/html" in accept and "application/json" not in accept:
        return FileResponse(_DIST / "index.html")
    return {"service":"Relief-RL","docs":"/docs","disclaimer":"Research/educational prototype. Does not replace emergency services, official evacuation orders, or human safety professionals. Live hazard information may be incomplete or delayed."}

if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")
    app.mount("/app", StaticFiles(directory=_DIST, html=True), name="frontend")
