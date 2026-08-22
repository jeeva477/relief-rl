"""
Async HTTP client for the Kivy mobile app.

Never blocks the UI thread: every method is a coroutine intended to be
scheduled via Kivy's asynckivy/async support (or a background thread
pool as a fallback -- see screens/navigation.py for usage). Handles
timeouts and connection failures explicitly rather than letting them
crash the UI (Section 27).
"""

from __future__ import annotations

import httpx

from mobile.config import API_BASE_URL, REQUEST_TIMEOUT_S


class ApiError(Exception):
    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(message)
        self.recoverable = recoverable


class SafeRouteApiClient:
    def __init__(self, base_url: str = API_BASE_URL, timeout: float = REQUEST_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _post(self, path: str, json: dict) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}{path}", json=json)
                resp.raise_for_status()
                return resp.json()
        except httpx.TimeoutException as exc:
            raise ApiError(f"Request to {path} timed out. Check your connection.") from exc
        except httpx.ConnectError as exc:
            raise ApiError(f"Could not reach the server at {self.base_url}. No live update available.") from exc
        except httpx.HTTPStatusError as exc:
            raise ApiError(f"Server error on {path}: {exc.response.status_code}") from exc

    async def _get(self, path: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.base_url}{path}")
                resp.raise_for_status()
                return resp.json()
        except httpx.TimeoutException as exc:
            raise ApiError(f"Request to {path} timed out.") from exc
        except httpx.ConnectError as exc:
            raise ApiError(f"Could not reach the server at {self.base_url}. No live update available.") from exc
        except httpx.HTTPStatusError as exc:
            raise ApiError(f"Server error on {path}: {exc.response.status_code}") from exc

    async def health(self) -> dict:
        return await self._get("/health")

    async def get_rl_decision(self, current_lat: float, current_lon: float, dest_lat: float, dest_lon: float) -> dict:
        return await self._post("/api/rl/decision", {
            "current_location": {"latitude": current_lat, "longitude": current_lon},
            "destination": {"latitude": dest_lat, "longitude": dest_lon},
        })

    async def check_proximity(self, lat: float, lon: float) -> dict:
        return await self._post("/api/proximity/check", {"location": {"latitude": lat, "longitude": lon}})

    async def get_route(self, origin_lat, origin_lon, dest_lat, dest_lon) -> dict:
        return await self._post("/api/route", {
            "origin": {"latitude": origin_lat, "longitude": origin_lon},
            "destination": {"latitude": dest_lat, "longitude": dest_lon},
        })

    async def list_hazards(self) -> list[dict]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.base_url}/api/hazards")
                resp.raise_for_status()
                return resp.json()
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError):
            return []  # graceful degradation: show cached/no hazards rather than crash
