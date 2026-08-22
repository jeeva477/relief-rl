"""Google Maps Platform integration for Relief-RL.

The live provider uses the current Google Geocoding API v4 and Routes API.
All live calls remain behind the GoogleMapsProvider interface so DEMO_MODE and
unit tests never require credentials or network access.
"""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass

import httpx

from backend.app.schemas.location import LatLng
from backend.app.schemas.route import RouteResponse, RouteSegment


@dataclass
class RouteMatrixEntry:
    origin_index: int
    destination_index: int
    distance_m: float
    duration_s: float
    traffic_factor: float


class GoogleMapsProvider(abc.ABC):
    @abc.abstractmethod
    async def geocode(self, address: str) -> LatLng: ...

    @abc.abstractmethod
    async def reverse_geocode(self, location: LatLng) -> str: ...

    @abc.abstractmethod
    async def compute_route(self, origin: LatLng, destination: LatLng) -> RouteResponse: ...

    @abc.abstractmethod
    async def compute_route_matrix(
        self, origins: list[LatLng], destinations: list[LatLng]
    ) -> list[RouteMatrixEntry]: ...


class LiveGoogleMapsProvider(GoogleMapsProvider):
    """Server-side Google Maps implementation. Requires a restricted API key."""

    GEOCODE_ADDRESS_URL = "https://geocode.googleapis.com/v4/geocode/address"
    GEOCODE_LOCATION_URL = "https://geocode.googleapis.com/v4/geocode/location"
    ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
    ROUTE_MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None):
        if not api_key:
            raise ValueError("LiveGoogleMapsProvider requires a non-empty API key")
        self.api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=15.0)

    def _headers(self, field_mask: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }

    async def geocode(self, address: str) -> LatLng:
        response = await self._client.get(
            self.GEOCODE_ADDRESS_URL,
            params={"addressQuery": address},
            headers={"X-Goog-Api-Key": self.api_key},
        )
        response.raise_for_status()
        data = response.json()
        result = self._first_result(data)
        location = result.get("location") or result.get("geometry", {}).get("location")
        if not location:
            raise ValueError(f"Google Geocoding returned no location for: {address}")
        return LatLng(latitude=float(location["latitude"]), longitude=float(location["longitude"]))

    async def reverse_geocode(self, location: LatLng) -> str:
        response = await self._client.get(
            self.GEOCODE_LOCATION_URL,
            params={"locationQuery": f"{location.latitude},{location.longitude}"},
            headers={"X-Goog-Api-Key": self.api_key},
        )
        response.raise_for_status()
        data = response.json()
        result = self._first_result(data)
        return str(result.get("formattedAddress") or result.get("formatted_address") or "Unknown address")

    async def compute_route(self, origin: LatLng, destination: LatLng) -> RouteResponse:
        body = {
            "origin": {"location": {"latLng": {"latitude": origin.latitude, "longitude": origin.longitude}}},
            "destination": {"location": {"latLng": {"latitude": destination.latitude, "longitude": destination.longitude}}},
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE",
            "computeAlternativeRoutes": True,
            "units": "METRIC",
        }
        response = await self._client.post(
            self.ROUTES_URL,
            json=body,
            headers=self._headers(
                "routes.distanceMeters,routes.duration,routes.staticDuration,routes.polyline.encodedPolyline"
            ),
        )
        response.raise_for_status()
        data = response.json()
        routes = data.get("routes", [])
        if not routes:
            raise ValueError("Google Routes API returned no routes")

        segments: list[RouteSegment] = []
        for route in routes:
            distance_m = float(route.get("distanceMeters", 0))
            duration_s = _duration_seconds(route.get("duration"))
            static_s = _duration_seconds(route.get("staticDuration"))
            traffic_factor = max(0.0, duration_s / static_s - 1.0) if static_s > 0 else 0.0
            encoded = route.get("polyline", {}).get("encodedPolyline", "")
            coordinates = _decode_polyline(encoded) if encoded else [origin, destination]
            if not coordinates:
                coordinates = [origin, destination]
            segments.append(
                RouteSegment(
                    start=coordinates[0],
                    end=coordinates[-1],
                    distance_m=distance_m,
                    duration_s=duration_s,
                    traffic_factor=traffic_factor,
                    coordinates=coordinates,
                    risk=0.0,
                    blocked=False,
                )
            )

        best = segments[0]
        return RouteResponse(
            segments=segments,
            total_distance_m=best.distance_m,
            total_duration_s=best.duration_s,
            source="google_routes_api",
        )

    async def compute_route_matrix(
        self, origins: list[LatLng], destinations: list[LatLng]
    ) -> list[RouteMatrixEntry]:
        if not origins or not destinations:
            return []
        if len(origins) * len(destinations) > 100:
            raise ValueError("Traffic-aware route matrix is limited to 100 origin-destination elements per request")

        body = {
            "origins": [
                {"waypoint": {"location": {"latLng": {"latitude": p.latitude, "longitude": p.longitude}}}}
                for p in origins
            ],
            "destinations": [
                {"waypoint": {"location": {"latLng": {"latitude": p.latitude, "longitude": p.longitude}}}}
                for p in destinations
            ],
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE_OPTIMAL",
        }
        response = await self._client.post(
            self.ROUTE_MATRIX_URL,
            json=body,
            headers=self._headers("originIndex,destinationIndex,status,condition,distanceMeters,duration,staticDuration"),
        )
        response.raise_for_status()

        # REST ComputeRouteMatrix returns an array of elements. Each element
        # identifies its pair through originIndex/destinationIndex.
        payload = response.json()
        elements = payload if isinstance(payload, list) else payload.get("elements", [])
        entries: list[RouteMatrixEntry] = []
        for element in elements:
            status = element.get("status", {})
            if isinstance(status, dict) and status.get("code", 0) not in (0, None):
                continue
            oi = int(element.get("originIndex", 0))
            di = int(element.get("destinationIndex", 0))
            duration_s = _duration_seconds(element.get("duration"))
            static_s = _duration_seconds(element.get("staticDuration"))
            traffic_factor = max(0.0, duration_s / static_s - 1.0) if static_s > 0 else 0.0
            entries.append(
                RouteMatrixEntry(
                    origin_index=oi,
                    destination_index=di,
                    distance_m=float(element.get("distanceMeters", 0)),
                    duration_s=duration_s,
                    traffic_factor=traffic_factor,
                )
            )
        return entries

    @staticmethod
    def _first_result(data: dict) -> dict:
        results = data.get("results", [])
        if not results:
            raise ValueError("Google Geocoding returned no results")
        return results[0]


class MockGoogleMapsProvider(GoogleMapsProvider):
    """Deterministic provider for DEMO_MODE and tests."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    async def geocode(self, address: str) -> LatLng:
        h = abs(hash(address)) % 10_000
        return LatLng(latitude=11.0 + (h % 100) / 1000.0, longitude=77.0 + (h % 137) / 1000.0)

    async def reverse_geocode(self, location: LatLng) -> str:
        return f"Mock address near ({location.latitude:.4f}, {location.longitude:.4f})"

    async def compute_route(self, origin: LatLng, destination: LatLng) -> RouteResponse:
        distance_m = self._haversine_m(origin, destination)
        traffic_factor = self._rng.uniform(0.1, 0.6)
        duration_s = distance_m / 11.0 * (1.0 + traffic_factor)
        coordinates = [
            LatLng(
                latitude=origin.latitude + (destination.latitude - origin.latitude) * i / 3,
                longitude=origin.longitude + (destination.longitude - origin.longitude) * i / 3,
            )
            for i in range(4)
        ]
        segment = RouteSegment(
            start=origin, end=destination, distance_m=distance_m, duration_s=duration_s,
            traffic_factor=traffic_factor, coordinates=coordinates, risk=0.0, blocked=False,
        )
        return RouteResponse(segments=[segment], total_distance_m=distance_m,
                             total_duration_s=duration_s, source="mock_provider")

    async def compute_route_matrix(self, origins, destinations) -> list[RouteMatrixEntry]:
        entries = []
        for oi, origin in enumerate(origins):
            for di, destination in enumerate(destinations):
                distance = self._haversine_m(origin, destination)
                traffic = self._rng.uniform(0.1, 0.6)
                duration = distance / 11.0 * (1.0 + traffic)
                entries.append(RouteMatrixEntry(oi, di, distance, duration, traffic))
        return entries

    @staticmethod
    def _haversine_m(a: LatLng, b: LatLng) -> float:
        from backend.app.services.geofence_service import haversine_distance_m
        return haversine_distance_m(a.latitude, a.longitude, b.latitude, b.longitude)


def _duration_seconds(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.endswith("s"):
        return float(text[:-1] or 0)
    return float(text or 0)


def _decode_polyline(encoded: str) -> list[LatLng]:
    """Decode Google's encoded polyline format into GPS coordinates."""
    points: list[LatLng] = []
    index = lat = lng = 0
    while index < len(encoded):
        result = shift = 0
        while True:
            byte = ord(encoded[index]) - 63
            index += 1
            result |= (byte & 0x1F) << shift
            shift += 5
            if byte < 0x20:
                break
        lat += ~(result >> 1) if result & 1 else result >> 1

        result = shift = 0
        while True:
            byte = ord(encoded[index]) - 63
            index += 1
            result |= (byte & 0x1F) << shift
            shift += 5
            if byte < 0x20:
                break
        lng += ~(result >> 1) if result & 1 else result >> 1
        points.append(LatLng(latitude=lat / 1e5, longitude=lng / 1e5))
    return points


def get_provider(api_key: str, demo_mode: bool) -> GoogleMapsProvider:
    if demo_mode or not api_key:
        return MockGoogleMapsProvider()
    return LiveGoogleMapsProvider(api_key)
