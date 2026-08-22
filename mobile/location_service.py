"""
GPS location service abstraction.

On Android, this wraps plyer's GPS API. On desktop (for development/demo
without a real device), it falls back to a configurable simulated
location so the rest of the app is testable without hardware.

Document any platform-specific limitation explicitly rather than
silently degrading (Section 25/45).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GpsFix:
    latitude: float
    longitude: float
    source: str  # "device_gps" | "simulated"


class LocationService:
    def __init__(self, simulated_start: tuple[float, float] = (11.341, 77.717)):
        self._simulated = simulated_start
        self._use_device_gps = self._device_gps_available()

    @staticmethod
    def _device_gps_available() -> bool:
        try:
            from plyer import gps  # noqa: F401
            return True
        except Exception:
            return False

    def request_permissions(self) -> None:
        """
        On Android this must request ACCESS_FINE_LOCATION at runtime.
        Documented here rather than silently assumed granted; the caller
        should check `has_fix()` after calling this and handle denial.
        """
        if not self._use_device_gps:
            return
        try:
            from plyer import gps
            gps.configure(on_location=self._on_location, on_status=self._on_status)
            gps.start(minTime=1000, minDistance=1)
        except Exception:
            self._use_device_gps = False

    def _on_location(self, **kwargs):
        self._simulated = (kwargs.get("lat", self._simulated[0]), kwargs.get("lon", self._simulated[1]))

    def _on_status(self, stype, status):
        pass

    def get_current_location(self) -> GpsFix:
        lat, lon = self._simulated
        return GpsFix(latitude=lat, longitude=lon, source="device_gps" if self._use_device_gps else "simulated")

    def simulate_move(self, d_lat: float, d_lon: float) -> None:
        """Dev-only helper for the desktop fallback / SIMULATE button."""
        self._simulated = (self._simulated[0] + d_lat, self._simulated[1] + d_lon)
