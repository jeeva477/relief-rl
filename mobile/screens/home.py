"""Relief-RL mobile dashboard screen."""

from __future__ import annotations

from kivy.clock import Clock
from kivy.uix.screenmanager import Screen
from kivymd.uix.snackbar import Snackbar

from mobile.api_client import ApiError, SafeRouteApiClient
from mobile.location_service import LocationService


class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api = SafeRouteApiClient()
        self.location_service = LocationService()
        self._destination = (11.350, 77.725)
        self._last_decision: dict | None = None

    def on_enter(self):
        self.location_service.request_permissions()
        Clock.schedule_once(lambda dt: self.refresh_status(), 0.5)

    def refresh_status(self):
        import asynckivy as ak
        ak.start(self._refresh_status_async())

    async def _refresh_status_async(self):
        fix = self.location_service.get_current_location()
        try:
            decision = await self.api.get_rl_decision(
                fix.latitude, fix.longitude, self._destination[0], self._destination[1]
            )
            self._last_decision = decision
            self._apply_decision_to_ui(decision)

            map_panel = self.ids.get("map_panel")
            if map_panel:
                map_panel.set_route(decision.get("route") or [])

            proximity = await self.api.check_proximity(fix.latitude, fix.longitude)
            if proximity.get("nearby") and proximity.get("severity") in ("HIGH", "CRITICAL"):
                self._show_emergency_popup(proximity)
        except ApiError as exc:
            self._show_degraded_banner(str(exc))

    def _apply_decision_to_ui(self, decision: dict):
        status_bar = self.ids.get("status_bar")
        metrics_panel = self.ids.get("metrics_panel")
        if status_bar:
            status_bar.set_status(decision.get("status", "UNKNOWN"), decision.get("risk_level"))
        if metrics_panel:
            metrics_panel.update_metrics(
                safety_score=decision.get("safety_score"),
                distance_m=decision.get("remaining_distance_m"),
                eta_s=decision.get("estimated_time_s"),
                hazard_level=decision.get("hazard_level"),
            )
        if decision.get("status") == "NO_SAFE_ROUTE":
            Snackbar(text=decision.get("message") or "No safe route available. Human intervention required.").open()

    def _show_emergency_popup(self, proximity: dict):
        from mobile.widgets.alert_popup import EmergencyPopup
        EmergencyPopup(
            distance_m=proximity.get("distance_m", 0),
            severity=proximity.get("severity", "HIGH"),
            recommended_action=proximity.get("recommended_action", "EVACUATE"),
        ).open()

    def _show_degraded_banner(self, message: str):
        Snackbar(text=f"Live update unavailable: {message}").open()

    def on_safe_route_pressed(self):
        self.refresh_status()

    def on_update_pressed(self):
        self.refresh_status()

    def on_simulate_pressed(self):
        self.location_service.simulate_move(0.001, 0.001)
        self.refresh_status()

    def on_emergency_pressed(self):
        self.manager.current = "emergency"
