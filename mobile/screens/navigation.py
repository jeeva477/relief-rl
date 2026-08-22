from __future__ import annotations

from kivy.uix.screenmanager import Screen


class NavigationScreen(Screen):
    """Full-screen route view: map panel + turn-by-turn/route summary."""

    def on_pre_enter(self):
        home = self.manager.get_screen("home")
        decision = getattr(home, "_last_decision", None)
        route_panel = self.ids.get("route_panel")
        if route_panel and decision:
            route_panel.display_route(decision.get("route", []), decision.get("action"))
