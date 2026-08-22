from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout


class RoutePanel(BoxLayout):
    summary = StringProperty("No route decision yet")

    def display_route(self, route: list | None, action: str | None):
        route = route or []
        self.summary = f"Action: {action or '—'}\nRoute points: {len(route)}"
