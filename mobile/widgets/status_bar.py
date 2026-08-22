from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout


class StatusBar(BoxLayout):
    status = StringProperty("READY")
    risk = StringProperty("LOW")

    def set_status(self, status: str, risk: str | None = None):
        self.status = status or "UNKNOWN"
        self.risk = risk or "UNKNOWN"
