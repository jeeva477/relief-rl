from kivy.properties import StringProperty
from kivy.uix.popup import Popup


class EmergencyPopup(Popup):
    severity = StringProperty("HIGH")
    distance = StringProperty("0 m")
    recommended_action = StringProperty("EVACUATE")

    def __init__(self, distance_m=0, severity="HIGH", recommended_action="EVACUATE", **kwargs):
        self.distance = f"{float(distance_m):.0f} m"
        self.severity = severity
        self.recommended_action = recommended_action
        super().__init__(title="EMERGENCY ALERT", **kwargs)
