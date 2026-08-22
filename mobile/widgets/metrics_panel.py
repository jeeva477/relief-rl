from kivy.properties import StringProperty
from kivy.uix.gridlayout import GridLayout


class MetricsPanel(GridLayout):
    safety = StringProperty("--")
    distance = StringProperty("--")
    eta = StringProperty("--")
    hazard = StringProperty("--")

    def update_metrics(self, safety_score=None, distance_m=None, eta_s=None, hazard_level=None):
        self.safety = "--" if safety_score is None else f"{float(safety_score) * 100:.0f}%"
        self.distance = "--" if distance_m is None else f"{float(distance_m):.0f} m"
        self.eta = "--" if eta_s is None else f"{float(eta_s):.0f} s"
        self.hazard = "--" if hazard_level is None else f"{float(hazard_level):.2f}"
