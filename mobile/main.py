from pathlib import Path

from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from kivymd.app import MDApp

from mobile.screens.emergency import EmergencyScreen
from mobile.screens.home import HomeScreen
from mobile.screens.navigation import NavigationScreen
from mobile.widgets.alert_popup import EmergencyPopup
from mobile.widgets.map_panel import MapPanel
from mobile.widgets.metrics_panel import MetricsPanel
from mobile.widgets.route_panel import RoutePanel
from mobile.widgets.status_bar import StatusBar


class ReliefRLApp(MDApp):
    title = "Relief-RL"

    def build(self):
        Builder.load_file(str(Path(__file__).with_name("relief.kv")))
        manager = ScreenManager()
        manager.add_widget(HomeScreen(name="home"))
        manager.add_widget(NavigationScreen(name="navigation"))
        manager.add_widget(EmergencyScreen(name="emergency"))
        return manager


if __name__ == "__main__":
    ReliefRLApp().run()
