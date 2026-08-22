from __future__ import annotations

from kivy.uix.screenmanager import Screen


class EmergencyScreen(Screen):
    """
    Prominent emergency-mode screen. Shown when the user taps EMERGENCY or
    when a HIGH/CRITICAL proximity alert fires. Makes clear this app does
    not replace official emergency services (Section 53).
    """

    disclaimer_text = (
        "This is a research/educational routing assistant. It does NOT "
        "replace emergency services, official evacuation orders, or human "
        "safety professionals. If you are in immediate danger, contact "
        "local emergency services directly."
    )
