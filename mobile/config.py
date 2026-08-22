"""
Mobile client configuration.

API_BASE_URL is deliberately NOT hard-coded to localhost (Section 44).
Set the SAFEROUTE_API_BASE_URL environment variable, or edit the default
below before packaging:

    LOCAL (Android emulator):  http://10.0.2.2:8000
    PHYSICAL DEVICE:           http://YOUR_COMPUTER_IP:8000
    PRODUCTION:                https://your-domain.example

No Google Maps API key is ever stored here -- the mobile client only
ever talks to the FastAPI backend (Section 16).
"""

import os

API_BASE_URL = os.environ.get("SAFEROUTE_API_BASE_URL", "http://10.0.2.2:8000")
REQUEST_TIMEOUT_S = float(os.environ.get("SAFEROUTE_REQUEST_TIMEOUT_S", "8.0"))
GPS_POLL_INTERVAL_S = float(os.environ.get("SAFEROUTE_GPS_POLL_INTERVAL_S", "5.0"))
