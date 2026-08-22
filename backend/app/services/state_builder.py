"""
State construction.

Converts (GPS + destination + route info + traffic + hazards) into the
fixed-size, normalized observation vector the trained Actor-Critic model
expects -- the same 37-feature schema produced by
rl.envs.evacuation_env.EvacuationEnv:

    [0:2]   agent position (x, y)
    [2:4]   goal position (x, y)
    [4]     normalized distance to goal
    [5]     soft hazard risk at the agent location
    [6]     traffic level
    [7]     disaster severity
    [8]     victims remaining fraction (demand)
    [9]     available resources fraction
    [10]    weather severity (0 clear .. 1 storm)
    [11]    time elapsed / max_steps
    [12]    blocked-roads fraction
    [13]    emergency priority of the goal zone
    [14:29] up to K=5 nearest hazards: (rel_x, rel_y, severity), zero-padded
    [29:37] one-hot of the previous action (8 discrete actions)

Real-world lat/lon is converted into a *local* normalized coordinate
frame using a bounding box around (origin, destination, all hazards).
"""

from __future__ import annotations

import numpy as np

from backend.app.schemas.hazard import HazardOut
from backend.app.schemas.location import LatLng
from backend.app.services.geofence_service import haversine_distance_m
from rl.envs.evacuation_env import MAX_HAZARDS_IN_OBS, N_ACTIONS, OBS_DIM


def _local_bounds(points: list[LatLng]) -> tuple[float, float, float, float]:
    lats = [p.latitude for p in points]
    lons = [p.longitude for p in points]
    pad = 0.01  # small padding so points aren't exactly on the boundary
    return min(lats) - pad, max(lats) + pad, min(lons) - pad, max(lons) + pad


def _normalize(lat: float, lon: float, bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    lat_min, lat_max, lon_min, lon_max = bounds
    nx = (lat - lat_min) / max(lat_max - lat_min, 1e-9)
    ny = (lon - lon_min) / max(lon_max - lon_min, 1e-9)
    return float(np.clip(nx, 0.0, 1.0)), float(np.clip(ny, 0.0, 1.0))


def build_observation(
    current_location: LatLng,
    destination: LatLng,
    hazards: list[HazardOut],
    traffic_factor: float,
    prev_action: int = 0,
    severity: float | None = None,
    victims_frac: float = 0.5,
    resources_frac: float = 1.0,
    weather_value: float = 0.0,
    time_frac: float = 0.0,
    blocked_frac: float = 0.0,
) -> np.ndarray:
    """Builds an observation vector matching EvacuationEnv's schema exactly."""
    bounds = _local_bounds([current_location, destination] + [h.location for h in hazards])

    ax, ay = _normalize(current_location.latitude, current_location.longitude, bounds)
    gx, gy = _normalize(destination.latitude, destination.longitude, bounds)
    real_distance_m = haversine_distance_m(
        current_location.latitude, current_location.longitude,
        destination.latitude, destination.longitude,
    )
    # Normalize distance by a nominal "large evacuation distance" scale
    # (10 km) purely for the observation feature; the real distance in
    # meters is reported separately in the API response.
    distance_norm = float(np.clip(real_distance_m / 10_000.0, 0.0, 1.0))

    risk = 0.0
    hazard_features = []
    max_severity = 0.0
    hazards_with_dist = sorted(
        hazards,
        key=lambda h: haversine_distance_m(current_location.latitude, current_location.longitude,
                                            h.location.latitude, h.location.longitude),
    )[:MAX_HAZARDS_IN_OBS]

    for h in hazards_with_dist:
        d = haversine_distance_m(current_location.latitude, current_location.longitude,
                                  h.location.latitude, h.location.longitude)
        outer = h.radius_m * 2.0
        if d < h.radius_m:
            risk += h.severity
        elif d < outer:
            risk += h.severity * (1.0 - (d - h.radius_m) / (outer - h.radius_m))
        max_severity = max(max_severity, h.severity)

        hx, hy = _normalize(h.location.latitude, h.location.longitude, bounds)
        hazard_features.extend([
            float(np.clip((hx - ax) * 0.5 + 0.5, 0.0, 1.0)),
            float(np.clip((hy - ay) * 0.5 + 0.5, 0.0, 1.0)),
            float(np.clip(h.severity, 0.0, 1.0)),
        ])
    while len(hazard_features) < 3 * MAX_HAZARDS_IN_OBS:
        hazard_features.append(0.0)

    risk = float(np.clip(risk, 0.0, 1.0))
    if severity is None:
        severity = max_severity
    priority = float(np.clip(0.5 + 0.5 * severity, 0.0, 1.0))

    prev_action_onehot = [0.0] * N_ACTIONS
    prev_action_onehot[prev_action] = 1.0

    obs = np.array(
        [ax, ay, gx, gy, distance_norm, risk, float(np.clip(traffic_factor, 0.0, 1.0)),
         float(np.clip(severity, 0.0, 1.0)), float(np.clip(victims_frac, 0.0, 1.0)),
         float(np.clip(resources_frac, 0.0, 1.0)), float(np.clip(weather_value, 0.0, 1.0)),
         float(np.clip(time_frac, 0.0, 1.0)), float(np.clip(blocked_frac, 0.0, 1.0)), priority,
         *hazard_features, *prev_action_onehot],
        dtype=np.float32,
    )
    assert obs.shape == (OBS_DIM,), f"unexpected obs shape {obs.shape} (expected {OBS_DIM})"
    return obs