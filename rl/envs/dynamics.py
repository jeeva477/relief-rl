"""
Mid-episode environment dynamics.

The disaster world is not static. During a single episode the following
can change (each change is emitted as an event so the frontend can show
it on the timeline and animate it):

    ROAD_BLOCKED    a previously open road closes
    ROAD_REOPENED   a previously closed road reopens
    SEVERITY_UP     the disaster intensifies (hazard growth)
    TRAFFIC_UP / TRAFFIC_DOWN   traffic drifts over time
    DEMAND_UP       more victims become stranded as severity rises

Road changes are validated against connectivity: a road is only closed
(or reopened) if the start-to-goal route remains traversable, so the
episode never becomes trivially impossible.
"""

from __future__ import annotations

from collections import deque

from rl.envs.hazard import Hazard


WEATHER_SPEED_FACTOR = {
    "clear": 1.0,
    "rain": 1.15,
    "heavy_rain": 1.35,
    "storm": 1.55,
}

WEATHER_VALUE = {
    "clear": 0.0,
    "rain": 0.33,
    "heavy_rain": 0.66,
    "storm": 1.0,
}


def _graph_connected(env, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    """BFS over non-blocked cells. Used to keep road dynamics feasible."""
    if start == goal:
        return True
    visited = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        for action_id, neighbor in env.graph.neighbors(*current).items():
            if action_id == 0 or neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return False


def update_road_blocks(env, rng) -> str | None:
    """
    Every `road_change_interval` steps, toggle one road while preserving
    connectivity. Returns an event string or None.
    """
    cell = (rng.randint(0, env.grid_size - 1), rng.randint(0, env.grid_size - 1))
    if cell in (env.agent_cell, env.goal_cell):
        return None
    was_blocked = env.graph.is_blocked(*cell)
    if was_blocked:
        # Try reopening.
        env.graph._blocked.discard(cell)
        if not _graph_connected(env, env.agent_cell, env.goal_cell):
            env.graph._blocked.add(cell)  # restore
            return None
        return "ROAD_REOPENED"
    # Try closing.
    env.graph._blocked.add(cell)
    if not _graph_connected(env, env.agent_cell, env.goal_cell):
        env.graph._blocked.discard(cell)  # restore
        return None
    return "ROAD_BLOCKED"


def update_severity(env) -> str | None:
    """Severity slowly climbs as hazards grow; returns event or None."""
    old = env.severity
    env.severity = min(1.0, env.severity + 0.002)
    if env.severity - old > 1e-9 and int(env.severity * 100) // 5 > int(old * 100) // 5:
        return "SEVERITY_UP"
    return None


def update_traffic(env, rng) -> str | None:
    """Traffic drifts within bounds; returns event or None."""
    old = env.traffic_level
    env.traffic_level = max(0.0, min(1.0, env.traffic_level + rng.uniform(-0.03, 0.03)))
    if env.traffic_level - old > 0.02:
        return "TRAFFIC_UP"
    if old - env.traffic_level > 0.02:
        return "TRAFFIC_DOWN"
    return None


def update_demand(env, rng) -> str | None:
    """As severity climbs, more victims become stranded (up to a cap)."""
    if env.severity > 0.55 and env.victims < env.victims_cap and rng.random() < 0.3:
        env.victims += 1
        return "DEMAND_UP"
    return None


def weather_speed_multiplier(env) -> float:
    """How much the current weather slows travel (multiplies time cost)."""
    return WEATHER_SPEED_FACTOR.get(env.weather.value if hasattr(env.weather, "value") else str(env.weather), 1.0)


def apply_dynamics(env, rng) -> list[str]:
    """Apply all mid-episode dynamics for one step. Returns event list."""
    events: list[str] = []

    if env.steps_taken > 0 and env.steps_taken % env.road_change_interval == 0:
        ev = update_road_blocks(env, rng)
        if ev:
            events.append(ev)

    ev = update_severity(env)
    if ev:
        events.append(ev)

    ev = update_traffic(env, rng)
    if ev:
        events.append(ev)

    ev = update_demand(env, rng)
    if ev:
        events.append(ev)

    return events