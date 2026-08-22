"""
Baseline 2 -- Rule-based safety heuristic (Section 29).

Priority order, evaluated per candidate neighbor cell:
    1. eliminate hard hazards   (never choose a cell with a hard violation)
    2. minimize risk            (prefer lower soft hazard risk)
    3. minimize distance        (prefer cells closer to goal)
    4. minimize travel time     (proxy: fewer remaining steps, i.e. same as distance here)

This is also the fallback "decision_source": "SAFETY_HEURISTIC" used by
the FastAPI backend when no trained RL checkpoint is available
(Section 47) -- see backend/app/services/safety_validator.py.
"""

from __future__ import annotations

from rl.envs.evacuation_env import EvacuationEnv


def safety_heuristic_action(env: EvacuationEnv) -> int:
    candidates = env.graph.neighbors(*env.agent_cell)
    if not candidates:
        return 0

    scored = []
    for action_id, cell in candidates.items():
        nx, ny = env.graph.normalize(*cell)
        hard_violation = any(hz.hard_constraint and hz.contains(nx, ny) for hz in env.hazards)
        risk = env._risk_at_cell(cell)
        distance = env._distance_to_goal(cell)
        # Sort key: (hard_violation, risk, distance) -- lexicographic priority
        # exactly matches the four-level priority list in the docstring.
        scored.append((hard_violation, risk, distance, action_id))

    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return scored[0][3]
