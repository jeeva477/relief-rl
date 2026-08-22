"""
Baseline 1 -- Shortest feasible route (Section 29).

Uses breadth-first search over the grid road graph to find the
shortest path from start to goal that never enters a cell violating a
hard-constraint hazard or a blocked cell. This gives a "distance-optimal
but safety-respecting" baseline to compare the learned policy against.
"""

from __future__ import annotations

from collections import deque

from rl.envs.evacuation_env import EvacuationEnv


def _cell_is_hard_unsafe(env: EvacuationEnv, cell: tuple[int, int]) -> bool:
    nx, ny = env.graph.normalize(*cell)
    return any(hz.hard_constraint and hz.contains(nx, ny) for hz in env.hazards)


def shortest_safe_path_action(env: EvacuationEnv) -> int:
    """
    Returns the action (0-4) that moves the agent one step along the
    shortest BFS path from its current cell to the goal, treating
    blocked cells and hard-hazard cells as impassable. Falls back to
    `stay` (0) if no safe path currently exists.
    """
    start = env.agent_cell
    goal = env.goal_cell

    if start == goal:
        return 0

    visited = {start}
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for action_id, neighbor in env.graph.neighbors(*current).items():
            if action_id == 0 or neighbor in visited:
                continue
            if _cell_is_hard_unsafe(env, neighbor):
                continue
            visited.add(neighbor)
            parent[neighbor] = (current, action_id)
            queue.append(neighbor)

    if goal not in parent and goal != start:
        return 0  # no safe path found -> stay, defer to safety layer / human intervention

    # Walk parent pointers back from goal to find the first step from start.
    node = goal
    first_action = 0
    while node in parent:
        prev, action_id = parent[node]
        if prev == start:
            first_action = action_id
            break
        node = prev
    return first_action
