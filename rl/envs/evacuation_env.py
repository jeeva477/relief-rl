"""
EvacuationEnv: Gymnasium environment for adaptive disaster-response routing.

STATE / OBSERVATION SPACE (37 features, all normalized to [0, 1])
------------------------------------------------------------------
    [0:2]    agent position (x, y)
    [2:4]    goal position (x, y)
    [4]      normalized distance to goal (by grid diagonal)
    [5]      soft hazard risk at the agent cell
    [6]      traffic level
    [7]      disaster severity
    [8]      victims remaining fraction (demand)
    [9]      available resources fraction
    [10]     weather severity (0 clear .. 1 storm)
    [11]     time elapsed / max_steps
    [12]     blocked-roads fraction
    [13]     emergency priority of the goal zone
    [14:29]  up to K=5 nearest hazards: (rel_x, rel_y, severity), zero-padded
    [29:37]  one-hot of the previous action (8 discrete actions)

Every feature the agent sees is a real property of the simulation (no
fake variables), so the policy can genuinely condition on disaster state.

ACTION SPACE
------------
Discrete(8):
    0 STAY
    1 NORTH
    2 EAST
    3 SOUTH
    4 WEST
    5 REROUTE      switch to the currently safest adjacent cell (detour)
    6 DISPATCH     spend one resource: dispatch a second rescue vehicle
                   (halves traffic cost for a few steps)
    7 PRIORITIZE   spend one resource: raise the goal's emergency priority
                   (rescues more victims and earns a larger success bonus)

Movement into a blocked cell is IMPOSSIBLE (masked). Dispatching /
prioritizing without resources is possible but wasteful (the agent must
learn to avoid it through the reward signal).

REWARD
------
Every component is configurable via :class:`RewardWeights`:

    + w.progress      * distance reduced toward goal this step
    + w.safe_zone     within goal radius
    + w.success       * (0.5 + 0.5*priority)        terminal success
    + w.rescue        * victims rescued             terminal success
    + w.efficiency    * route-efficiency bonus      terminal success
    + w.dispatch      * successful dispatch

    - w.distance      * remaining distance
    - w.time          * time_cost_multiplier (weather/dispatch affect it)
    - w.risk          * soft hazard risk at new cell
    - w.traffic       * traffic level (when actually moving; halved if dispatching)
    - w.blocked       * attempted move into a blocked cell
    - w.hard          * entered a hard-constraint hazard zone
    - w.reroute_cost  * used REROUTE
    - w.unnecessary   * moved away from the goal without reducing risk
    - w.resource_waste * dispatch/prioritize with no resources
    - w.failed_rescue * episode truncated without reaching the goal

EPISODE / TERMINATION
---------------------
An episode terminates when the agent reaches the goal zone (success) or
truncates at `max_steps` (failure). `info["episode_metrics"]` reports
rescue count, response time, route efficiency, resource usage and the
other quantities the training/evaluation/UI layers display.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl.envs import dynamics
from rl.envs.hazard import Hazard
from rl.envs.road_graph import GridRoadGraph
from rl.envs.scenarios import (
    Difficulty,
    Scenario,
    ScenarioConfig,
    WeatherType,
    generate_scenario,
    generate_scenario_from_config,
)

MAX_HAZARDS_IN_OBS = 5  # K nearest hazards encoded in the observation
N_ACTIONS = 8
TIME_PER_STEP_S = 2.0   # each discrete step represents this many real seconds

_ACTION_NAMES = {0: "STAY", 1: "NORTH", 2: "EAST", 3: "SOUTH", 4: "WEST",
                 5: "REROUTE", 6: "DISPATCH", 7: "PRIORITIZE"}

OBS_DIM = 14 + 3 * MAX_HAZARDS_IN_OBS + N_ACTIONS


@dataclass
class RewardWeights:
    """
    Configurable reward weights.

    R_t =
        + w_progress * progress
        + w_safe_zone * (in goal radius)
        + w_success * (0.5 + 0.5*priority) * success
        + w_rescue * victims_rescued * (0.5 + 0.5*priority)
        + w_efficiency * efficiency_bonus
        + w_dispatch * dispatched
        - w_distance * distance_remaining
        - w_time * time_cost_multiplier
        - w_risk * risk_at_new
        - w_traffic * traffic_effect (if moved)
        - w_blocked * blocked_attempt
        - w_hard * hard_violation
        - w_reroute_cost * reroute_used
        - w_unnecessary * unnecessary_move
        - w_resource_waste * wasted_resource_action
        - w_failed_rescue * (0.5 + 0.5*priority) * victims_remaining
    """
    progress: float = 6.0
    distance: float = 0.01
    time: float = 0.04
    risk: float = 2.0
    traffic: float = 0.8
    blocked: float = 0.8
    hard: float = 15.0
    reroute_cost: float = 0.4
    unnecessary: float = 0.15
    resource_waste: float = 2.0
    safe_zone: float = 2.0
    success: float = 60.0
    rescue: float = 0.6
    efficiency: float = 10.0
    failed_rescue: float = 0.3
    dispatch: float = 1.0


class EvacuationEnv(gym.Env):
    """Custom Gymnasium environment simulating adaptive disaster response."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        difficulty: Difficulty | str = Difficulty.MEDIUM,
        grid_size: int = 10,
        max_steps: int = 100,
        reward_weights: RewardWeights | None = None,
        goal_radius_cells: float = 1.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.difficulty = Difficulty(difficulty) if isinstance(difficulty, str) else difficulty
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.weights = reward_weights or RewardWeights()
        self.goal_radius_cells = goal_radius_cells
        self.render_mode = render_mode
        self.goal_radius_norm = goal_radius_cells / max(grid_size - 1, 1)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.graph = GridRoadGraph(size=grid_size)
        self.scenario: Scenario | None = None
        self.hazards: list[Hazard] = []
        self.agent_cell: tuple[int, int] = (0, 0)
        self.goal_cell: tuple[int, int] = (grid_size - 1, grid_size - 1)
        self.traffic_level: float = 0.0
        self.weather: WeatherType = WeatherType.CLEAR
        self.severity: float = 0.5
        self.victims: int = 0
        self.victims_cap: int = 0
        self.resources: int = 0
        self.priority: float = 0.5
        self.vehicles: int = 1
        self.dispatch_steps_left: int = 0
        self.prev_action = 0
        self.steps_taken = 0
        self.route_distance = 0
        self.blocked_attempts = 0
        self.wasted_actions = 0
        self.hard_violations = 0
        self.cumulative_risk = 0.0
        self.shortest_feasible: int = 0
        self.road_change_interval = max(5, max_steps // 8)
        self._prev_distance: float | None = None
        self._np_random_seed: int | None = None
        self.episode_reward: float = 0.0
        self.episode_breakdown: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._np_random_seed = seed
        rng = self.np_random  # gymnasium's seeded Generator
        py_rng = __import__("random").Random(int(rng.integers(0, 2**31 - 1)))

        options = options or {}
        if "scenario_config" in options:
            cfg = options["scenario_config"]
            if not isinstance(cfg, ScenarioConfig):
                cfg = ScenarioConfig(**cfg)
            self.scenario = generate_scenario_from_config(cfg, rng=py_rng)
        else:
            difficulty = Difficulty(options["difficulty"]) if "difficulty" in options else self.difficulty
            self.scenario = generate_scenario(difficulty, grid_size=self.grid_size, rng=py_rng)

        self.graph.set_blocked(self.scenario.blocked_cells)
        self.hazards = self.scenario.hazards
        self.agent_cell = self.scenario.start
        self.goal_cell = self.scenario.goal
        self.traffic_level = self.scenario.traffic_level
        self.weather = self.scenario.weather
        self.severity = self.scenario.severity
        self.victims = self.scenario.victims
        self.victims_cap = max(self.victims, 60)
        self.resources = self.scenario.resources
        self.priority = self.scenario.priority
        self.vehicles = 1
        self.dispatch_steps_left = 0
        self.prev_action = 0
        self.steps_taken = 0
        self.route_distance = 0
        self.blocked_attempts = 0
        self.wasted_actions = 0
        self.hard_violations = 0
        self.cumulative_risk = 0.0
        self.shortest_feasible = self._shortest_feasible_path()
        self._prev_distance = self._distance_to_goal(self.agent_cell)
        self.episode_reward = 0.0
        self.episode_breakdown = {}

        obs = self._build_observation()
        info = self._build_info(action_masked=False, hard_violation=False)
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"invalid action {action}"
        self.steps_taken += 1
        prev_cell = self.agent_cell
        prev_dist = self._distance_to_goal(prev_cell)
        prev_risk = self._risk_at_cell(prev_cell)

        neighbors = self.graph.neighbors(*self.agent_cell)
        action_masked = False
        blocked_attempt = False
        wasted = False
        dispatched = False
        reroute_used = False
        moved = False

        # --- Resolve the chosen action --------------------------------
        if action in (1, 2, 3, 4):
            if action in neighbors:
                new_cell = neighbors[action]
                moved = True
                self.route_distance += 1
            else:
                action_masked = True
                blocked_attempt = True
                self.blocked_attempts += 1
                new_cell = self.agent_cell
        elif action == 5:  # REROUTE
            reroute_used = True
            target = self._safest_improving_neighbor(prev_risk)
            if target is not None and target != self.agent_cell:
                new_cell = target
                moved = True
                self.route_distance += 1
            else:
                new_cell = self.agent_cell
        elif action == 6:  # DISPATCH
            if self.resources > 0:
                self.resources -= 1
                self.vehicles += 1
                self.dispatch_steps_left = 5
                dispatched = True
            else:
                wasted = True
                self.wasted_actions += 1
            new_cell = self.agent_cell
        elif action == 7:  # PRIORITIZE
            if self.resources > 0:
                self.resources -= 1
                self.priority = min(1.0, self.priority + 0.25)
            else:
                wasted = True
                self.wasted_actions += 1
            new_cell = self.agent_cell
        else:  # STAY
            new_cell = self.agent_cell

        # --- Advance the disaster world --------------------------------
        for hz in self.hazards:
            hz.step(dt=1.0)
        events = dynamics.apply_dynamics(self, self._dynamics_rng())

        # --- Measure the destination cell ------------------------------
        risk_at_new = self._risk_at_cell(new_cell)
        hard_violation = self._is_hard_violation(new_cell)
        self.agent_cell = new_cell
        self.cumulative_risk += risk_at_new
        if hard_violation:
            self.hard_violations += 1

        if self.dispatch_steps_left > 0:
            self.dispatch_steps_left -= 1

        reward, reward_breakdown = self._compute_reward(
            new_cell, prev_cell, prev_dist, prev_risk,
            risk_at_new, hard_violation, blocked_attempt, moved,
            reroute_used, dispatched, wasted,
        )
        self.episode_reward += reward
        for k, v in reward_breakdown.items():
            self.episode_breakdown[k] = self.episode_breakdown.get(k, 0.0) + v

        distance = self._distance_to_goal(self.agent_cell)
        reached_goal = distance <= self.goal_radius_norm
        terminated = bool(reached_goal)
        truncated = bool(self.steps_taken >= self.max_steps)

        self.prev_action = action
        self._prev_distance = distance

        obs = self._build_observation()
        info = self._build_info(
            action_masked=action_masked,
            hard_violation=hard_violation,
            prev_cell=prev_cell,
            action_name=_ACTION_NAMES[action],
            events=events,
            moved=moved,
            wasted=wasted,
            dispatched=dispatched,
        )
        info["reward_breakdown"] = reward_breakdown
        info["success"] = reached_goal
        if terminated or truncated:
            info["episode_metrics"] = self._episode_metrics(reached_goal)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "ansi":
            return None
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for (r, c) in self.graph._blocked:
            grid[r][c] = "#"
        gr, gc = self.goal_cell
        grid[gr][gc] = "G"
        ar, ac = self.agent_cell
        grid[ar][ac] = "A"
        return "\n".join(" ".join(row) for row in grid)

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _distance_to_goal(self, cell: tuple[int, int]) -> float:
        nx, ny = self.graph.normalize(*cell)
        gx, gy = self.graph.normalize(*self.goal_cell)
        return float(np.hypot(nx - gx, ny - gy))

    def _risk_at_cell(self, cell: tuple[int, int]) -> float:
        nx, ny = self.graph.normalize(*cell)
        total = 0.0
        for hz in self.hazards:
            total += hz.risk_at(nx, ny)
        return float(min(total, 1.0))

    def _is_hard_violation(self, cell: tuple[int, int]) -> bool:
        nx, ny = self.graph.normalize(*cell)
        return any(hz.hard_constraint and hz.contains(nx, ny) for hz in self.hazards)

    def _shortest_feasible_path(self) -> int:
        """BFS length start->goal avoiding blocked and hard-hazard cells."""
        start, goal = self.agent_cell, self.goal_cell
        if start == goal:
            return 0
        dist = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for action_id, neighbor in self.graph.neighbors(*current).items():
                if action_id == 0 or neighbor in dist:
                    continue
                nx, ny = self.graph.normalize(*neighbor)
                if any(hz.hard_constraint and hz.contains(nx, ny) for hz in self.hazards):
                    continue
                dist[neighbor] = dist[current] + 1
                if neighbor == goal:
                    return dist[neighbor]
                queue.append(neighbor)
        return self.grid_size * 3  # unreachable -> large but finite

    def _safest_improving_neighbor(self, prev_risk: float) -> tuple[int, int] | None:
        """Best (risk, distance)-scored non-blocked neighbor for REROUTE.

        Distance is weighted more strongly than soft risk so REROUTE prefers
        goal progress; the chosen neighbor must STRICTLY improve on the
        current cell, otherwise REROUTE is a no-op (charged as wasted).
        """
        nx, ny = self.graph.normalize(*self.agent_cell)
        current_score = self._distance_to_goal(self.agent_cell) + self.weights.risk * self._risk_at_cell(self.agent_cell)
        best = None
        best_score = current_score
        for action_id, cell in self.graph.neighbors(*self.agent_cell).items():
            if action_id == 0:
                continue
            if self._is_hard_violation(cell):
                continue
            score = 2.5 * self._distance_to_goal(cell) + self.weights.risk * self._risk_at_cell(cell)
            if score < best_score:
                best_score = score
                best = cell
        return best

    def _compute_reward(
        self, new_cell, prev_cell, prev_dist, prev_risk,
        risk_at_new, hard_violation, blocked_attempt, moved,
        reroute_used, dispatched, wasted,
    ):
        """See RewardWeights docstring for the full formulation."""
        w = self.weights
        new_distance = self._distance_to_goal(new_cell)
        progress = (prev_dist - new_distance) if self._prev_distance is not None else 0.0

        time_mult = dynamics.weather_speed_multiplier(self)
        if self.dispatch_steps_left > 0:
            time_mult *= 0.5

        unnecessary_move = bool(
            moved and new_distance > prev_dist + 1e-6 and risk_at_new >= prev_risk - 1e-6
        )
        wasted_reroute = bool(
            reroute_used and (not moved or new_distance > prev_dist + 1e-6)
        )
        in_safe_zone = new_distance <= self.goal_radius_norm
        priority_scale = 0.5 + 0.5 * self.priority

        breakdown: dict[str, float] = {
            "progress": w.progress * progress,
            "distance_cost": -w.distance * new_distance,
            "time_cost": -w.time * time_mult,
            "risk_cost": -w.risk * risk_at_new,
            "traffic_cost": -(w.traffic * self.traffic_level * (0.5 if self.dispatch_steps_left > 0 else 1.0))
            if moved else 0.0,
            "blocked_penalty": -w.blocked if blocked_attempt else 0.0,
            "hard_violation_penalty": -w.hard if hard_violation else 0.0,
            "reroute_cost": -(2.0 * w.reroute_cost) if wasted_reroute else (-w.reroute_cost if reroute_used else 0.0),
            "unnecessary_move": -w.unnecessary if unnecessary_move else 0.0,
            "resource_waste": -w.resource_waste if wasted else 0.0,
            "dispatch_bonus": w.dispatch if dispatched else 0.0,
            "safe_zone_bonus": w.safe_zone if in_safe_zone else 0.0,
            "success_bonus": w.success * priority_scale if in_safe_zone else 0.0,
        }

        if in_safe_zone:
            rescued = self.victims
            breakdown["rescue_bonus"] = w.rescue * rescued * priority_scale
            if self.shortest_feasible > 0 and self.route_distance <= 1.5 * self.shortest_feasible:
                eff = 1.0 - min(1.0, self.route_distance / (2.0 * self.shortest_feasible))
                breakdown["efficiency_bonus"] = w.efficiency * eff
            else:
                breakdown["efficiency_bonus"] = 0.0
        else:
            breakdown["rescue_bonus"] = 0.0
            breakdown["efficiency_bonus"] = 0.0

        if self.steps_taken >= self.max_steps and not in_safe_zone:
            breakdown["failed_rescue"] = -w.failed_rescue * self.victims * priority_scale
        else:
            breakdown["failed_rescue"] = 0.0

        reward = float(sum(breakdown.values()))
        return reward, breakdown

    def _build_observation(self) -> np.ndarray:
        ax, ay = self.graph.normalize(*self.agent_cell)
        gx, gy = self.graph.normalize(*self.goal_cell)
        distance = self._distance_to_goal(self.agent_cell) / np.sqrt(2.0)
        risk = self._risk_at_cell(self.agent_cell)
        traffic = self.traffic_level
        severity = self.severity
        victims_frac = self.victims / max(self.victims_cap, 1)
        resources_frac = self.resources / max(self.scenario.resources if self.scenario else 4, 1)
        weather_val = dynamics.WEATHER_VALUE.get(self.weather.value, 0.0)
        time_frac = self.steps_taken / max(self.max_steps, 1)
        blocked_frac = len(self.graph._blocked) / max(self.grid_size * self.grid_size, 1)
        priority = self.priority

        hazards_sorted = sorted(self.hazards, key=lambda h: h.distance_to(ax, ay))[:MAX_HAZARDS_IN_OBS]
        hazard_features = []
        for hz in hazards_sorted:
            hazard_features.extend([
                float(np.clip((hz.x - ax) * 0.5 + 0.5, 0.0, 1.0)),
                float(np.clip((hz.y - ay) * 0.5 + 0.5, 0.0, 1.0)),
                float(np.clip(hz.severity, 0.0, 1.0)),
            ])
        while len(hazard_features) < 3 * MAX_HAZARDS_IN_OBS:
            hazard_features.append(0.0)

        prev_action_onehot = [0.0] * N_ACTIONS
        prev_action_onehot[self.prev_action] = 1.0

        obs = np.array(
            [ax, ay, gx, gy, distance, risk, traffic, severity, victims_frac,
             resources_frac, weather_val, time_frac, blocked_frac, priority,
             *hazard_features, *prev_action_onehot],
            dtype=np.float32,
        )
        return obs

    def _build_info(self, action_masked: bool, hard_violation: bool, prev_cell=None,
                    action_name: str = "STAY", events: list[str] | None = None,
                    moved: bool = False, wasted: bool = False, dispatched: bool = False) -> dict:
        return {
            "agent_cell": self.agent_cell,
            "goal_cell": self.goal_cell,
            "prev_cell": prev_cell if prev_cell is not None else self.agent_cell,
            "steps_taken": self.steps_taken,
            "difficulty": self.scenario.difficulty.value if self.scenario else None,
            "disaster_type": self.scenario.disaster_type.value if self.scenario else None,
            "action_masked": action_masked,
            "action_name": action_name,
            "moved": moved,
            "wasted": wasted,
            "dispatched": dispatched,
            "hard_violation": hard_violation,
            "cumulative_risk": self.cumulative_risk,
            "hard_violations_total": self.hard_violations,
            "events": events or [],
            "victims": self.victims,
            "victims_rescued": self.victims if self._in_goal_radius(self.agent_cell) else 0,
            "resources": self.resources,
            "vehicles": self.vehicles,
            "priority": self.priority,
            "severity": self.severity,
            "traffic_level": self.traffic_level,
            "weather": self.weather.value,
            "route_distance": self.route_distance,
            "shortest_feasible": self.shortest_feasible,
        }

    def _in_goal_radius(self, cell: tuple[int, int]) -> bool:
        return self._distance_to_goal(cell) <= self.goal_radius_norm

    def _episode_metrics(self, success: bool) -> dict:
        rescued = self.victims if success else 0
        unmet = self.victims - rescued
        total_penalty = sum(-v for v in self.episode_breakdown.values() if v < 0)
        return {
            "success": success,
            "steps": self.steps_taken,
            "response_time_s": round(self.steps_taken * TIME_PER_STEP_S, 1) if success else None,
            "rescued": rescued,
            "unmet": unmet,
            "victims": self.victims,
            "total_reward": round(self.episode_reward, 2),
            "total_penalty": round(total_penalty, 2),
            "score": round(self.episode_reward, 2),
            "route_distance": self.route_distance,
            "shortest_feasible": self.shortest_feasible,
            "route_efficiency": round(self.route_distance / self.shortest_feasible, 3)
            if success and self.shortest_feasible > 0 else None,
            "resources_used": max(0, (self.scenario.resources if self.scenario else 0) - self.resources),
            "resources_wasted": self.wasted_actions,
            "failed_actions": self.blocked_attempts + self.wasted_actions,
            "hard_violations": self.hard_violations,
            "mean_risk": round(self.cumulative_risk / max(self.steps_taken, 1), 3),
            "disaster_type": self.scenario.disaster_type.value if self.scenario else None,
            "difficulty": self.scenario.difficulty.value if self.scenario else None,
            "severity": round(self.severity, 3),
        }

    def _dynamics_rng(self) -> random.Random:
        """Deterministic per-step RNG so dynamic events are reproducible."""
        seed = (self.steps_taken * 2654435761 + (self._np_random_seed or 0)) & 0xFFFFFFFF
        return random.Random(seed)

    def valid_action_mask(self) -> np.ndarray:
        """Boolean mask of which actions are currently *possible* road moves.

        Movement into a blocked/out-of-bounds cell is impossible (masked).
        Risky-but-possible actions (entering a soft hazard, dispatching with
        no resources) are NOT masked so the policy learns from their outcomes.
        """
        neighbors = self.graph.neighbors(*self.agent_cell)
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[0] = True                      # STAY always possible
        for a in (1, 2, 3, 4):
            mask[a] = a in neighbors
        mask[5] = any(a in neighbors for a in (1, 2, 3, 4))  # REROUTE if any move exists
        mask[6] = True                      # DISPATCH (wasteful if no resources -> learnable)
        mask[7] = True                      # PRIORITIZE (same)
        return mask

    def state_dict(self) -> dict:
        """Serializable snapshot of the whole simulation state (for the API/UI)."""
        return {
            "step": self.steps_taken,
            "agent_cell": list(self.agent_cell),
            "goal_cell": list(self.goal_cell),
            "grid_size": self.grid_size,
            "max_steps": self.max_steps,
            "blocked_cells": [list(c) for c in sorted(self.graph._blocked)],
            "hazards": [hz.to_dict() for hz in self.hazards],
            "traffic": round(self.traffic_level, 3),
            "weather": self.weather.value,
            "severity": round(self.severity, 3),
            "victims": self.victims,
            "victims_rescued": self.victims if self._in_goal_radius(self.agent_cell) else 0,
            "resources": self.resources,
            "vehicles": self.vehicles,
            "priority": round(self.priority, 3),
            "disaster_type": self.scenario.disaster_type.value if self.scenario else None,
            "difficulty": self.scenario.difficulty.value if self.scenario else None,
            "shortest_feasible": self.shortest_feasible,
            "route_distance": self.route_distance,
        }