"""
Integration tests: exercise multiple layers together (state builder ->
model inference; baseline vs environment; end-to-end demo scenario).
"""

import os

os.environ["DEMO_MODE"] = "true"

import numpy as np
import torch

from backend.app.schemas.hazard import HazardOut, HazardType
from backend.app.schemas.location import LatLng
from backend.app.services.state_builder import build_observation
from rl.baselines.safety_heuristic import safety_heuristic_action
from rl.baselines.shortest_path import shortest_safe_path_action
from rl.envs.evacuation_env import EvacuationEnv
from rl.models.actor_critic import ActorCritic


def test_state_builder_produces_env_compatible_observation():
    env = EvacuationEnv()
    obs_dim = env.observation_space.shape[0]

    hazard = HazardOut(
        id="h1", location=LatLng(latitude=11.343, longitude=77.719), radius_m=300,
        severity=0.6, hazard_type=HazardType.FLOOD, hard_constraint=False, active=True,
    )
    obs = build_observation(
        current_location=LatLng(latitude=11.341, longitude=77.717),
        destination=LatLng(latitude=11.350, longitude=77.725),
        hazards=[hazard],
        traffic_factor=0.3,
    )
    assert obs.shape == (obs_dim,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)


def test_real_world_observation_feeds_trained_model_without_error():
    env = EvacuationEnv()
    obs_dim = env.observation_space.shape[0]
    model = ActorCritic(obs_dim=obs_dim, n_actions=env.action_space.n, hidden_dim=32)

    obs = build_observation(
        current_location=LatLng(latitude=11.341, longitude=77.717),
        destination=LatLng(latitude=11.350, longitude=77.725),
        hazards=[],
        traffic_factor=0.2,
    )
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        action, _, value = model.get_action(obs_t, deterministic=True)
    assert 0 <= int(action.item()) < env.action_space.n


def test_baselines_never_choose_hard_hazard_cell():
    env = EvacuationEnv(grid_size=10, difficulty="HARD")
    for seed in range(5):
        env.reset(seed=seed)
        for _ in range(20):
            action = shortest_safe_path_action(env)
            neighbors = env.graph.neighbors(*env.agent_cell)
            target = neighbors.get(action, env.agent_cell)
            tx, ty = env.graph.normalize(*target)
            assert not any(hz.hard_constraint and hz.contains(tx, ty) for hz in env.hazards)

            heuristic_action = safety_heuristic_action(env)
            target2 = neighbors.get(heuristic_action, env.agent_cell)
            tx2, ty2 = env.graph.normalize(*target2)
            assert not any(hz.hard_constraint and hz.contains(tx2, ty2) for hz in env.hazards)

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break


def test_end_to_end_demo_scenario_hazard_forces_reroute():
    """
    Deterministic demo scenario (Section 57): create a scenario where a
    hazard sits between the agent and the goal, and confirm the
    rule-based baseline routes around it rather than through it.
    """
    env = EvacuationEnv(grid_size=8, difficulty="EASY", max_steps=60)
    env.reset(seed=99)

    from rl.envs.hazard import Hazard
    mid_row = (env.agent_cell[0] + env.goal_cell[0]) // 2
    mid_col = (env.agent_cell[1] + env.goal_cell[1]) // 2
    hx, hy = env.graph.normalize(mid_row, mid_col)
    env.hazards = [Hazard(id="blocker", x=hx, y=hy, radius=0.4, severity=1.0, hard_constraint=True)]

    hard_violations = 0
    for _ in range(env.max_steps):
        action = safety_heuristic_action(env)
        _, _, terminated, truncated, info = env.step(action)
        hard_violations += int(info["hard_violation"])
        if terminated or truncated:
            break

    assert hard_violations == 0
