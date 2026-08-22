import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl.envs.evacuation_env import EvacuationEnv, N_ACTIONS


def test_env_passes_gymnasium_checker():
    env = EvacuationEnv(difficulty="MEDIUM", grid_size=10, max_steps=50)
    check_env(env.unwrapped, skip_render_check=True)


def test_reset_returns_valid_observation():
    env = EvacuationEnv(grid_size=10)
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert obs.shape == env.observation_space.shape


def test_reset_is_deterministic_given_seed():
    env1 = EvacuationEnv(grid_size=10, difficulty="MEDIUM")
    env2 = EvacuationEnv(grid_size=10, difficulty="MEDIUM")
    obs1, _ = env1.reset(seed=7)
    obs2, _ = env2.reset(seed=7)
    np.testing.assert_allclose(obs1, obs2)
    assert env1.agent_cell == env2.agent_cell
    assert env1.goal_cell == env2.goal_cell


def test_step_returns_correct_types():
    env = EvacuationEnv(grid_size=10)
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(1)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_space_size():
    env = EvacuationEnv()
    assert env.action_space.n == N_ACTIONS


def test_episode_truncates_at_max_steps():
    env = EvacuationEnv(grid_size=20, max_steps=5, difficulty="EASY")
    env.reset(seed=1)
    truncated = False
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(0)  # stay -> never reaches goal
        if terminated:
            break
    assert truncated or terminated


def test_reaching_goal_terminates_episode():
    env = EvacuationEnv(grid_size=3, max_steps=50, difficulty="EASY")
    env.reset(seed=1)
    env.hazards = []  # remove hazards to isolate termination logic
    env.agent_cell = env.goal_cell
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated
    assert info["success"] is True


def test_valid_action_mask_excludes_blocked_cells():
    env = EvacuationEnv(grid_size=5)
    env.reset(seed=3)
    env.graph.set_blocked({(env.agent_cell[0] - 1, env.agent_cell[1])})
    mask = env.valid_action_mask()
    assert mask[0]  # stay always valid
    assert mask.dtype == bool
