from rl.envs.evacuation_env import EvacuationEnv


def test_staying_still_is_penalized_relative_to_progress():
    """The reward function must not make 'stay' the optimal policy."""
    env = EvacuationEnv(grid_size=10, difficulty="EASY", max_steps=50)
    env.reset(seed=1)
    env.hazards = []  # isolate progress/time incentives from hazard noise

    _, stay_reward, _, _, _ = env.step(0)

    env.reset(seed=1)
    env.hazards = []
    neighbors = env.graph.neighbors(*env.agent_cell)
    move_actions = [a for a in neighbors if a != 0]
    assert move_actions, "expected at least one valid move action"

    # find a move that reduces distance to goal
    best_action = None
    best_progress = -1
    for a in move_actions:
        cell = neighbors[a]
        d_before = env._distance_to_goal(env.agent_cell)
        d_after = env._distance_to_goal(cell)
        if d_before - d_after > best_progress:
            best_progress = d_before - d_after
            best_action = a

    _, move_reward, _, _, _ = env.step(best_action)
    if best_progress > 0:
        assert move_reward > stay_reward


def test_hard_hazard_violation_incurs_large_penalty():
    env = EvacuationEnv(grid_size=10, difficulty="EASY", max_steps=50)
    env.reset(seed=1)
    from rl.envs.hazard import Hazard
    ax, ay = env.graph.normalize(*env.agent_cell)
    neighbors = env.graph.neighbors(*env.agent_cell)
    move_action = next(a for a in neighbors if a != 0)
    target_cell = neighbors[move_action]
    tx, ty = env.graph.normalize(*target_cell)

    env.hazards = [Hazard(id="hard", x=tx, y=ty, radius=0.5, severity=1.0, hard_constraint=True)]
    _, reward, _, _, info = env.step(move_action)
    assert info["hard_violation"] is True
    assert reward < 0


def test_reward_breakdown_keys_present():
    env = EvacuationEnv(grid_size=10)
    env.reset(seed=2)
    _, _, _, _, info = env.step(1)
    breakdown = info["reward_breakdown"]
    for key in ["progress", "distance_cost", "time_cost", "risk_cost", "hard_violation_penalty",
                "safe_zone_bonus", "success_bonus"]:
        assert key in breakdown
