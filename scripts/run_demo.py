"""
Deterministic end-to-end demo (Section 57).

Runs: train a tiny model (if none exists) -> print a scripted decision
sequence showing a hazard forcing a reroute -> summary.

Usage:
    python scripts/run_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.baselines.safety_heuristic import safety_heuristic_action
from rl.envs.evacuation_env import EvacuationEnv
from rl.envs.hazard import Hazard


def main():
    print("=" * 70)
    print("SafeRoute-RL DEMO SCENARIO")
    print("=" * 70)

    env = EvacuationEnv(grid_size=8, difficulty="EASY", max_steps=40, render_mode="ansi")
    env.reset(seed=42)
    print(f"Start: {env.agent_cell}  Goal: {env.goal_cell}")
    print(env.render())

    print("\nStep 1: No hazard yet. Route heads directly toward the goal.\n")

    # Inject a hard-constraint hazard directly in the agent's path,
    # simulating an expanding wildfire/flood cutting off the natural route.
    mid_row = (env.agent_cell[0] + env.goal_cell[0]) // 2
    mid_col = (env.agent_cell[1] + env.goal_cell[1]) // 2
    hx, hy = env.graph.normalize(mid_row, mid_col)
    env.hazards = [Hazard(id="DEMO-HZ", x=hx, y=hy, radius=0.35, severity=1.0,
                           hard_constraint=True, growth_rate=0.01)]
    print(f"Step 2: Hazard DEMO-HZ appears near cell ({mid_row}, {mid_col}) and begins expanding.\n")

    step = 0
    while step < env.max_steps:
        action = safety_heuristic_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        if step <= 5 or terminated or truncated:
            print(f"  step={step:2d} agent={env.agent_cell} hard_violation={info['hard_violation']} "
                  f"reward={reward:.2f}")
        if terminated or truncated:
            break

    print("\n" + env.render())
    if info.get("success"):
        print(f"\nResult: SAFE_ROUTE reached goal in {step} steps with 0 hard safety violations "
              f"({env.hard_violations} total).")
    else:
        print(f"\nResult: episode ended without reaching the goal in {step} steps "
              f"(hard_violations={env.hard_violations}). This can happen -- the rule-based "
              f"heuristic baseline is not guaranteed optimal; see evaluate.py for the trained "
              f"Actor-Critic's comparative performance.")
    print("=" * 70)


if __name__ == "__main__":
    main()
