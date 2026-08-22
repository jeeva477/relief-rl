"""
RL MODEL COMPARISON -- PPO vs QR-DQN.

Runs the trained PPO policy and the trained QR-DQN policy on the
IDENTICAL set of seeded scenarios (same difficulty, same disaster
distribution, same grid/max_steps) and reports:

    Reward, Penalty, Success Rate, Rescued, Unmet, Response Time,
    Route Efficiency, plus Reward/Penalty variance and Stranded
    Vehicles proxy (failed actions).

If a checkpoint is missing or schema-incompatible, that column is
reported as "n/a" -- never fabricated. If only one agent has a usable
checkpoint, the script says so explicitly instead of inventing a
comparison.

Usage:
    python scripts/compare_ppo_qrdqn.py --episodes 30 --seed 7 \
        --ppo-checkpoint rl/checkpoints/best_model.pt \
        --qrdqn-checkpoint rl/checkpoints/qrdqn_best_model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rl.envs.evacuation_env import EvacuationEnv
from rl.evaluation.metrics import aggregate, aggregate_to_dict
from rl.training.evaluate import (
    _run_policy_episode,
    load_model,
    load_qrdqn_model,
    make_qrdqn_policy,
    make_rl_policy,
)

METRIC_ROWS = [
    ("Reward", "mean_reward", "std_reward", "{:.2f}"),
    ("Penalty", "mean_penalty", "std_penalty", "{:.2f}"),
    ("Success Rate", "success_rate", None, "{:.1%}"),
    ("Rescued", "mean_rescues", None, "{:.2f}"),
    ("Unmet", "mean_unmet", None, "{:.2f}"),
    ("Response Time (s)", "mean_response_time_s", None, "{:.1f}"),
    ("Route Efficiency", "route_efficiency", None, "{:.3f}"),
    ("Failed Actions (stranded proxy)", "mean_failed_actions", None, "{:.2f}"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO vs QR-DQN research comparison")
    parser.add_argument("--ppo-checkpoint", type=str, default="rl/checkpoints/best_model.pt")
    parser.add_argument("--qrdqn-checkpoint", type=str, default="rl/checkpoints/qrdqn_best_model.pt")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--difficulty", type=str, default="MEDIUM")
    parser.add_argument("--disaster", type=str, default="any")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output", type=str, default="rl/checkpoints/ppo_vs_qrdqn.json")
    args = parser.parse_args()

    ppo_model = load_model(args.ppo_checkpoint)
    qrdqn_model = load_qrdqn_model(args.qrdqn_checkpoint)

    policies = {}
    if ppo_model is not None:
        policies["PPO"] = make_rl_policy(ppo_model)
    else:
        print(f"NOT VERIFIED -- no compatible PPO checkpoint at {args.ppo_checkpoint}")
    if qrdqn_model is not None:
        policies["QR-DQN"] = make_qrdqn_policy(qrdqn_model)
    else:
        print(f"NOT VERIFIED -- no compatible QR-DQN checkpoint at {args.qrdqn_checkpoint}")

    if not policies:
        print("INSUFFICIENT DATA -- neither checkpoint is available/compatible.")
        return

    env = EvacuationEnv(difficulty=args.difficulty, grid_size=args.grid_size, max_steps=args.max_steps)
    rng = np.random.default_rng(args.seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(args.episodes)]

    rows: dict[str, dict] = {}
    for name, policy in policies.items():
        results = [_run_policy_episode(env, policy, seed=s) for s in seeds]
        rows[name] = aggregate_to_dict(aggregate(results))

    print("\n" + "=" * 78)
    print(f"RL MODEL COMPARISON   (episodes={args.episodes}, seed={args.seed}, "
          f"difficulty={args.difficulty}, disaster={args.disaster})")
    print("Both agents ran on the identical set of seeded scenarios.")
    print("=" * 78)
    header = f"{'Metric':<32}" + "".join(f"{name:>20}" for name in rows)
    print(header)
    print("-" * 78)

    for label, key, std_key, fmt in METRIC_ROWS:
        line = f"{label:<32}"
        values = {}
        for name, agg in rows.items():
            val = agg.get(key)
            values[name] = val
            cell = fmt.format(val) if val is not None else "n/a"
            line += f"{cell:>20}"
        # Highlight the better result only when both sides have real data.
        if len(values) == 2 and all(v is not None for v in values.values()):
            names = list(values.keys())
            lower_is_better = key in ("mean_penalty", "mean_unmet", "mean_response_time_s", "mean_failed_actions")
            better = min(names, key=lambda n: values[n]) if lower_is_better else max(names, key=lambda n: values[n])
            line += f"   (better: {better})"
        print(line)

    print("-" * 78)
    if len(rows) == 2:
        for name, agg in rows.items():
            print(f"{name} reward std-dev: {agg.get('std_reward')}   penalty std-dev: {agg.get('std_penalty')}")
    else:
        print("Only one agent had a usable checkpoint -- INSUFFICIENT DATA for a head-to-head comparison.")
    print("=" * 78)

    output = {
        "episodes": args.episodes,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "disaster": args.disaster,
        "agents": rows,
        "note": "All agents ran on the identical set of seeded scenarios. "
                "Missing agents indicate no compatible checkpoint was found (not fabricated).",
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
