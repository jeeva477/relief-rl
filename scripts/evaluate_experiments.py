"""Evaluate all trained experiment checkpoints and create one research summary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rl.training.evaluate import DIFFICULTIES, evaluate_policy, heuristic_policy, load_model, make_rl_policy, shortest_path_policy
from rl.envs.evacuation_env import EvacuationEnv
from rl.evaluation.metrics import multi_seed_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default="experiments")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", default="experiments/research_evaluation.json")
    args = parser.parse_args()

    root = Path(args.experiment_dir)
    run_configs = sorted(root.glob("*/run_config.json"))
    if not run_configs:
        raise SystemExit(f"No experiment runs found under {root}")

    summary: dict = {"episodes_per_evaluation": args.episodes, "runs": []}
    for config_path in run_configs:
        with config_path.open(encoding="utf-8") as f:
            config = json.load(f)
        cfg = config.get("config", {})
        difficulty = cfg.get("difficulty", "MEDIUM")
        seed = int(cfg.get("seed", 0))
        checkpoint = config_path.parent / "best_model.pt"

        env_probe = EvacuationEnv(
            difficulty=difficulty,
            grid_size=int(cfg.get("grid_size", 10)),
            max_steps=int(cfg.get("max_steps", 100)),
        )
        model = load_model(str(checkpoint), env_probe.observation_space.shape[0])
        if model is None:
            continue

        policies = {
            "ShortestSafe": shortest_path_policy,
            "RuleHeuristic": heuristic_policy,
            "ActorCritic": make_rl_policy(model),
        }
        metrics = {}
        for name, policy in policies.items():
            result = evaluate_policy(
                policy,
                difficulty,
                args.episodes,
                seed,
                int(cfg.get("grid_size", 10)),
                int(cfg.get("max_steps", 100)),
            )
            metrics[name] = {
                "success_rate": result.success_rate,
                "mean_reward": result.mean_reward,
                "mean_steps": result.mean_steps,
                "mean_hazard_exposure": result.mean_hazard_exposure,
                "violation_rate": result.violation_rate,
                "unsafe_action_rate": result.unsafe_action_rate,
                "timeout_rate": result.timeout_rate,
            }

        summary["runs"].append({
            "run": config_path.parent.name,
            "difficulty": difficulty,
            "seed": seed,
            "checkpoint": str(checkpoint),
            "policies": metrics,
        })

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved consolidated research evaluation to {args.output}")


if __name__ == "__main__":
    main()
