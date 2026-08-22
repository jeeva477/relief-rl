"""Reproducible Relief-RL training experiment runner.

This orchestration script keeps research runs separate from the 60-episode
smoke-test checkpoint. It trains one independent model per
(difficulty, seed) pair and stores metadata/checkpoints under an experiment
folder that can be archived for a paper or project report.

Example (small local validation):
    python scripts/run_research.py --episodes 200 --seeds 1 --difficulties EASY

Example (research run):
    python scripts/run_research.py --episodes 2000 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import os
import time

from rl.training.config import TrainingConfig
from rl.training.train import train

ALL_DIFFICULTIES = ["EASY", "MEDIUM", "HARD", "EXTREME"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible Relief-RL research experiments")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--difficulties", nargs="+", choices=ALL_DIFFICULTIES, default=ALL_DIFFICULTIES)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--checkpoint-frequency", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()

    if args.episodes < 1 or args.seeds < 1:
        parser.error("--episodes and --seeds must be >= 1")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {
        "episodes_per_run": args.episodes,
        "seeds": [args.seed_start + i for i in range(args.seeds)],
        "difficulties": args.difficulties,
        "grid_size": args.grid_size,
        "max_steps": args.max_steps,
        "checkpoint_frequency": args.checkpoint_frequency,
        "runs": [],
    }

    started = time.time()
    for difficulty in args.difficulties:
        for seed in manifest["seeds"]:
            run_name = f"{difficulty.lower()}_seed_{seed}"
            checkpoint_dir = os.path.join(args.output_dir, run_name)
            print(f"\n=== Training {run_name} ===")
            config = TrainingConfig(
                episodes=args.episodes,
                difficulty=difficulty,
                seed=seed,
                grid_size=args.grid_size,
                max_steps=args.max_steps,
                checkpoint_frequency=args.checkpoint_frequency,
                checkpoint_dir=checkpoint_dir,
            )
            checkpoint = train(config)
            manifest["runs"].append({
                "run": run_name,
                "difficulty": difficulty,
                "seed": seed,
                "checkpoint": checkpoint,
                "run_config": os.path.join(checkpoint_dir, "run_config.json"),
                "training_log": os.path.join(checkpoint_dir, "training_log.json"),
            })

    manifest["elapsed_seconds"] = time.time() - started
    manifest_path = os.path.join(args.output_dir, "experiment_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nResearch experiment complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
