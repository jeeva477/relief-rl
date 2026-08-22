"""Generate publication-style research plots from consolidated evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _runs_by_difficulty(data: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for run in data.get("runs", []):
        grouped.setdefault(run["difficulty"], []).append(run)
    return grouped


def _mean_std(group: list[dict], policy: str, metric: str) -> tuple[float, float]:
    values = [r["policies"][policy][metric] for r in group if policy in r["policies"]]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _save_bar(path: Path, title: str, ylabel: str, grouped: dict[str, list[dict]], metric: str):
    difficulties = list(grouped)
    policies = ["ActorCritic", "ShortestSafe", "RuleHeuristic"]
    x = np.arange(len(difficulties))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, policy in enumerate(policies):
        means, stds = [], []
        for difficulty in difficulties:
            mean, std = _mean_std(grouped[difficulty], policy, metric)
            means.append(mean)
            stds.append(std)
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=4, label=policy)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_report(input_path: str, output_dir: str) -> None:
    data = _load(input_path)
    grouped = _runs_by_difficulty(data)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _save_bar(out / "success_rate_comparison.png", "Evacuation Success Rate by Difficulty", "Success rate", grouped, "success_rate")
    _save_bar(out / "reward_comparison.png", "Mean Episode Reward by Difficulty", "Mean reward", grouped, "mean_reward")
    _save_bar(out / "hazard_exposure_comparison.png", "Hazard Exposure by Difficulty", "Mean hazard exposure", grouped, "mean_hazard_exposure")
    _save_bar(out / "violation_rate_comparison.png", "Safety Violation Rate by Difficulty", "Violation rate", grouped, "violation_rate")
    _save_bar(out / "unsafe_action_comparison.png", "Unsafe Action Rate by Difficulty", "Unsafe action rate", grouped, "unsafe_action_rate")

    # Consolidated CSV-like JSON summary for easy inclusion in reports.
    summary = []
    for difficulty, runs in grouped.items():
        for policy in ["ActorCritic", "ShortestSafe", "RuleHeuristic"]:
            if not any(policy in r["policies"] for r in runs):
                continue
            row = {"difficulty": difficulty, "policy": policy}
            for metric in ["success_rate", "mean_reward", "mean_steps", "mean_hazard_exposure", "violation_rate", "unsafe_action_rate", "timeout_rate"]:
                mean, std = _mean_std(runs, policy, metric)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            summary.append(row)

    with (out / "research_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Research plots and summary written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="experiments/research_evaluation.json")
    parser.add_argument("--output-dir", default="artifacts/research")
    args = parser.parse_args()
    generate_report(args.input, args.output_dir)


if __name__ == "__main__":
    main()
