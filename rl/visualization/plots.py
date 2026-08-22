"""Plotting utilities (Section 33). Saves PNGs under artifacts/plots/."""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _smooth(values, window=10):
    if len(values) < window:
        return values
    return [sum(values[max(0, i - window):i + 1]) / len(values[max(0, i - window):i + 1]) for i in range(len(values))]


def plot_training_curves(training_log_path: str, output_dir: str = "artifacts/plots"):
    os.makedirs(output_dir, exist_ok=True)
    with open(training_log_path) as f:
        log = json.load(f)

    episodes = [r["episode"] for r in log]
    series = {
        "reward": [r["reward"] for r in log],
        "policy_loss": [r["policy_loss"] for r in log],
        "value_loss": [r["value_loss"] for r in log],
        "entropy": [r["entropy"] for r in log],
        "success": [1.0 if r["success"] else 0.0 for r in log],
        "unsafe_rate": [r["unsafe_rate"] for r in log],
        "mean_risk": [r["mean_risk"] for r in log],
        "steps": [r["steps"] for r in log],
    }

    for name, values in series.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(episodes, values, alpha=0.3, label="raw")
        ax.plot(episodes, _smooth(values), label="smoothed")
        ax.set_xlabel("Episode")
        ax.set_ylabel(name.replace("_", " ").title())
        ax.set_title(f"{name.replace('_', ' ').title()} vs Episode")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(output_dir, f"{name}_vs_episode.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)

    print(f"Saved {len(series)} training curve plots to {output_dir}")


def plot_baseline_comparison(evaluation_results_path: str, output_dir: str = "artifacts/plots"):
    os.makedirs(output_dir, exist_ok=True)
    with open(evaluation_results_path) as f:
        results = json.load(f)

    difficulties = list(results.keys())
    policies = list(next(iter(results.values())).keys())
    metric = "success_rate"

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(policies), 1)
    x = range(len(difficulties))
    for i, policy in enumerate(policies):
        values = [results[d][policy][metric] for d in difficulties]
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, values, width=width, label=policy)

    ax.set_xticks([xi + width * (len(policies) - 1) / 2 for xi in x])
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Difficulty and Policy")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "baseline_comparison_success_rate.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved baseline comparison plot to {path}")
