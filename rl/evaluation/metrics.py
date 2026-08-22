"""
Research metrics.

    success_rate        = successful_episodes / total_episodes
    violation_rate      = hard_safety_violations / total_episodes
    hazard_exposure     = mean cumulative risk experienced per episode
    response_time_s     = mean steps-to-success converted to seconds
                          (successful episodes only)
    route_efficiency    = mean(actual_distance / shortest_feasible_distance)
    mean_rescues        = mean victims rescued per episode
    mean_resource_usage = mean resource units spent per episode
    mean_failed_actions = mean blocked/wasted actions per episode
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl.envs.evacuation_env import TIME_PER_STEP_S


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    total_reward: float
    hard_violations: int
    cumulative_risk: float
    unsafe_action_rate: float
    timed_out: bool
    route_distance: float | None = None
    shortest_feasible_distance: float | None = None
    rescues: int = 0
    victims: int = 0
    resources_used: int = 0
    resources_wasted: int = 0
    blocked_attempts: int = 0
    response_time_s: float | None = None
    total_penalty: float = 0.0
    unmet: int = 0


@dataclass
class AggregateMetrics:
    n_episodes: int
    success_rate: float
    mean_reward: float
    mean_steps: float
    mean_hazard_exposure: float
    violation_rate: float
    unsafe_action_rate: float
    timeout_rate: float
    route_efficiency: float | None
    std_reward: float = 0.0
    std_success_rate: float = 0.0
    mean_response_time_s: float | None = None
    std_response_time_s: float | None = None
    mean_rescues: float = 0.0
    mean_unmet: float = 0.0
    mean_resource_usage: float = 0.0
    mean_failed_actions: float = 0.0
    mean_distance: float = 0.0
    mean_penalty: float = 0.0
    std_penalty: float = 0.0
    mean_unmet_reported: float = 0.0


def aggregate(results: list[EpisodeResult]) -> AggregateMetrics:
    n = len(results)
    if n == 0:
        raise ValueError("No episode results to aggregate")

    successes = [r.success for r in results]
    rewards = [r.total_reward for r in results]
    steps = [r.steps for r in results]
    risks = [r.cumulative_risk for r in results]
    violations = [1 if r.hard_violations > 0 else 0 for r in results]
    unsafe_rates = [r.unsafe_action_rate for r in results]
    timeouts = [1 if r.timed_out else 0 for r in results]

    efficiencies = [
        r.route_distance / r.shortest_feasible_distance
        for r in results
        if r.route_distance and r.shortest_feasible_distance and r.shortest_feasible_distance > 0
    ]
    response_times = [r.response_time_s for r in results if r.response_time_s is not None]
    rescues = [r.rescues for r in results]
    unmet = [max(0, r.victims - r.rescues) for r in results]
    resources = [r.resources_used + r.resources_wasted for r in results]
    failed = [r.blocked_attempts + r.resources_wasted for r in results]
    distances = [r.route_distance or 0.0 for r in results]
    penalties = [r.total_penalty for r in results]
    unmet_reported = [r.unmet for r in results]

    return AggregateMetrics(
        n_episodes=n,
        success_rate=float(np.mean(successes)),
        mean_reward=float(np.mean(rewards)),
        mean_steps=float(np.mean(steps)),
        mean_hazard_exposure=float(np.mean(risks)),
        violation_rate=float(np.mean(violations)),
        unsafe_action_rate=float(np.mean(unsafe_rates)),
        timeout_rate=float(np.mean(timeouts)),
        route_efficiency=float(np.mean(efficiencies)) if efficiencies else None,
        std_reward=float(np.std(rewards)),
        std_success_rate=float(np.std(successes)),
        mean_response_time_s=float(np.mean(response_times)) if response_times else None,
        std_response_time_s=float(np.std(response_times)) if response_times else None,
        mean_rescues=float(np.mean(rescues)),
        mean_unmet=float(np.mean(unmet)),
        mean_resource_usage=float(np.mean(resources)),
        mean_failed_actions=float(np.mean(failed)),
        mean_distance=float(np.mean(distances)),
        mean_penalty=float(np.mean(penalties)),
        std_penalty=float(np.std(penalties)),
        mean_unmet_reported=float(np.mean(unmet_reported)),
    )


def aggregate_to_dict(metrics: AggregateMetrics) -> dict:
    return {
        "n_episodes": metrics.n_episodes,
        "success_rate": round(metrics.success_rate, 4),
        "mean_reward": round(metrics.mean_reward, 3),
        "std_reward": round(metrics.std_reward, 3),
        "mean_steps": round(metrics.mean_steps, 2),
        "mean_response_time_s": round(metrics.mean_response_time_s, 2)
        if metrics.mean_response_time_s is not None else None,
        "mean_rescues": round(metrics.mean_rescues, 2),
        "mean_unmet": round(metrics.mean_unmet, 2),
        "mean_resource_usage": round(metrics.mean_resource_usage, 2),
        "mean_failed_actions": round(metrics.mean_failed_actions, 2),
        "mean_distance": round(metrics.mean_distance, 2),
        "route_efficiency": round(metrics.route_efficiency, 3)
        if metrics.route_efficiency is not None else None,
        "violation_rate": round(metrics.violation_rate, 4),
        "timeout_rate": round(metrics.timeout_rate, 4),
        "mean_penalty": round(metrics.mean_penalty, 3),
        "std_penalty": round(metrics.std_penalty, 3),
        "mean_unmet": round(metrics.mean_unmet_reported, 2),
    }


def multi_seed_summary(per_seed_metrics: list[AggregateMetrics]) -> dict:
    """Report mean +/- std across multiple random seeds."""
    fields = {
        "success_rate": [m.success_rate for m in per_seed_metrics],
        "reward": [m.mean_reward for m in per_seed_metrics],
        "violation_rate": [m.violation_rate for m in per_seed_metrics],
        "response_time_s": [m.mean_response_time_s or 0.0 for m in per_seed_metrics],
        "rescues": [m.mean_rescues for m in per_seed_metrics],
        "route_efficiency": [m.route_efficiency or 0.0 for m in per_seed_metrics],
        "resource_usage": [m.mean_resource_usage for m in per_seed_metrics],
        "failed_actions": [m.mean_failed_actions for m in per_seed_metrics],
    }
    return {
        "n_seeds": len(per_seed_metrics),
        **{
            f"{name}_mean": float(np.mean(values))
            for name, values in fields.items()
        },
        **{
            f"{name}_std": float(np.std(values))
            for name, values in fields.items()
        },
    }


def time_to_steps(seconds: float) -> int:
    """Convert response time in seconds back to environment steps."""
    return int(round(seconds / TIME_PER_STEP_S))