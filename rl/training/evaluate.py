"""Evaluation pipeline for Relief-RL.

Evaluates the trained PPO policy against baselines:

    random         uniform over valid actions (untrained behaviour)
    ShortestSafe   BFS shortest path avoiding blocked roads + hard hazards
    RuleHeuristic  greedy (hard-avoid, risk, distance) heuristic

across EASY/MEDIUM/HARD/EXTREME and disaster types. Supports repeated
random seeds and reports mean +/- standard deviation so a single run is
not presented as definitive.

UNSEEN SCENARIO TESTING
-----------------------
Scenario generation is randomized per episode from a seed. Evaluation
uses an independent seed range (--seed-offset, default 100000) that is
never used during training, so the reported numbers measure how well the
policy generalizes to scenarios it has not seen.

Example:
    python scripts/evaluate.py --checkpoint rl/checkpoints/best_model.pt \
        --episodes 50 --seeds 5 --seed 123
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

from rl.baselines.safety_heuristic import safety_heuristic_action
from rl.baselines.shortest_path import shortest_safe_path_action
from rl.envs.evacuation_env import EvacuationEnv, N_ACTIONS, OBS_DIM
from rl.evaluation.metrics import EpisodeResult, aggregate, aggregate_to_dict, multi_seed_summary
from rl.models.actor_critic import ActorCritic
from rl.models.qrdqn_net import QuantileNetwork
from rl.training.seeding import set_seed

DIFFICULTIES = ["EASY", "MEDIUM", "HARD", "EXTREME"]
DISASTERS = ["any", "flood", "wildfire", "earthquake", "cyclone", "tsunami",
             "landslide", "heavy_rain", "road_blockage", "traffic_jam", "combined"]


def load_model(checkpoint_path: str, obs_dim: int = OBS_DIM) -> ActorCritic | None:
    if not os.path.exists(checkpoint_path):
        return None
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_obs_dim = int(ckpt.get("obs_dim", 0))
    ckpt_n_actions = int(ckpt.get("n_actions", 0))
    if ckpt_obs_dim != obs_dim or ckpt_n_actions != N_ACTIONS:
        return None  # schema mismatch -> caller reports honestly
    model = ActorCritic(
        obs_dim=ckpt_obs_dim,
        n_actions=ckpt_n_actions,
        hidden_dim=ckpt.get("hidden_dim", 128),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _run_policy_episode(env: EvacuationEnv, policy_fn, seed: int) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    unsafe_actions = 0
    steps = 0
    total_reward = 0.0
    success = False
    timed_out = False
    start_distance = env._distance_to_goal(env.agent_cell)

    for _ in range(env.max_steps):
        action = policy_fn(env, obs)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        steps += 1
        if step_info.get("hard_violation"):
            unsafe_actions += 1
        if terminated or truncated:
            success = bool(step_info.get("success", False))
            timed_out = bool(truncated and not success)
            break

    ep_metrics = env._episode_metrics(success)
    return EpisodeResult(
        success=success,
        steps=steps,
        total_reward=total_reward,
        hard_violations=env.hard_violations,
        cumulative_risk=env.cumulative_risk,
        unsafe_action_rate=unsafe_actions / max(steps, 1),
        timed_out=timed_out,
        route_distance=float(env.route_distance),
        shortest_feasible_distance=float(env.shortest_feasible)
        if env.shortest_feasible and env.shortest_feasible < env.grid_size * 3 else None,
        rescues=ep_metrics.get("rescued", 0),
        victims=ep_metrics.get("victims", 0),
        resources_used=ep_metrics.get("resources_used", 0),
        resources_wasted=ep_metrics.get("resources_wasted", 0),
        blocked_attempts=ep_metrics.get("failed_actions", 0) - ep_metrics.get("resources_wasted", 0),
        response_time_s=ep_metrics.get("response_time_s"),
        total_penalty=ep_metrics.get("total_penalty", 0.0),
        unmet=ep_metrics.get("unmet", 0),
    )


def make_rl_policy(model: ActorCritic):
    def policy(env: EvacuationEnv, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(env.valid_action_mask()).unsqueeze(0)
        action, _, _ = model.get_action(obs_t, action_mask=mask, deterministic=True)
        return int(action.item())
    return policy


def load_qrdqn_model(checkpoint_path: str, obs_dim: int = OBS_DIM) -> QuantileNetwork | None:
    """Load a QR-DQN checkpoint saved by rl/training/train_qrdqn.py.
    Returns None (never a fabricated model) if the file is missing or the
    checkpoint schema doesn't match the current env's obs/action space."""
    if not os.path.exists(checkpoint_path):
        return None
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_obs_dim = int(ckpt.get("obs_dim", 0))
    ckpt_n_actions = int(ckpt.get("n_actions", 0))
    if ckpt_obs_dim != obs_dim or ckpt_n_actions != N_ACTIONS:
        return None
    net = QuantileNetwork(
        obs_dim=ckpt_obs_dim,
        n_actions=ckpt_n_actions,
        n_quantiles=ckpt.get("n_quantiles", 51),
        hidden_dim=ckpt.get("hidden_dim", 128),
    )
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net


def make_qrdqn_policy(net: QuantileNetwork):
    """Deterministic (greedy, epsilon=0) policy over expected Q-values --
    the QR-DQN analogue of make_rl_policy for PPO."""
    def policy(env: EvacuationEnv, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(env.valid_action_mask()).unsqueeze(0)
        action = net.act(obs_t, action_mask=mask, deterministic=True)
        return int(action.item())
    return policy


def random_policy(env: EvacuationEnv, obs):
    mask = env.valid_action_mask()
    valid = np.flatnonzero(mask)
    return int(env.np_random.choice(valid))


def shortest_path_policy(env: EvacuationEnv, obs):
    return shortest_safe_path_action(env)


def heuristic_policy(env: EvacuationEnv, obs):
    return safety_heuristic_action(env)


def evaluate_policy(policy_fn, difficulty: str, episodes: int, seed: int,
                    grid_size: int, max_steps: int, disaster: str = "any"):
    env = EvacuationEnv(difficulty=difficulty, grid_size=grid_size, max_steps=max_steps)
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(episodes):
        episode_seed = int(rng.integers(0, 2**31 - 1))
        options = None
        if disaster != "any":
            options = {"scenario_config": {"disaster_type": disaster,
                                           "difficulty": difficulty,
                                           "grid_size": grid_size,
                                           "max_steps": max_steps}}
        env.reset(seed=episode_seed, options=options)
        obs = env._build_observation()
        results.append(_run_policy_episode(env, policy_fn, episode_seed))
    return aggregate(results)


def print_comparison_table(results_by_policy: dict[str, object], title: str = ""):
    policies = list(results_by_policy.keys())
    print("\n" + "=" * 100)
    if title:
        print(title)
        print("-" * 100)
    print(f"{'Metric':<24}" + "".join(f"{p:>19}" for p in policies))
    print("-" * 100)
    rows = [
        ("Success Rate", "success_rate", "{:.2%}"),
        ("Mean Reward", "mean_reward", "{:.2f}"),
        ("Mean Steps", "mean_steps", "{:.1f}"),
        ("Response Time (s)", "mean_response_time_s", "{:.1f}"),
        ("Rescues", "mean_rescues", "{:.1f}"),
        ("Route Efficiency", "route_efficiency", "{:.3f}"),
        ("Resource Usage", "mean_resource_usage", "{:.1f}"),
        ("Failed Actions", "mean_failed_actions", "{:.1f}"),
        ("Hazard Exposure", "mean_hazard_exposure", "{:.3f}"),
        ("Violation Rate", "violation_rate", "{:.2%}"),
        ("Timeout Rate", "timeout_rate", "{:.2%}"),
    ]
    for label, key, fmt in rows:
        line = f"{label:<24}"
        for p in policies:
            val = getattr(results_by_policy[p], key)
            line += f"{fmt.format(val) if val is not None else 'n/a':>19}"
        print(line)
    print("=" * 100 + "\n")


def compare_agents(
    model: ActorCritic | None,
    episodes: int,
    seed: int,
    difficulty: str,
    grid_size: int = 10,
    max_steps: int = 100,
    disaster: str = "any",
) -> dict:
    """
    BEFORE vs AFTER learning: run the untrained (random) policy and the
    trained PPO policy on the IDENTICAL set of seeded scenarios and return
    a side-by-side table. All numbers come from actual environment runs.
    """
    policies = {"Untrained (random)": random_policy}
    if model is not None:
        policies["Trained PPO"] = make_rl_policy(model)

    env = EvacuationEnv(difficulty=difficulty, grid_size=grid_size, max_steps=max_steps)
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(episodes)]

    rows = {}
    for name, policy in policies.items():
        results = []
        for episode_seed in seeds:
            results.append(_run_policy_episode(env, policy, episode_seed))
        rows[name] = aggregate_to_dict(aggregate(results))

    return {
        "episodes": episodes,
        "seed": seed,
        "difficulty": difficulty,
        "disaster": disaster,
        "policies": rows,
        "note": "Both policies ran on the identical set of seeded scenarios.",
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Relief-RL against safety baselines")
    parser.add_argument("--checkpoint", type=str, default="rl/checkpoints/best_model.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=1, help="Number of independent evaluation seeds")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seed-offset", type=int, default=100_000,
                        help="unseen-scenario seed offset (never used in training)")
    parser.add_argument("--difficulty", type=str, default=None, choices=DIFFICULTIES,
                        help="restrict to one difficulty (default: all)")
    parser.add_argument("--disaster", type=str, default="any", choices=DISASTERS)
    parser.add_argument("--output", type=str, default="rl/checkpoints/evaluation_results.json")
    parser.add_argument("--csv", type=str, default="rl/checkpoints/evaluation_results.csv")
    args = parser.parse_args()

    if args.episodes < 1 or args.seeds < 1:
        parser.error("--episodes and --seeds must be >= 1")

    model = load_model(args.checkpoint)
    if model is None:
        print(f"WARNING: no compatible checkpoint at {args.checkpoint}; evaluating baselines only.")

    difficulties = [args.difficulty] if args.difficulty else DIFFICULTIES
    all_results: dict[str, dict] = {}
    csv_rows: list[dict] = []

    for difficulty in difficulties:
        policies = {
            "Random": random_policy,
            "ShortestSafe": shortest_path_policy,
            "RuleHeuristic": heuristic_policy,
        }
        if model is not None:
            policies["PPO"] = make_rl_policy(model)

        per_policy_seed_metrics: dict[str, list] = {name: [] for name in policies}
        for seed_offset in range(args.seeds):
            eval_seed = args.seed + seed_offset
            set_seed(eval_seed)
            for name, policy in policies.items():
                per_policy_seed_metrics[name].append(
                    evaluate_policy(policy, difficulty, args.episodes, eval_seed + args.seed_offset,
                                    args.grid_size, args.max_steps, args.disaster)
                )

        summary = {name: multi_seed_summary(metrics) for name, metrics in per_policy_seed_metrics.items()}
        all_results[difficulty] = {
            "episodes_per_seed": args.episodes,
            "n_seeds": args.seeds,
            "base_seed": args.seed,
            "seed_offset": args.seed_offset,
            "disaster": args.disaster,
            "unseen_scenarios": True,
            "policies": summary,
        }

        if args.seeds == 1:
            print_comparison_table(
                {name: metrics[0] for name, metrics in per_policy_seed_metrics.items()},
                title=f"difficulty={difficulty} disaster={args.disaster} (unseen seed range)",
            )
        else:
            print(f"\nDifficulty={difficulty} | {args.seeds} independent seeds (unseen range)")
            for name, values in summary.items():
                print(
                    f"  {name:<18} success={values['success_rate_mean']:.2%} "
                    f"+/- {values['success_rate_std']:.2%}; "
                    f"reward={values['reward_mean']:.2f} +/- {values['reward_std']:.2f}; "
                    f"violations={values['violation_rate_mean']:.2%} +/- {values['violation_rate_std']:.2%}"
                )

        for name, values in summary.items():
            csv_rows.append({"difficulty": difficulty, "policy": name, **values})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    if csv_rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"Saved evaluation results to {args.output} and {args.csv}")


if __name__ == "__main__":
    main()