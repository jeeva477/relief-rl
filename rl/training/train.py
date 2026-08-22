"""
Training entry point for Relief-RL's PPO agent (A2C kept for compatibility).

Usage:
    python scripts/train.py --episodes 500 --difficulty MEDIUM --algo ppo

Saves:
    rl/checkpoints/latest_model.pt   (overwritten every checkpoint interval)
    rl/checkpoints/best_model.pt     (only overwritten when eval reward improves)
    rl/checkpoints/training_log.json (per-episode metrics, for plotting)
    rl/checkpoints/metrics.csv       (same metrics as CSV, for research tables)
    rl/checkpoints/run_config.json   (config + seed, for reproducibility)

Every number written here comes from the actual training loop:
episode reward, success, rescues, response time, route efficiency,
resource usage and penalty components are measured from the environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from rl.algorithms.actor_critic import A2CConfig, A2CTrainer
from rl.algorithms.ppo import PPOConfig, PPOTrainer
from rl.envs.evacuation_env import EvacuationEnv, N_ACTIONS, OBS_DIM
from rl.envs.scenarios import DisasterType
from rl.models.actor_critic import ActorCritic
from rl.training.config import TrainingConfig
from rl.training.seeding import get_git_commit, set_seed


def _make_env(config: TrainingConfig) -> EvacuationEnv:
    return EvacuationEnv(
        difficulty=config.difficulty,
        grid_size=config.grid_size,
        max_steps=config.max_steps,
    )


def _episode_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def _choose_disaster(config: TrainingConfig, rng: np.random.Generator) -> str:
    """Scenario distribution: a specific disaster type or all types."""
    if config.disaster == "any":
        return rng.choice([d.value for d in DisasterType])
    return config.disaster


def _record_episode(episode: int, metrics: dict, update_stats: dict | None = None) -> dict:
    record = {
        "episode": episode,
        "reward": round(metrics["total_reward"], 2),
        "penalty": round(metrics.get("total_penalty", 0.0), 2),
        "net_reward": round(metrics["total_reward"] - metrics.get("total_penalty", 0.0), 2),
        "steps": metrics["steps"],
        "success": bool(metrics["success"]),
        "response_time_s": metrics["response_time_s"],
        "rescued": metrics["rescued"],
        "route_efficiency": metrics["route_efficiency"],
        "hard_violations": metrics["hard_violations"],
        "wasted_actions": metrics["wasted_actions"],
        "blocked_attempts": metrics["blocked_attempts"],
        "failed_actions": metrics["wasted_actions"] + metrics["blocked_attempts"],
        "mean_risk": round(metrics["mean_risk"], 4),
        "disaster": metrics.get("disaster", None),
    }
    if update_stats:
        record.update(update_stats)
    return record


def train(config: TrainingConfig) -> str:
    set_seed(config.seed)
    device = config.device
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    env = _make_env(config)
    model = ActorCritic(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_dim=config.hidden_dim)

    if config.algo == "ppo":
        ppo_cfg = PPOConfig(
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_epsilon=config.clip_epsilon,
            entropy_coef=config.entropy_coef,
            value_coef=config.value_coef,
            max_grad_norm=config.max_grad_norm,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            rollout_episodes=config.rollout_episodes,
        )
        trainer: PPOTrainer | A2CTrainer = PPOTrainer(model, ppo_cfg, device=device)
    else:
        a2c_cfg = A2CConfig(
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            entropy_coef_start=config.entropy_coef,
            entropy_coef_min=config.entropy_coef / 10.0,
            value_coef=config.value_coef,
        )
        trainer = A2CTrainer(model, a2c_cfg, device=device)

    training_log = []
    best_reward = float("-inf")
    rng = np.random.default_rng(config.seed)
    window: list[dict] = []
    start_time = time.time()

    for episode in range(1, config.episodes + 1):
        # A fresh randomized scenario every episode (with a controlled seed).
        seed = _episode_seed(rng)
        options = None
        if config.disaster != "any":
            options = {"scenario_config": {"disaster_type": config.disaster,
                                           "difficulty": config.difficulty,
                                           "grid_size": config.grid_size,
                                           "max_steps": config.max_steps}}
        obs, _ = env.reset(seed=seed, options=options)

        episode_metrics = _run_episode(env, model, device, trainer)
        episode_metrics["disaster"] = env.scenario.disaster_type.value if env.scenario else None

        update_stats = None
        if isinstance(trainer, PPOTrainer):
            if episode % config.rollout_episodes == 0 or episode == config.episodes:
                advantages, returns = trainer.compute_gae()
                update_stats = trainer.update(advantages, returns)
                trainer.buffer.clear()
        else:
            obs_batch = torch.as_tensor(np.array(trainer.buffer.obs), dtype=torch.float32, device=device)
            action_batch = torch.as_tensor(trainer.buffer.actions, dtype=torch.long, device=device)
            returns = trainer.compute_returns(trainer.buffer.rewards, trainer.buffer.dones)
            update_stats = trainer.update(obs_batch, action_batch, returns)
            trainer.buffer.clear()

        record = _record_episode(episode, episode_metrics, update_stats)
        training_log.append(record)
        window.append(record)
        if len(window) > config.log_every:
            window.pop(0)

        if episode % config.log_every == 0:
            mean_reward = np.mean([r["reward"] for r in window])
            success_rate = np.mean([r["success"] for r in window])
            print(
                f"[episode {episode}/{config.episodes}] mean_reward={mean_reward:.2f} "
                f"success_rate={success_rate:.2f} rescues={np.sum([r['rescued'] for r in window])} "
                f"algo={config.algo}"
            )

        best_reward = _maybe_checkpoint(model, config, episode, window, best_reward)

    elapsed = time.time() - start_time

    with open(os.path.join(config.checkpoint_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    _write_csv(training_log, os.path.join(config.checkpoint_dir, "metrics.csv"))

    # Post-training evaluation on UNSEEN seeds (never used during training).
    unseen = None
    if config.eval_after_episodes > 0:
        unseen = _quick_unseen_eval(model, config)

    run_metadata = {
        "config": asdict(config),
        "git_commit": get_git_commit(),
        "elapsed_seconds": round(elapsed, 1),
        "episodes": len(training_log),
        "final_mean_reward": round(float(np.mean([r["reward"] for r in training_log[-config.log_every:]])), 2),
        "final_success_rate": round(float(np.mean([r["success"] for r in training_log[-config.log_every:]])), 3),
        "final_best_reward": round(best_reward, 2),
        "unseen_evaluation": unseen,
    }
    with open(os.path.join(config.checkpoint_dir, "run_config.json"), "w") as f:
        json.dump(run_metadata, f, indent=2)

    print(f"Training complete in {elapsed:.1f}s. Final mean reward: {run_metadata['final_mean_reward']:.2f}")
    return os.path.join(config.checkpoint_dir, "best_model.pt")


def _run_episode(env, model, device, trainer) -> dict:
    """Run one episode storing transitions in the trainer's buffer."""
    episode_metrics = {
        "success": False,
        "steps": 0,
        "total_reward": 0.0,
        "hard_violations": 0,
        "wasted_actions": 0,
        "blocked_attempts": 0,
        "mean_risk": 0.0,
        "response_time_s": None,
        "rescued": 0,
        "route_efficiency": None,
    }
    obs = env._build_observation()
    for _ in range(env.max_steps):
        mask_used = env.valid_action_mask()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.as_tensor(mask_used, device=device).unsqueeze(0)
        action, log_prob, value = model.get_action(obs_t, action_mask=mask)
        action_int = int(action.item())

        next_obs, reward, terminated, truncated, step_info = env.step(action_int)

        if isinstance(trainer, PPOTrainer):
            next_value = 0.0 if terminated else float(
                model.forward(torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0))[1].item()
            )
            trainer.buffer.add(
                obs=obs, action=action_int, log_prob=float(log_prob.item()), reward=reward,
                value=float(value.item()), next_value=next_value,
                done=terminated or truncated, mask=mask_used,
            )
        else:
            trainer.buffer.add(
                obs=obs, action=action_int, log_prob=float(log_prob.item()), reward=reward,
                value=float(value.item()), next_value=0.0,
                done=terminated or truncated, mask=mask_used,
            )

        episode_metrics["steps"] += 1
        episode_metrics["total_reward"] += reward
        obs = next_obs

        if terminated or truncated:
            episode_metrics["success"] = bool(step_info.get("success", False))
            ep = step_info.get("episode_metrics", {})
            episode_metrics["response_time_s"] = ep.get("response_time_s")
            episode_metrics["rescued"] = ep.get("rescued", 0)
            episode_metrics["route_efficiency"] = ep.get("route_efficiency")
            break

    episode_metrics["hard_violations"] = env.hard_violations
    episode_metrics["wasted_actions"] = env.wasted_actions
    episode_metrics["blocked_attempts"] = env.blocked_attempts
    episode_metrics["mean_risk"] = env.cumulative_risk / max(env.steps_taken, 1)
    return episode_metrics


def _maybe_checkpoint(model, config: TrainingConfig, episode: int, window: list[dict], best_reward: float) -> float:
    if episode % config.checkpoint_frequency != 0 and episode != config.episodes:
        return best_reward
    latest_path = os.path.join(config.checkpoint_dir, "latest_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "hidden_dim": config.hidden_dim,
        "model_name": "ReliefRL-PPO",
        "model_version": "2.0",
        "algo": config.algo,
        "episode": episode,
    }, latest_path)

    recent_reward = float(np.mean([r["reward"] for r in window]))
    if recent_reward > best_reward:
        best_reward = recent_reward
        best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "obs_dim": OBS_DIM,
            "n_actions": N_ACTIONS,
            "hidden_dim": config.hidden_dim,
            "model_name": "ReliefRL-PPO",
            "model_version": "2.0",
            "algo": config.algo,
            "episode": episode,
            "mean_reward": round(recent_reward, 3),
        }, best_path)
    return best_reward


def _write_csv(training_log: list[dict], path: str) -> None:
    if not training_log:
        return
    columns = [
        "episode", "reward", "penalty", "net_reward", "steps", "success", "response_time_s", "rescued",
        "route_efficiency", "hard_violations", "wasted_actions", "blocked_attempts",
        "failed_actions", "mean_risk", "disaster",
        "policy_loss", "value_loss", "entropy", "entropy_coef", "total_loss",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in training_log:
            writer.writerow(record)


def _quick_unseen_eval(model: ActorCritic, config: TrainingConfig) -> dict:
    """
    Evaluate the trained policy on N episodes using seeds from a hold-out
    range (config.eval_seed_offset) that was never used in training.
    """
    from rl.training.evaluate import _run_policy_episode, make_rl_policy

    env = _make_env(config)
    policy = make_rl_policy(model)
    rng = np.random.default_rng(config.eval_seed_offset)
    results = [
        _run_policy_episode(env, policy, seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(config.eval_after_episodes)
    ]
    from rl.evaluation.metrics import aggregate

    agg = aggregate(results)
    return {
        "episodes": len(results),
        "seed_offset": config.eval_seed_offset,
        "success_rate": round(agg.success_rate, 3),
        "mean_reward": round(agg.mean_reward, 2),
        "std_reward": round(agg.std_reward, 2),
        "mean_response_time_s": round(agg.mean_response_time_s or 0.0, 1),
        "mean_rescues": round(agg.mean_rescues or 0.0, 2),
        "mean_route_efficiency": round(agg.route_efficiency or 0.0, 3),
        "mean_resource_usage": round(agg.mean_resource_usage or 0.0, 2),
        "violation_rate": round(agg.violation_rate, 3),
    }


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train Relief-RL PPO agent")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c"])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rollout-episodes", type=int, default=8)
    parser.add_argument("--difficulty", type=str, default="MEDIUM", choices=["EASY", "MEDIUM", "HARD", "EXTREME"])
    parser.add_argument("--disaster", type=str, default="any",
                        help="any, flood, wildfire, earthquake, cyclone, tsunami, landslide, "
                             "heavy_rain, road_blockage, traffic_jam, combined")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-frequency", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="rl/checkpoints")
    parser.add_argument("--eval-after-episodes", type=int, default=0,
                        help="run this many evaluation episodes on UNSEEN seeds after training")
    args = parser.parse_args()
    return TrainingConfig(
        episodes=args.episodes,
        algo=args.algo,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        rollout_episodes=args.rollout_episodes,
        difficulty=args.difficulty,
        disaster=args.disaster,
        max_steps=args.max_steps,
        grid_size=args.grid_size,
        seed=args.seed,
        checkpoint_frequency=args.checkpoint_frequency,
        checkpoint_dir=args.checkpoint_dir,
        eval_after_episodes=args.eval_after_episodes,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)