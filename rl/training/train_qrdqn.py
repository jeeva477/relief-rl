"""
Training entry point for Relief-RL's QR-DQN agent -- the alternative,
distributional discrete-action agent used for research comparison
against PPO (see rl/algorithms/qrdqn.py for the algorithm itself).

Usage:
    python scripts/train_qrdqn.py --episodes 500 --difficulty MEDIUM

Saves (mirrors rl/training/train.py's PPO output format so the two
agents can be compared apples-to-apples):
    rl/checkpoints/qrdqn_latest_model.pt
    rl/checkpoints/qrdqn_best_model.pt     (best = highest window mean reward)
    rl/checkpoints/qrdqn_training_log.json
    rl/checkpoints/qrdqn_metrics.csv
    rl/checkpoints/qrdqn_run_config.json

Every number written here comes from the actual training loop: episode
reward, success, rescues, response time, route efficiency, resource
usage and TD/quantile loss are measured from the environment and the
optimizer, never fabricated.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from rl.algorithms.qrdqn import QRDQNConfig, QRDQNTrainer
from rl.envs.evacuation_env import EvacuationEnv, N_ACTIONS, OBS_DIM
from rl.envs.scenarios import DisasterType
from rl.training.seeding import get_git_commit, set_seed


@dataclass
class QRDQNTrainingConfig:
    episodes: int = 500
    learning_rate: float = 5e-4
    gamma: float = 0.99
    n_quantiles: int = 51
    buffer_size: int = 50_000
    batch_size: int = 64
    min_replay_size: int = 500
    target_update_freq: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    double_dqn: bool = True
    train_freq: int = 4
    max_steps: int = 100
    grid_size: int = 10
    difficulty: str = "MEDIUM"
    disaster: str = "any"
    seed: int = 42
    checkpoint_frequency: int = 50
    checkpoint_dir: str = "rl/checkpoints"
    hidden_dim: int = 128
    device: str = "cpu"
    log_every: int = 10
    eval_after_episodes: int = 0
    eval_seed_offset: int = 100_000


def _make_env(config: QRDQNTrainingConfig) -> EvacuationEnv:
    return EvacuationEnv(difficulty=config.difficulty, grid_size=config.grid_size, max_steps=config.max_steps)


def _episode_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def _choose_disaster(config: QRDQNTrainingConfig, rng: np.random.Generator) -> str:
    if config.disaster == "any":
        return rng.choice([d.value for d in DisasterType])
    return config.disaster


def _run_episode(env: EvacuationEnv, trainer: QRDQNTrainer) -> tuple[dict, list[dict]]:
    """Runs one episode, storing transitions in replay and taking gradient
    steps per train_freq. Returns episode metrics + per-step loss stats."""
    episode_metrics = {
        "success": False, "steps": 0, "total_reward": 0.0,
        "hard_violations": 0, "wasted_actions": 0, "blocked_attempts": 0,
        "mean_risk": 0.0, "response_time_s": None, "rescued": 0, "route_efficiency": None,
    }
    step_stats: list[dict] = []
    obs = env._build_observation()
    for _ in range(env.max_steps):
        mask = env.valid_action_mask()
        action = trainer.select_action(obs, mask, deterministic=False)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        next_mask = env.valid_action_mask()

        trainer.store(obs=obs, action=action, reward=reward, next_obs=next_obs,
                      done=terminated, mask=mask, next_mask=next_mask)
        stats = trainer.maybe_train_step()
        if stats is not None:
            step_stats.append(stats)

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
    return episode_metrics, step_stats


def _record_episode(episode: int, metrics: dict, step_stats: list[dict]) -> dict:
    record = {
        "episode": episode,
        "reward": round(metrics["total_reward"], 2),
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
        "disaster": metrics.get("disaster"),
    }
    finite_losses = [s["loss"] for s in step_stats if not s.get("skipped") and np.isfinite(s.get("loss", np.nan))]
    record["mean_loss"] = round(float(np.mean(finite_losses)), 4) if finite_losses else None
    record["epsilon"] = round(step_stats[-1]["epsilon"], 4) if step_stats else None
    record["gradient_steps"] = len(finite_losses)
    return record


def train(config: QRDQNTrainingConfig) -> str:
    set_seed(config.seed)
    device = config.device
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    env = _make_env(config)
    qrdqn_cfg = QRDQNConfig(
        n_quantiles=config.n_quantiles, learning_rate=config.learning_rate, gamma=config.gamma,
        buffer_size=config.buffer_size, batch_size=config.batch_size, min_replay_size=config.min_replay_size,
        target_update_freq=config.target_update_freq, epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end, epsilon_decay_steps=config.epsilon_decay_steps,
        double_dqn=config.double_dqn, train_freq=config.train_freq, hidden_dim=config.hidden_dim,
    )
    trainer = QRDQNTrainer(obs_dim=OBS_DIM, n_actions=N_ACTIONS, config=qrdqn_cfg, device=device)

    training_log = []
    best_reward = float("-inf")
    rng = np.random.default_rng(config.seed)
    window: list[dict] = []
    start_time = time.time()

    for episode in range(1, config.episodes + 1):
        seed = _episode_seed(rng)
        options = None
        if config.disaster != "any":
            options = {"scenario_config": {"disaster_type": config.disaster, "difficulty": config.difficulty,
                                           "grid_size": config.grid_size, "max_steps": config.max_steps}}
        env.reset(seed=seed, options=options)

        episode_metrics, step_stats = _run_episode(env, trainer)
        episode_metrics["disaster"] = env.scenario.disaster_type.value if env.scenario else None

        record = _record_episode(episode, episode_metrics, step_stats)
        training_log.append(record)
        window.append(record)
        if len(window) > config.log_every:
            window.pop(0)

        if episode % config.log_every == 0:
            mean_reward = np.mean([r["reward"] for r in window])
            success_rate = np.mean([r["success"] for r in window])
            print(
                f"[episode {episode}/{config.episodes}] mean_reward={mean_reward:.2f} "
                f"success_rate={success_rate:.2f} epsilon={trainer.epsilon():.3f} "
                f"buffer={len(trainer.buffer)} algo=qrdqn"
            )

        best_reward = _maybe_checkpoint(trainer, config, episode, window, best_reward)

    elapsed = time.time() - start_time

    with open(os.path.join(config.checkpoint_dir, "qrdqn_training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)
    _write_csv(training_log, os.path.join(config.checkpoint_dir, "qrdqn_metrics.csv"))

    unseen = None
    if config.eval_after_episodes > 0:
        unseen = _quick_unseen_eval(trainer, config)

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
    with open(os.path.join(config.checkpoint_dir, "qrdqn_run_config.json"), "w") as f:
        json.dump(run_metadata, f, indent=2)

    print(f"QR-DQN training complete in {elapsed:.1f}s. Final mean reward: {run_metadata['final_mean_reward']:.2f}")
    return os.path.join(config.checkpoint_dir, "qrdqn_best_model.pt")


def _checkpoint_dict(trainer: QRDQNTrainer, config: QRDQNTrainingConfig, episode: int) -> dict:
    return {
        "model_state_dict": trainer.online.state_dict(),
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "n_quantiles": config.n_quantiles,
        "hidden_dim": config.hidden_dim,
        "model_name": "ReliefRL-QRDQN",
        "model_version": "1.0",
        "algo": "qrdqn",
        "episode": episode,
    }


def _maybe_checkpoint(trainer: QRDQNTrainer, config: QRDQNTrainingConfig, episode: int,
                       window: list[dict], best_reward: float) -> float:
    if episode % config.checkpoint_frequency != 0 and episode != config.episodes:
        return best_reward
    latest_path = os.path.join(config.checkpoint_dir, "qrdqn_latest_model.pt")
    torch.save(_checkpoint_dict(trainer, config, episode), latest_path)

    recent_reward = float(np.mean([r["reward"] for r in window]))
    if recent_reward > best_reward:
        best_reward = recent_reward
        best_path = os.path.join(config.checkpoint_dir, "qrdqn_best_model.pt")
        ckpt = _checkpoint_dict(trainer, config, episode)
        ckpt["mean_reward"] = round(recent_reward, 3)
        torch.save(ckpt, best_path)
    return best_reward


def _write_csv(training_log: list[dict], path: str) -> None:
    if not training_log:
        return
    columns = [
        "episode", "reward", "steps", "success", "response_time_s", "rescued",
        "route_efficiency", "hard_violations", "wasted_actions", "blocked_attempts",
        "failed_actions", "mean_risk", "disaster", "mean_loss", "epsilon", "gradient_steps",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in training_log:
            writer.writerow(record)


def _quick_unseen_eval(trainer: QRDQNTrainer, config: QRDQNTrainingConfig) -> dict:
    """Evaluate on UNSEEN seeds (never used in training), deterministic
    (greedy, no epsilon exploration) policy -- mirrors PPO's
    _quick_unseen_eval in rl/training/train.py."""
    from rl.training.evaluate import _run_policy_episode, make_qrdqn_policy
    from rl.evaluation.metrics import aggregate

    env = _make_env(config)
    policy = make_qrdqn_policy(trainer.online)
    rng = np.random.default_rng(config.eval_seed_offset)
    results = [
        _run_policy_episode(env, policy, seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(config.eval_after_episodes)
    ]
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


def parse_args() -> QRDQNTrainingConfig:
    parser = argparse.ArgumentParser(description="Train Relief-RL QR-DQN agent")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-quantiles", type=int, default=51)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-replay-size", type=int, default=500)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--no-double-dqn", action="store_true")
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--difficulty", type=str, default="MEDIUM", choices=["EASY", "MEDIUM", "HARD", "EXTREME"])
    parser.add_argument("--disaster", type=str, default="any")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-frequency", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="rl/checkpoints")
    parser.add_argument("--eval-after-episodes", type=int, default=0)
    args = parser.parse_args()
    return QRDQNTrainingConfig(
        episodes=args.episodes, learning_rate=args.learning_rate, gamma=args.gamma,
        n_quantiles=args.n_quantiles, buffer_size=args.buffer_size, batch_size=args.batch_size,
        min_replay_size=args.min_replay_size, target_update_freq=args.target_update_freq,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps, double_dqn=not args.no_double_dqn,
        train_freq=args.train_freq, difficulty=args.difficulty, disaster=args.disaster,
        max_steps=args.max_steps, grid_size=args.grid_size, seed=args.seed,
        checkpoint_frequency=args.checkpoint_frequency, checkpoint_dir=args.checkpoint_dir,
        eval_after_episodes=args.eval_after_episodes,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
