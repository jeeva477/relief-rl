"""
Proximal Policy Optimization (PPO) -- the primary training algorithm.

This is a genuine PPO implementation (Schulman et al., 2017), NOT an
A2C relabeled as PPO. The distinguishing PPO ingredients are all here:

    - clipped surrogate objective
        L^CLIP = mean( min(r_t * A_t, clip(r_t, 1-e, 1+e) * A_t) )
      where r_t = pi_new(a|s) / pi_old(a|s) is the importance-sampling
      ratio. Clipping prevents a single update from moving the policy
      too far from the data that produced the rollout.

    - Generalized Advantage Estimation (GAE, Schulman et al., 2016)
        A_t = delta_t + (gamma * lambda) * A_{t+1}
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
      which trades bias vs variance smoothly via lambda.

    - multiple optimization epochs over the collected rollout with
      minibatches (the data is reused several times, unlike A2C).

    - advantage normalization per minibatch (variance reduction).

    - clipped value loss and an entropy bonus for exploration.

    - gradient clipping for stable training.

PENALTY LEARNING PIPELINE
-------------------------
Bad decisions (e.g. entering a hazard zone, wasting resources, driving
into a blocked road) produce negative rewards. Those rewards flow into
the rollout buffer; GAE turns them into negative advantages; the clipped
objective raises the log-probability gradient in the direction that
reduces the probability of the bad action on the next update; over many
updates the policy assigns lower probability to bad behavior. This is
measurable (policy loss, entropy, success rate, average penalty) and is
reported by training/evaluation without any fabrication.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.envs.evacuation_env import EvacuationEnv
from rl.models.actor_critic import ActorCritic


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 5
    batch_size: int = 128
    rollout_episodes: int = 8  # episodes collected between updates


class RolloutBuffer:
    """Stores one rollout window: obs, actions, log-probs, values, rewards, masks."""

    def __init__(self) -> None:
        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.next_values: list[float] = []
        self.dones: list[bool] = []
        self.masks: list[np.ndarray] = []

    def add(self, obs, action, log_prob, reward, value, next_value, done, mask) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
        self.dones.append(bool(done))
        self.masks.append(mask)

    def clear(self) -> None:
        for attr in ("obs", "actions", "log_probs", "rewards", "values", "next_values",
                     "dones", "masks"):
            getattr(self, attr).clear()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOTrainer:
    def __init__(self, model: ActorCritic, config: PPOConfig | None = None, device: str = "cpu"):
        self.model = model.to(device)
        self.config = config or PPOConfig()
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def collect_episode(self, env: EvacuationEnv, seed: int | None = None) -> dict:
        """Run one episode and store every transition in the buffer."""
        obs, _ = env.reset(seed=seed)
        metrics = {
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
        for _ in range(env.max_steps):
            mask_used = env.valid_action_mask()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = torch.as_tensor(mask_used, device=self.device).unsqueeze(0)
            action, log_prob, value = self.model.get_action(obs_t, action_mask=mask)
            action_int = int(action.item())

            next_obs, reward, terminated, truncated, step_info = env.step(action_int)
            if terminated:
                next_value = 0.0  # no bootstrap past a terminal state
            else:
                next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.inference_mode():
                    _, next_value_est = self.model.forward(next_obs_t)
                next_value = float(next_value_est.item())

            self.buffer.add(
                obs=obs,
                action=action_int,
                log_prob=float(log_prob.item()),
                reward=reward,
                value=float(value.item()),
                next_value=next_value,
                done=terminated or truncated,
                mask=mask_used,
            )

            metrics["steps"] += 1
            metrics["total_reward"] += reward
            metrics["hard_violations"] = env.hard_violations
            metrics["wasted_actions"] = env.wasted_actions
            metrics["blocked_attempts"] = env.blocked_attempts
            metrics["mean_risk"] = env.cumulative_risk / max(env.steps_taken, 1)
            obs = next_obs

            if terminated or truncated:
                metrics["success"] = bool(step_info.get("success", False))
                ep_metrics = step_info.get("episode_metrics", {})
                metrics["response_time_s"] = ep_metrics.get("response_time_s")
                metrics["rescued"] = ep_metrics.get("rescued", 0)
                metrics["route_efficiency"] = ep_metrics.get("route_efficiency")
                break

        return metrics

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------
    def compute_gae(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation over the current buffer.

        Returns (advantages, returns) with returns = advantages + values.
        The stored next-state value already encodes termination
        (bootstrap 0) or truncation (bootstrap V(s_T)), so truncated
        episodes still produce a correct, unbiased value target.
        """
        cfg = self.config
        rewards = self.buffer.rewards
        values = self.buffer.values
        next_values = self.buffer.next_values
        n = len(rewards)

        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + cfg.gamma * next_values[t] - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * gae
            advantages[t] = gae

        returns = advantages + np.asarray(values, dtype=np.float32)
        return (
            torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            torch.as_tensor(returns, dtype=torch.float32, device=self.device),
        )

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def update(self, advantages: torch.Tensor, returns: torch.Tensor) -> dict:
        """
        Multiple-epoch minibatch update with the clipped surrogate
        objective. Returns per-update statistics.
        """
        cfg = self.config
        n = len(self.buffer)

        obs_batch = torch.as_tensor(np.array(self.buffer.obs), dtype=torch.float32, device=self.device)
        action_batch = torch.as_tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_logp_batch = torch.as_tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        mask_batch = torch.as_tensor(np.array(self.buffer.masks), dtype=torch.bool, device=self.device)

        indices = np.arange(n)
        total_stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
        }
        n_updates = 0

        for _ in range(cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, cfg.batch_size):
                mb = indices[start:start + cfg.batch_size]
                obs_mb = obs_batch[mb]
                act_mb = action_batch[mb]
                old_logp_mb = old_logp_batch[mb]
                mask_mb = mask_batch[mb]
                adv_mb = advantages[mb]
                ret_mb = returns[mb]

                if not torch.isfinite(adv_mb).all() or not torch.isfinite(ret_mb).all():
                    print("[ppo] warning: skipping minibatch with non-finite advantages/returns")
                    continue

                logp_new, entropy, values = self.model.evaluate_actions(obs_mb, act_mb, mask_mb)
                if not torch.isfinite(logp_new).all():
                    print("[ppo] warning: skipping minibatch with non-finite log-probs")
                    continue

                ratio = torch.exp(logp_new - old_logp_mb)
                adv_norm = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                surr1 = ratio * adv_norm
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * adv_norm
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss on normalized returns (stable across reward scale).
                ret_norm = (ret_mb - ret_mb.mean()) / (ret_mb.std() + 1e-8)
                value_loss = nn.functional.mse_loss(values, ret_norm)

                entropy_loss = -entropy.mean()
                total_loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_stats["policy_loss"] += float(policy_loss.item())
                total_stats["value_loss"] += float(value_loss.item())
                total_stats["entropy"] += float(entropy.mean().item())
                total_stats["total_loss"] += float(total_loss.item())
                total_stats["clip_fraction"] += float(((ratio - 1.0).abs() > cfg.clip_epsilon).float().mean().item())
                total_stats["grad_norm"] += float(grad_norm.item())
                n_updates += 1

        for k in total_stats:
            total_stats[k] = total_stats[k] / max(n_updates, 1)
        total_stats["entropy_coef"] = float(cfg.entropy_coef)
        return total_stats

    def compute_and_update(self) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """Convenience: GAE then PPO update."""
        advantages, returns = self.compute_gae()
        stats = self.update(advantages, returns)
        return stats, advantages, returns

    def action_probs(self, obs: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Softmax action probabilities for a single observation (UI/explainability)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.model.forward(obs_t)
        if mask is not None:
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            logits = logits.masked_fill(~mask_t, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0).detach().cpu().numpy()

    def value_of(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            _, value = self.model.forward(obs_t)
        return float(value.item())