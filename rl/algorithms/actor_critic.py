"""
Advantage Actor-Critic (A2C) update rule with adaptive entropy
regularization, per Sections 13-14 of the spec.

LOSS FUNCTION
-------------
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    policy_loss = -mean( log_pi(a_t|s_t) * A_t )      # REINFORCE w/ baseline
    value_loss  =  mean( (V(s_t) - R_t)^2 )            # regression to return
    entropy     =  mean( H[pi(.|s_t)] )                # exploration bonus

We use the discounted return as the target for V(s_t):
    R_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}
and the advantage:
    A_t = R_t - V(s_t)

ADAPTIVE EXPLORATION
---------------------
Section 14 asks for a non-fixed entropy coefficient: high early in
training, lower later, but able to spike back up when the agent is
struggling in unfamiliar/high-risk scenarios. We implement this as:

    entropy_coef(t) = max(
        entropy_coef_min,
        entropy_coef_start * decay_rate^t
    ) * risk_multiplier

where `risk_multiplier > 1` when the recent rolling unsafe-action rate
or hazard exposure is elevated relative to its historical average, and
`decay_rate` produces the expected explore-early/exploit-later schedule.
This is intentionally simple and fully documented rather than a black
box -- see `AdaptiveEntropyScheduler` below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.algorithms.ppo import RolloutBuffer
from rl.models.actor_critic import ActorCritic


@dataclass
class A2CConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    value_coef: float = 0.5
    entropy_coef_start: float = 0.05
    entropy_coef_min: float = 0.001
    entropy_decay_rate: float = 0.9995  # applied per-update
    max_grad_norm: float = 0.5


class AdaptiveEntropyScheduler:
    """
    Tracks recent unsafe-action rate and hazard exposure to modulate the
    entropy coefficient beyond simple time-decay. See module docstring.
    """

    def __init__(self, config: A2CConfig, window: int = 50):
        self.config = config
        self.window = window
        self._unsafe_rate_history: list[float] = []
        self._risk_history: list[float] = []
        self._step = 0

    def record_episode(self, unsafe_action_rate: float, mean_risk: float) -> None:
        self._unsafe_rate_history.append(unsafe_action_rate)
        self._risk_history.append(mean_risk)
        if len(self._unsafe_rate_history) > self.window:
            self._unsafe_rate_history.pop(0)
            self._risk_history.pop(0)

    def current_coefficient(self) -> float:
        cfg = self.config
        base = max(cfg.entropy_coef_min, cfg.entropy_coef_start * (cfg.entropy_decay_rate ** self._step))
        self._step += 1

        if len(self._unsafe_rate_history) < 5:
            return base

        recent = np.mean(self._unsafe_rate_history[-5:])
        historical = np.mean(self._unsafe_rate_history)
        risk_multiplier = 1.0
        if historical > 1e-6 and recent > historical * 1.5:
            # Recent behavior is meaningfully riskier than the historical
            # average -> temporarily boost exploration so the agent can
            # find a better strategy rather than collapsing onto an unsafe
            # local optimum.
            risk_multiplier = min(3.0, recent / max(historical, 1e-6))

        return float(base * risk_multiplier)


class A2CTrainer:
    def __init__(self, model: ActorCritic, config: A2CConfig | None = None, device: str = "cpu"):
        self.model = model.to(device)
        self.config = config or A2CConfig()
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.entropy_scheduler = AdaptiveEntropyScheduler(self.config)
        self.buffer = RolloutBuffer()

    def compute_returns(self, rewards: list[float], dones: list[bool]) -> torch.Tensor:
        """Discounted Monte-Carlo return R_t = sum gamma^k * r_{t+k}."""
        returns = []
        running = 0.0
        for r, done in zip(reversed(rewards), reversed(dones)):
            running = r + self.config.gamma * running * (1.0 - float(done))
            returns.insert(0, running)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(
        self,
        obs_batch: torch.Tensor,
        action_batch: torch.Tensor,
        returns: torch.Tensor,
        entropy_coef: float | None = None,
    ) -> dict:
        log_probs, entropy, values = self.model.evaluate_actions(obs_batch, action_batch)

        advantages = returns - values.detach()
        # Normalize advantages for training stability (common practice,
        # reduces sensitivity to reward scale).
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        entropy_mean = entropy.mean()

        coef = entropy_coef if entropy_coef is not None else self.entropy_scheduler.current_coefficient()
        total_loss = policy_loss + self.config.value_coef * value_loss - coef * entropy_mean

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_mean.item()),
            "entropy_coef": float(coef),
            "total_loss": float(total_loss.item()),
        }
