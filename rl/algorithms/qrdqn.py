"""
Quantile Regression DQN (QR-DQN) -- the alternative, distributional
discrete-action RL agent used for research comparison against PPO.

QR-DQN (Dabney, Rowland, Bellemare & Munos, 2017) replaces the single
scalar Q(s,a) learned by vanilla DQN with a set of N quantile locations
that approximate the full distribution of the return Z(s,a). This lets
the agent reason about *risk* (variance / spread of outcomes), not just
the expected value -- relevant for a disaster-response environment where
two actions can have the same expected reward but very different downside
risk (e.g. a risky shortcut vs a safe detour).

PENALTY LEARNING PIPELINE (QR-DQN side, mirrors the PPO docstring in
rl/algorithms/ppo.py)
-------------------------------------------------------------------
    Penalty (negative reward)
        -> stored in replay buffer alongside (s, a, s', done)
        -> TD target: r + gamma * Z_target(s', a*)   [a* selected Double-DQN
           style by the ONLINE network, evaluated by the TARGET network]
        -> quantile Huber (pinball) regression loss between the online
           network's quantiles for (s, a) and the TD-target quantiles
        -> gradient step lowers Q(s,a) for actions that led to penalties
        -> epsilon-greedy action selection is progressively less likely to
           pick that action as its expected quantile mean falls.

This is a genuine value-based off-policy method (experience replay +
target network), not a rebadged bandit or evaluation-only wrapper, and it
uses the SAME EvacuationEnv observation/action space and the SAME action
mask convention as PPO so that a PPO vs QR-DQN comparison is scientifically
meaningful (see rl/training/evaluate.py and scripts/compare_ppo_qrdqn.py).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.models.qrdqn_net import QuantileNetwork


@dataclass
class QRDQNConfig:
    n_quantiles: int = 51
    learning_rate: float = 5e-4
    gamma: float = 0.99
    buffer_size: int = 50_000
    batch_size: int = 64
    min_replay_size: int = 500       # warm-up transitions before learning starts
    target_update_freq: int = 500     # hard target-network sync, in gradient steps
    max_grad_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000  # env steps, not gradient steps
    double_dqn: bool = True
    train_freq: int = 4               # env steps between gradient updates
    hidden_dim: int = 128


class ReplayBuffer:
    """Fixed-size FIFO experience replay buffer."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data: deque = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done, mask, next_mask) -> None:
        self._data.append((obs, action, reward, next_obs, done, mask, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self._data, batch_size)
        obs, actions, rewards, next_obs, dones, masks, next_masks = zip(*batch)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(masks, dtype=bool),
            np.asarray(next_masks, dtype=bool),
        )

    def __len__(self) -> int:
        return len(self._data)


def quantile_huber_loss(
    predicted: torch.Tensor,   # (batch, N) quantiles for the taken action
    target: torch.Tensor,      # (batch, N) TD-target quantiles
    taus: torch.Tensor,        # (N,) quantile fractions
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    Standard QR-DQN pinball/Huber quantile regression loss
    (Dabney et al., 2017, Eq. 3), pairwise over all (predicted, target)
    quantile combinations.
    """
    # td_error[b, i, j] = target_j - predicted_i
    td_error = target.unsqueeze(1) - predicted.unsqueeze(2)  # (batch, N_pred, N_target)

    huber = torch.where(
        td_error.abs() <= kappa,
        0.5 * td_error.pow(2),
        kappa * (td_error.abs() - 0.5 * kappa),
    )
    tau = taus.view(1, -1, 1)  # (1, N_pred, 1)
    weight = (tau - (td_error.detach() < 0).float()).abs()
    loss = (weight * huber).mean(dim=2).sum(dim=1)  # sum over predicted quantiles
    return loss.mean()


class QRDQNTrainer:
    def __init__(self, obs_dim: int, n_actions: int, config: QRDQNConfig | None = None, device: str = "cpu"):
        self.config = config or QRDQNConfig()
        self.device = device
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.online = QuantileNetwork(
            obs_dim, n_actions, n_quantiles=self.config.n_quantiles, hidden_dim=self.config.hidden_dim
        ).to(device)
        self.target = QuantileNetwork(
            obs_dim, n_actions, n_quantiles=self.config.n_quantiles, hidden_dim=self.config.hidden_dim
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.buffer = ReplayBuffer(self.config.buffer_size)

        self._env_steps = 0
        self._grad_steps = 0

    # ------------------------------------------------------------------
    # Exploration schedule
    # ------------------------------------------------------------------
    def epsilon(self) -> float:
        cfg = self.config
        frac = min(1.0, self._env_steps / max(1, cfg.epsilon_decay_steps))
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = False) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        eps = 0.0 if deterministic else self.epsilon()
        action = self.online.act(obs_t, action_mask=mask_t, epsilon=eps, deterministic=deterministic)
        self._env_steps += 1
        return int(action.item())

    # ------------------------------------------------------------------
    # Replay + learning
    # ------------------------------------------------------------------
    def store(self, obs, action, reward, next_obs, done, mask, next_mask) -> None:
        self.buffer.add(obs, action, reward, next_obs, done, mask, next_mask)

    def maybe_train_step(self) -> dict | None:
        """Runs one gradient step if enough data is buffered and it's time
        to train (per train_freq); returns None otherwise so the caller
        can distinguish "no update happened" from a real stats dict."""
        cfg = self.config
        if len(self.buffer) < max(cfg.min_replay_size, cfg.batch_size):
            return None
        if self._env_steps % cfg.train_freq != 0:
            return None
        return self.train_step()

    def train_step(self) -> dict:
        cfg = self.config
        obs, actions, rewards, next_obs, dones, masks, next_masks = self.buffer.sample(cfg.batch_size)

        obs_t = torch.as_tensor(obs, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)
        next_masks_t = torch.as_tensor(next_masks, device=self.device)

        # Predicted quantiles for the taken action.
        quantiles = self.online(obs_t)  # (batch, n_actions, N)
        idx = actions_t.view(-1, 1, 1).expand(-1, 1, cfg.n_quantiles)
        predicted = quantiles.gather(1, idx).squeeze(1)  # (batch, N)

        with torch.no_grad():
            if cfg.double_dqn:
                # Double-DQN: ONLINE network selects the next action,
                # TARGET network evaluates it -- reduces Q-value overestimation.
                next_q_online = self.online.q_values(next_obs_t, action_mask=next_masks_t)
                next_actions = torch.argmax(next_q_online, dim=-1)
            else:
                next_q_target = self.target.q_values(next_obs_t, action_mask=next_masks_t)
                next_actions = torch.argmax(next_q_target, dim=-1)

            next_quantiles = self.target(next_obs_t)  # (batch, n_actions, N)
            next_idx = next_actions.view(-1, 1, 1).expand(-1, 1, cfg.n_quantiles)
            next_quantiles = next_quantiles.gather(1, next_idx).squeeze(1)  # (batch, N)

            target = rewards_t.unsqueeze(1) + cfg.gamma * (1.0 - dones_t.unsqueeze(1)) * next_quantiles

        loss = quantile_huber_loss(predicted, target, self.online.taus)

        if not torch.isfinite(loss):
            print("[qrdqn] warning: skipping non-finite loss batch")
            return {"loss": float("nan"), "grad_norm": 0.0, "epsilon": self.epsilon(), "skipped": True}

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        self._grad_steps += 1
        if self._grad_steps % cfg.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()),
            "epsilon": self.epsilon(),
            "skipped": False,
        }

    # ------------------------------------------------------------------
    # Inspection helpers (research mode / explainability)
    # ------------------------------------------------------------------
    def q_values(self, obs: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0) if mask is not None else None
        with torch.no_grad():
            q = self.online.q_values(obs_t, action_mask=mask_t)
        return q.squeeze(0).cpu().numpy()

    def return_distribution(self, obs: np.ndarray, action: int) -> np.ndarray:
        """Full quantile set for a given (obs, action) -- used by
        research-mode 'quantile return distribution' panel."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            quantiles = self.online(obs_t)
        return quantiles[0, action].cpu().numpy()
