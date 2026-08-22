"""
Quantile network for QR-DQN (Dabney et al., 2017 -- Distributional RL
with Quantile Regression).

Architecture:

    obs -> Linear -> ReLU -> Linear -> ReLU -> Linear -> (n_actions * n_quantiles)
                                                            reshaped to
                                                         (batch, n_actions, n_quantiles)

Instead of a single scalar Q(s,a), the network predicts N quantile
locations theta_1(s,a) ... theta_N(s,a) that approximate the distribution
of the random return Z(s,a). The mean of the quantiles recovers the usual
expected Q-value:

    Q(s,a) = E[Z(s,a)] ~= mean_i theta_i(s,a)

which is what action selection (argmax) uses. The full quantile set is
kept around for research-mode return-distribution plots and for the
quantile-regression TD loss during training.

This mirrors the structure of rl/models/actor_critic.py (same hidden_dim
convention, same masking convention: action_mask is a boolean tensor of
shape [batch, n_actions] where True = legal action).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QuantileNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, n_quantiles: int = 51, hidden_dim: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.quantile_head = nn.Linear(hidden_dim, n_actions * n_quantiles)

        # tau_i = (2i + 1) / (2N), the midpoint quantile fractions used by
        # the standard QR-DQN loss (Dabney et al., 2017, Eq. 3).
        taus = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
        self.register_buffer("taus", taus)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns quantiles with shape (batch, n_actions, n_quantiles)."""
        features = self.shared(obs)
        raw = self.quantile_head(features)
        return raw.view(-1, self.n_actions, self.n_quantiles)

    def q_values(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Expected Q-value per action: mean over the quantile dimension."""
        quantiles = self.forward(obs)
        q = quantiles.mean(dim=-1)
        if action_mask is not None:
            q = q.masked_fill(~action_mask, float("-inf"))
        return q

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        epsilon: float = 0.0,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Epsilon-greedy action selection over masked expected Q-values.
        `obs` is a single-row batch (shape [1, obs_dim]).
        """
        q = self.q_values(obs, action_mask)
        if not deterministic and torch.rand(1).item() < epsilon:
            if action_mask is not None:
                valid = torch.nonzero(action_mask[0], as_tuple=False).squeeze(-1)
                idx = valid[torch.randint(0, valid.numel(), (1,))]
                return idx
            return torch.randint(0, self.n_actions, (1,))
        return torch.argmax(q, dim=-1)
