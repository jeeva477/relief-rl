"""
Actor-Critic neural network for SafeRoute-RL.

Architecture (Section 12):

    obs -> Linear -> ReLU -> Linear -> ReLU -> [Actor head, Critic head]

The Actor outputs logits over the discrete action space; a
`torch.distributions.Categorical` turns these into a proper policy
distribution pi(a|s) we can sample from and compute log-probabilities
for. The Critic outputs a scalar estimate V(s) of the expected return
from state s, used to compute the advantage:

    A_t = R_t - V(s_t)

Why the advantage instead of the raw return R_t?
--------------------------------------------------
Policy gradient theorem says we can scale the log-probability of an
action by *any* baseline that doesn't depend on the action itself
without introducing bias, and subtracting a good baseline (V(s_t))
reduces the *variance* of the gradient estimate dramatically. Intuitively:
R_t alone tells the policy "this trajectory was worth 40 units", but
that doesn't say whether action a_t was a *good* decision -- maybe every
action from state s_t would have scored similarly. A_t = R_t - V(s_t)
instead asks "was this action better or worse than what we normally
expect from this state?", which is a much cleaner training signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, state_value)."""
        features = self.shared(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from pi(a|s).

        `action_mask` (optional boolean tensor, shape [n_actions]) lets the
        caller zero out the probability of actions that are not currently
        legal *road moves* (e.g. blocked cells). This is a soft/training-time
        convenience -- it is NOT the same thing as the hard safety validator,
        which is a separate, deterministic, non-differentiable check applied
        after the policy has produced a candidate action (see Section 15 and
        backend/app/services/safety_validator.py). Masking here only helps
        the policy learn faster by not wasting probability mass on obviously
        invalid moves.
        """
        logits, value = self.forward(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a batch of (obs, actions) taken during rollout, recompute:
            log_prob(a_t | s_t)  -- for the policy gradient
            entropy(pi(.|s_t))   -- for the entropy bonus
            V(s_t)               -- for the value loss

        The SAME action mask used during rollout is applied so that the
        recomputed log-probabilities are consistent with the behaviour
        policy that actually produced the data (required for a correct
        importance-sampling ratio in PPO).

        Used during the PPO update in rl/algorithms/ppo.py.
        """
        logits, values = self.forward(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values
