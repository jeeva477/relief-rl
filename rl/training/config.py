from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    episodes: int = 500
    algo: str = "ppo"               # "ppo" (default) or "a2c"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 5
    batch_size: int = 128
    rollout_episodes: int = 8       # episodes collected between PPO updates
    max_steps: int = 100
    grid_size: int = 10
    difficulty: str = "MEDIUM"
    disaster: str = "any"           # "any" (all types) or a specific DisasterType value
    seed: int = 42
    checkpoint_frequency: int = 50
    checkpoint_dir: str = "rl/checkpoints"
    hidden_dim: int = 128
    device: str = "cpu"
    log_every: int = 10
    eval_after_episodes: int = 0    # quick unseen-seed eval after training
    eval_seed_offset: int = 100_000  # hold-out seed range (never used in training)