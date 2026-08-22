"""
Train the QR-DQN agent (alternative RL agent for research comparison
against PPO).

Usage:
    python scripts/train_qrdqn.py --episodes 500 --difficulty MEDIUM --seed 42 \
        --checkpoint-dir rl/checkpoints --eval-after-episodes 50
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.training.train_qrdqn import parse_args, train

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
