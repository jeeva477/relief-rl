"""Thin CLI wrapper so `python scripts/train.py` works from repo root."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.training.train import parse_args, train

if __name__ == "__main__":
    train(parse_args())
