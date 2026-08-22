"""Thin CLI wrapper so `python scripts/evaluate.py` works from repo root."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.training.evaluate import main

if __name__ == "__main__":
    main()
