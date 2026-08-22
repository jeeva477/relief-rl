"""
Road graph abstraction.

The training environment uses a discrete grid (Section 8 of the spec
allows this for the initial simulation), but the neural network and
reward code never assume "grid" directly -- they only ever see:

    - a current node id
    - a set of candidate neighbor node ids ("actions")
    - normalized coordinates for each node

This lets the exact same Actor-Critic model later be pointed at a real
road graph built from Google Route Matrix results (see
backend/app/services/state_builder.py) without retraining the network
architecture, only what feeds it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    id: str
    x: float  # normalized [0, 1]
    y: float  # normalized [0, 1]
    blocked: bool = False


class GridRoadGraph:
    """
    A simple 4-connected grid road graph used for training.

    Grid cells are addressed (row, col) and normalized to [0, 1] x [0, 1]
    for the observation space. This is the "simulation" side of the
    simulation/real-world distinction described in the spec.
    """

    def __init__(self, size: int = 10):
        self.size = size
        self._blocked: set[tuple[int, int]] = set()

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def is_blocked(self, row: int, col: int) -> bool:
        return (row, col) in self._blocked

    def set_blocked(self, cells: set[tuple[int, int]]) -> None:
        self._blocked = set(cells)

    def normalize(self, row: int, col: int) -> tuple[float, float]:
        if self.size <= 1:
            return 0.0, 0.0
        return row / (self.size - 1), col / (self.size - 1)

    def neighbors(self, row: int, col: int) -> dict[int, tuple[int, int]]:
        """
        Map action id -> neighbor cell, for the 5 discrete actions
        (0=stay, 1=north, 2=east, 3=south, 4=west). Only in-bounds,
        non-blocked neighbors are included; the caller treats a missing
        action id as "not currently valid" (a form of road unavailability).
        """
        candidates = {
            0: (row, col),
            1: (row - 1, col),
            2: (row, col + 1),
            3: (row + 1, col),
            4: (row, col - 1),
        }
        valid = {}
        for action_id, (r, c) in candidates.items():
            if action_id == 0:
                valid[action_id] = (r, c)
                continue
            if self.in_bounds(r, c) and not self.is_blocked(r, c):
                valid[action_id] = (r, c)
        return valid
