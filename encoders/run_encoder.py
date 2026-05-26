"""Run encoder with global run-state and soft seed-gradient weighting.

Key differences from legacy C9:
- Run state is computed over **all 20 protospacer positions** (1-20), not just seed.
- PAM (21-23) is excluded from the run-state computation.
- Soft seed-gradient weights are applied after run encoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from utils.sequence import encode_c9_matrix, encode_r9_matrix, normalize_sequence


@dataclass
class RegionEncoder:
    """R9-style region encoder limited to positions 1-20."""

    length: int = 20

    def encode_pair(self, on_seq: str, off_seq: str) -> List[List[int]]:
        on_seq = normalize_sequence(on_seq, length=23)
        off_seq = normalize_sequence(off_seq, length=23)
        matrix = encode_r9_matrix(on_seq, off_seq)
        return matrix[: self.length]

    def encode_batch(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        rows = [self.encode_pair(on_seq, off_seq) for on_seq, off_seq in pairs]
        return np.asarray(rows, dtype=np.float32)


@dataclass
class RunEncoder:
    """Global run-state encoder with configurable seed-gradient weighting.

    Computes run-state across **all** protospacer positions (1-20).
    Supported weight modes:
      - soft: w(pos) = exp(-distance_to_PAM / tau)
      - hard: pos 1-15 weight=1.0, pos 16-20 weight=2.0
      - none: uniform weight=1.0
    """

    length: int = 20
    tau: float = 4.0
    weight_mode: str = "soft"  # "soft" | "hard" | "none"

    def encode_pair(self, on_seq: str, off_seq: str) -> List[List[int]]:
        on_seq = normalize_sequence(on_seq, length=23)
        off_seq = normalize_sequence(off_seq, length=23)
        matrix = encode_c9_matrix(on_seq, off_seq)
        return matrix[: self.length]

    def encode_batch(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        rows = [self.encode_pair(on_seq, off_seq) for on_seq, off_seq in pairs]
        return np.asarray(rows, dtype=np.float32)

    def seed_weights(self) -> np.ndarray:
        """Return (length,) vector of seed-gradient weights."""
        if self.weight_mode == "soft":
            weights = []
            for pos in range(1, self.length + 1):
                distance_to_pam = self.length - pos
                weight = np.exp(-distance_to_pam / self.tau)
                weights.append(float(weight))
            return np.asarray(weights, dtype=np.float32)
        elif self.weight_mode == "hard":
            weights = np.ones(self.length, dtype=np.float32)
            weights[15:] = 2.0  # positions 16-20 (0-indexed 15-19)
            return weights
        else:  # "none"
            return np.ones(self.length, dtype=np.float32)
