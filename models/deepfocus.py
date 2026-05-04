from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from encoders.r9_encoder import R9Encoder
from models.base import BaseSequenceModel
from utils.sequence import mismatch_mask_from_matrix, run_length_stats


def _round6(value: float) -> float:
    return round(float(value), 6)


def _r9_feature_vector(matrix: Sequence[Sequence[int]]) -> List[float]:
    mask = mismatch_mask_from_matrix(matrix)
    run_count, longest_run, adjacent_pairs, runs = run_length_stats(mask)
    mismatch_count = sum(mask)
    ordinary = seed = pam = 0
    for row, is_mismatch in zip(matrix, mask):
        if not is_mismatch:
            continue
        if row[7:9] == [0, 1]:
            ordinary += 1
        elif row[7:9] == [1, 0]:
            seed += 1
        elif row[7:9] == [0, 0]:
            pam += 1

    base_active = 0
    gc_active = 0
    gap_count = 0
    for row in matrix:
        base_active += sum(row[:4])
        gc_active += row[2] + row[3]
        gap_count += row[4]

    total_positions = max(1, len(matrix))
    active_bases = max(1, base_active)
    weighted_mismatch = ordinary * 1.0 + seed * 1.6 + pam * 2.1
    tail_mismatch = sum(mask[15:])
    transition_density = adjacent_pairs / max(1, len(matrix) - 1)
    gc_ratio = gc_active / active_bases
    match_ratio = 1.0 - (mismatch_count / total_positions)
    mismatch_density = mismatch_count / total_positions
    return [
        _round6(mismatch_density),
        _round6(weighted_mismatch / total_positions),
        _round6(ordinary / total_positions),
        _round6(seed / total_positions),
        _round6(pam / total_positions),
        _round6(longest_run / total_positions),
        _round6(run_count / total_positions),
        _round6(transition_density),
        _round6(tail_mismatch / max(1, total_positions - 15)),
        _round6(gc_ratio),
        _round6(match_ratio),
        _round6(gap_count / total_positions),
    ]


@dataclass
class DeepFocusModel(BaseSequenceModel):
    model_name: str = "deepfocus"
    encoder_name: str = "r9"
    feature_names: List[str] = None  # type: ignore[assignment]

    def __init__(self, state=None):
        if self.feature_names is None:
            self.feature_names = [
                "mismatch_density",
                "weighted_mismatch_density",
                "ordinary_mismatch_density",
                "seed_mismatch_density",
                "pam_mismatch_density",
                "longest_run_density",
                "run_count_density",
                "transition_density",
                "tail_mismatch_density",
                "gc_ratio",
                "match_ratio",
                "gap_density",
            ]
        super().__init__(state=state)
        if all(weight == 0.0 for weight in self.state.weights):
            self.state.bias = -3.0
            self.state.weights = [
                1.8,
                2.8,
                0.3,
                1.2,
                1.7,
                1.0,
                0.4,
                0.8,
                1.5,
                -0.3,
                -1.4,
                0.2,
            ]

    def feature_vector(self, on_seq: str, off_seq: str) -> List[float]:
        matrix = R9Encoder().encode_pair(on_seq, off_seq)
        return _r9_feature_vector(matrix)

    def summarize_matrix(self, matrix: Sequence[Sequence[int]]) -> List[float]:
        return _r9_feature_vector(matrix)
