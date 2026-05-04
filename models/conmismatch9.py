from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from encoders.c9_encoder import C9Encoder
from models.base import BaseSequenceModel
from utils.sequence import c9_mask_from_matrix, run_length_stats


def _round6(value: float) -> float:
    return round(float(value), 6)


def _nested_slice(x: Sequence[Sequence[Sequence[int]]], start: int, stop: int) -> List[List[List[int]]]:
    if x and x[0] and isinstance(x[0][0], int):  # type: ignore[index]
        return [[list(row[start:stop]) for row in x]]  # type: ignore[union-attr]
    return [
        [list(row[start:stop]) for row in sample]
        for sample in x
    ]


def _summarize_c9_matrix(matrix: Sequence[Sequence[int]]) -> List[float]:
    mismatch_mask = c9_mask_from_matrix(matrix)
    run_count, longest_run, adjacent_pairs, _ = run_length_stats(mismatch_mask)
    mismatch_count = sum(mismatch_mask)

    single = run2 = run3p = 0
    idx = 0
    while idx < len(mismatch_mask):
        if not mismatch_mask[idx]:
            idx += 1
            continue
        j = idx
        while j < len(mismatch_mask) and mismatch_mask[j]:
            j += 1
        run_len = j - idx
        if run_len == 1:
            single += 1
        elif run_len == 2:
            run2 += 1
        else:
            run3p += 1
        idx = j

    backbone_mismatch = sum(mismatch_mask[:4])
    modulation_mismatch = sum(mismatch_mask[4:7])
    run_branch_mismatch = sum(mismatch_mask[7:])
    continuous_score = 0
    gap_count = 0
    base_active = 0
    gc_active = 0
    for row in matrix:
        # continuous-state bits are in columns 7 and 8
        if row[7:9] == [0, 1]:
            continuous_score += 1
        elif row[7:9] == [1, 0]:
            continuous_score += 2
        elif row[7:9] == [1, 1]:
            continuous_score += 3
        gap_count += row[4]
        base_active += sum(row[:4])
        gc_active += row[2] + row[3]

    total_positions = max(1, len(matrix))
    active_bases = max(1, base_active)
    gc_ratio = gc_active / active_bases
    mismatch_density = mismatch_count / total_positions
    backbone_density = backbone_mismatch / total_positions
    modulation_density = modulation_mismatch / total_positions
    run_branch_density = run_branch_mismatch / total_positions
    continuous_density = continuous_score / (3.0 * total_positions)
    tail_mismatch = sum(mismatch_mask[15:]) / max(1, total_positions - 15)
    gate_score = 0.45 * modulation_density + 0.55 * continuous_density
    return [
        _round6(mismatch_density),
        _round6(single / total_positions),
        _round6(run2 / total_positions),
        _round6(run3p / total_positions),
        _round6(longest_run / total_positions),
        _round6(backbone_density),
        _round6(modulation_density),
        _round6(run_branch_density),
        _round6(continuous_density),
        _round6(gate_score),
        _round6(gc_ratio),
        _round6(gap_count / total_positions),
    ]


@dataclass
class ConMismatch9Model(BaseSequenceModel):
    model_name: str = "conmismatch9"
    encoder_name: str = "c9"
    feature_names: List[str] = None  # type: ignore[assignment]

    def __init__(self, state=None):
        if self.feature_names is None:
            self.feature_names = [
                "mismatch_density",
                "single_run_density",
                "run2_density",
                "run3p_density",
                "longest_run_density",
                "backbone_mismatch_density",
                "modulation_mismatch_density",
                "run_branch_mismatch_density",
                "continuous_density",
                "gate_score",
                "gc_ratio",
                "gap_density",
            ]
        super().__init__(state=state)
        if all(weight == 0.0 for weight in self.state.weights):
            self.state.bias = -2.8
            self.state.weights = [
                1.9,
                0.9,
                1.2,
                1.6,
                1.4,
                0.8,
                1.7,
                1.5,
                2.0,
                1.1,
                -0.2,
                0.2,
            ]

    def split_branches(self, x):
        try:
            import torch  # type: ignore
        except Exception:  # pragma: no cover - torch is optional
            torch = None

        if torch is not None and hasattr(torch, "Tensor") and isinstance(x, torch.Tensor):
            x_split = [x[:, :, 0:4], x[:, :, 4:7], x[:, :, 7:9]]
            return x_split
        return [self._slice_nested(x, 0, 4), self._slice_nested(x, 4, 7), self._slice_nested(x, 7, 9)]

    def forward(self, x):
        return self.split_branches(x)

    def _slice_nested(self, x, start: int, stop: int):
        return _nested_slice(x, start, stop)

    def feature_vector(self, on_seq: str, off_seq: str) -> List[float]:
        matrix = C9Encoder().encode_pair(on_seq, off_seq)
        return _summarize_c9_matrix(matrix)

    def summarize_matrix(self, matrix: Sequence[Sequence[int]]) -> List[float]:
        return _summarize_c9_matrix(matrix)
