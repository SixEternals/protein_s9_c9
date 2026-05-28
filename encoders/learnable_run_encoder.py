"""Learnable Run encoder based on base-pair embeddings and context CNN.

AGENTS.md compliance: [use_rnafm=False, freeze_rnafm=N/A,
                       split_mode=sgrna_safe, pos_weight=None]
This encoder only consumes positions 1-20 and never computes Run states over
PAM positions 21-23.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from utils.guardrails import check_run_encoding_positions
from utils.sequence import normalize_sequence


BASE_TO_INDEX = {
    "A": 0,
    "U": 1,
    "T": 1,
    "G": 2,
    "C": 3,
    "N": 0,
    "-": 0,
}


def base_pair_to_index(sgrna_base: str, off_base: str) -> int:
    """Map a canonical 4x4 base pair to an index in [0, 15]."""
    left = BASE_TO_INDEX.get(str(sgrna_base).upper(), 0)
    right = BASE_TO_INDEX.get(str(off_base).upper(), 0)
    return left * 4 + right


def encode_base_pair_indices(
    sgrna_seqs: Sequence[str],
    off_seqs: Sequence[str],
    *,
    length: int = 20,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return base-pair ids with shape (B, 20), excluding PAM positions."""
    if len(sgrna_seqs) != len(off_seqs):
        raise ValueError(f"sequence count mismatch: {len(sgrna_seqs)} vs {len(off_seqs)}")
    rows: list[list[int]] = []
    for sgrna, off in zip(sgrna_seqs, off_seqs):
        sgrna_20 = normalize_sequence(str(sgrna), length=23)[:length]
        off_20 = normalize_sequence(str(off), length=23)[:length]
        rows.append([base_pair_to_index(s, o) for s, o in zip(sgrna_20, off_20)])
    check_run_encoding_positions(length, max_pos=20, name="LearnableRunEncoder.base_pairs")
    return torch.tensor(rows, dtype=torch.long, device=device)


class LearnableRunEncoder(nn.Module):
    """Base-pair embedding + position encoding + context CNN.

    The module replaces hand-crafted match/isolated/run2/run3+ states. It
    learns a continuous token representation for each protospacer position and
    lets local convolutions learn mismatch-run patterns from data.
    """

    def __init__(self, d_model: int = 128, length: int = 20, dropout: float = 0.2):
        super().__init__()
        if length != 20:
            raise ValueError("LearnableRunEncoder is constrained to 20nt protospacer inputs")
        self.d_model = int(d_model)
        self.length = int(length)
        self.base_pair_embed = nn.Embedding(16, self.d_model)
        self.pos_embed = nn.Parameter(torch.empty(self.length, self.d_model))
        self.context_cnn = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.base_pair_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        for module in self.context_cnn.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight.data)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        base_pair_indices: torch.Tensor | Sequence[str],
        off_seqs: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Return H_run with shape (B, 20, d_model)."""
        if isinstance(base_pair_indices, torch.Tensor):
            ids = base_pair_indices
            if ids.dtype != torch.long:
                ids = ids.long()
            if ids.dim() != 2 or ids.shape[1] != self.length:
                raise ValueError(f"base_pair_indices must have shape (B, 20), got {tuple(ids.shape)}")
        else:
            if off_seqs is None:
                raise ValueError("off_seqs is required when passing sequence strings")
            ids = encode_base_pair_indices(
                base_pair_indices,
                off_seqs,
                length=self.length,
                device=self.pos_embed.device,
            )

        check_run_encoding_positions(ids.shape[1], max_pos=20, name="LearnableRunEncoder.forward")
        x = self.base_pair_embed(ids.to(self.pos_embed.device))
        x = x + self.pos_embed.unsqueeze(0)
        x = self.context_cnn(x.transpose(1, 2)).transpose(1, 2)
        return x


__all__ = [
    "LearnableRunEncoder",
    "base_pair_to_index",
    "encode_base_pair_indices",
]
