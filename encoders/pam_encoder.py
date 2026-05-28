"""PAM encoder for protospacer-adjacent positions 21-23.

AGENTS.md compliance: [use_rnafm=False, freeze_rnafm=N/A,
                       split_mode=sgrna_safe, pos_weight=None]
确认本文件遵守 AGENTS.md 约束：PAM positions 21-23 单独编码，不参与 Run
连续错配状态计算；模型 forward 只接收数值化 one-hot 张量，不直接消费原始序列。
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from utils.sequence import normalize_sequence


BASE_TO_INDEX = {
    "A": 0,
    "T": 1,
    "U": 1,
    "G": 2,
    "C": 3,
}


def encode_pam_onehot(
    off_seqs: Sequence[str],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Encode off-target PAM positions 21-23 as a (B, 12) one-hot tensor.

    Unknown or gap characters are left as all-zero for that PAM position instead
    of being folded into a real base category.
    """
    rows = torch.zeros((len(off_seqs), 12), dtype=dtype, device=device)
    for row, off_seq in enumerate(off_seqs):
        normalized = normalize_sequence(str(off_seq), length=23)
        pam = normalized[20:23]
        if len(pam) != 3:
            raise ValueError(f"PAM slice must have length 3, got {len(pam)} for {off_seq!r}")
        for pos, base in enumerate(pam):
            idx = BASE_TO_INDEX.get(base.upper())
            if idx is not None:
                rows[row, pos * 4 + idx] = 1.0
    return rows


class PAMEncoder(nn.Module):
    """Small MLP over numeric PAM one-hot features."""

    def __init__(self, d_out: int = 16):
        super().__init__()
        self.d_out = int(d_out)
        self.pam_seq_embed = nn.Sequential(
            nn.Linear(12, self.d_out),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_out, self.d_out),
        )

    def forward(self, pam_onehot: torch.Tensor) -> torch.Tensor:
        if not isinstance(pam_onehot, torch.Tensor):
            raise TypeError("PAMEncoder.forward expects a numeric one-hot tensor, not raw sequence strings")
        if pam_onehot.dim() != 2 or pam_onehot.shape[1] != 12:
            raise ValueError(f"pam_onehot must have shape (B, 12), got {tuple(pam_onehot.shape)}")
        return self.pam_seq_embed(pam_onehot.float())


__all__ = ["PAMEncoder", "encode_pam_onehot"]
