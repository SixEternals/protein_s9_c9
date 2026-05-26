"""BL3-hard: Region + Run prior encoding with CNN + MLP.

No RNA-FM. Pure hand-crafted feature pipeline.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BL3HardPrior(nn.Module):
    """Region + Run → weighted → 1D-CNN → GlobalAvgPool → MLP → probability.

    Supports:
      - learnable seed-gradient weights (BL3-hard-C)
      - ablation: use_region / use_run switches
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        cfg = config or {}

        self.use_region = cfg.get("use_region", True)
        self.use_run = cfg.get("use_run", True)
        self.learnable_weights = cfg.get("learnable_seed_weights", False)

        # Input channels
        in_channels = 0
        if self.use_region:
            in_channels += 9
        if self.use_run:
            in_channels += 9
        if in_channels == 0:
            raise ValueError("At least one of use_region or use_run must be True")
        self.in_channels = in_channels

        # CNN hyperparameters
        cnn_channels = cfg.get("cnn_channels", [64, 128])
        kernel_size = cfg.get("kernel_size", 3)
        dropout = cfg.get("dropout", 0.3)
        mlp_hidden = cfg.get("mlp_hidden_dim", 64)

        # Optional learnable seed weights
        if self.learnable_weights:
            self.seed_weight_logits = nn.Parameter(torch.zeros(20))

        # 1D-CNN on sequence dimension
        layers: list[nn.Module] = []
        prev_ch = in_channels
        for ch in cnn_channels:
            layers.append(nn.Conv1d(prev_ch, ch, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU(inplace=True))
            prev_ch = ch
        self.cnn = nn.Sequential(*layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(prev_ch, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        region: torch.Tensor,
        run: torch.Tensor,
        seed_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            region: (B, 20, 9)
            run:    (B, 20, 9)
            seed_weights: (B, 20) or (20,)
        Returns:
            logits: (B, 1)
        """
        # Build input feature stack
        parts = []
        if self.use_region:
            parts.append(region)
        if self.use_run:
            parts.append(run)
        x = torch.cat(parts, dim=-1)  # (B, 20, in_channels)

        # Apply seed-gradient weighting
        if self.learnable_weights:
            # Learnable weights, initialized near soft-gradient
            w = torch.sigmoid(self.seed_weight_logits) * 2.0  # (20,), range ~0-2
            w = w.unsqueeze(0).unsqueeze(-1)  # (1, 20, 1)
        else:
            w = seed_weights.unsqueeze(-1)  # (B, 20, 1) or broadcast from (20,)
        x = x * w  # (B, 20, in_channels)

        # Conv1d
        x = x.transpose(1, 2)  # (B, in_channels, 20)
        x = self.cnn(x)        # (B, cnn_channels[-1], 20)
        x = self.global_pool(x)  # (B, cnn_channels[-1], 1)
        x = x.squeeze(-1)      # (B, cnn_channels[-1])

        return self.mlp(x)     # (B, 1)
