"""BL3b — Seed regression on Run-only encoding.

Adds learnable position embedding and/or seed gate to BL3a Run-only CNN.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class BL3bSeedRegression(nn.Module):
    """Run-only CNN with optional position embedding and seed gate.

    Args:
        config: dict with keys:
            - hidden_dim (int): CNN output channels, default 128
            - dropout (float): default 0.2
            - position_embedding (str): "learnable" | "fixed" | "none"
            - use_seed_gate (bool): default False
            - cnn_channels (list[int]): default [64, 128]
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        cfg = config or {}

        self.hidden_dim = cfg.get("hidden_dim", 128)
        self.dropout = cfg.get("dropout", 0.2)
        pe_mode = cfg.get("position_embedding", "learnable")
        use_gate = cfg.get("use_seed_gate", False)
        cnn_channels = cfg.get("cnn_channels", [64, 128])

        # Build 1D-CNN: 9 -> cnn_channels -> hidden_dim
        layers: list[nn.Module] = []
        prev_ch = 9
        for ch in cnn_channels:
            layers.append(nn.Conv1d(prev_ch, ch, kernel_size=3, padding=1))
            layers.append(nn.ELU(inplace=True))
            prev_ch = ch
        layers.append(nn.Conv1d(prev_ch, self.hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.ELU(inplace=True))
        layers.append(nn.Dropout(self.dropout))
        self.cnn = nn.Sequential(*layers)

        # Position embedding (added to CNN output [B, hidden_dim, 20])
        if pe_mode == "learnable":
            self.pos_embed = nn.Parameter(torch.randn(20, self.hidden_dim) * 0.02)
        elif pe_mode == "fixed":
            pe = torch.zeros(20, self.hidden_dim)
            position = torch.arange(0, 20, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.hidden_dim, 2, dtype=torch.float32)
                * -(math.log(10000.0) / self.hidden_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pos_embed", pe)
        else:
            self.pos_embed = None

        # Seed gate (multiplied to CNN output [B, hidden_dim, 20])
        if use_gate:
            gate = torch.ones(20, dtype=torch.float32)
            gate[15:] = 2.0  # positions 16-20 (0-indexed 15-19)
            self.seed_gate = nn.Parameter(gate)
        else:
            self.seed_gate = None

        # Global pool + MLP
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ELU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        region: torch.Tensor,
        run: torch.Tensor,
        seed_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            region: ignored (for API compatibility)
            run: (B, 20, 9) Run-encoded features
            seed_weights: (20,) seed-gradient weights
        Returns:
            logits: (B, 1)
        """
        # Apply seed-gradient weighting
        w = seed_weights.unsqueeze(0).unsqueeze(-1)  # (1, 20, 1)
        x = run * w  # (B, 20, 9)

        # CNN on sequence dimension
        x = x.transpose(1, 2)  # (B, 9, 20)
        x = self.cnn(x)  # (B, hidden_dim, 20)

        # Position embedding: add [hidden_dim, 20] broadcast
        if self.pos_embed is not None:
            x = x + self.pos_embed.T.unsqueeze(0)  # (B, hidden_dim, 20)

        # Seed gate: multiply [20] broadcast
        if self.seed_gate is not None:
            gate = self.seed_gate.view(1, 1, 20)  # (1, 1, 20)
            x = x * gate  # (B, hidden_dim, 20)

        # Pool + MLP
        x = self.pool(x).squeeze(-1)  # (B, hidden_dim)
        return self.mlp(x)  # (B, 1) logits
