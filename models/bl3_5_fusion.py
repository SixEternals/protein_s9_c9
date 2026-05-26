"""BL3.5-Full: C9 Run + R9 Region bidirectional Cross-Attention fusion.

AGENTS.md compliance: [use_rnafm=False, split_mode=sgrna_safe, pos_weight=auto]
确认本文件遵守 AGENTS.md 约束 #15（BL3/BL3.5 禁止 RNA-FM）和 #14（中端必须动态融合）。

Architecture:
  C9 Run CNN   -> H_c9 [B, 20, d]
  R9 Region CNN -> H_r9 [B, 20, d]
  Cross-Attn(Q=H_c9, K=H_r9, V=H_r9) -> residual + LN
  Cross-Attn(Q=H_r9, K=H_c9, V=H_c9) -> residual + LN
  concat -> proj -> global pool -> MLP -> logit
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SequenceEncoder(nn.Module):
    """1D-CNN encoder: (B, 9, 20) -> (B, hidden_dim, 20)."""

    def __init__(self, in_channels: int = 9, hidden_dim: int = 128,
                 cnn_channels: list[int] | None = None, dropout: float = 0.2):
        super().__init__()
        channels = cnn_channels or [64, 128]
        layers: list[nn.Module] = []
        prev = in_channels
        for ch in channels:
            layers.append(nn.Conv1d(prev, ch, kernel_size=3, padding=1))
            layers.append(nn.ELU(inplace=True))
            prev = ch
        layers.append(nn.Conv1d(prev, hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.ELU(inplace=True))
        layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 9, 20)
        return self.net(x)  # (B, hidden_dim, 20)


class BL35FullFusion(nn.Module):
    """C9 Run + R9 Region dynamic fusion via bidirectional Cross-Attn + Residual + LN.

    Args:
        config: dict with keys:
            - hidden_dim (int): default 128
            - num_heads (int): MHA heads, default 4
            - dropout (float): default 0.2
            - cnn_channels (list[int]): default [64, 128]
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        cfg = config or {}
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.num_heads = int(cfg.get("num_heads", 4))
        self.dropout = float(cfg.get("dropout", 0.2))
        cnn_channels = cfg.get("cnn_channels", [64, 128])

        # Separate encoders for C9 Run and R9 Region
        self.run_encoder = _SequenceEncoder(
            in_channels=9, hidden_dim=self.hidden_dim,
            cnn_channels=cnn_channels, dropout=self.dropout,
        )
        self.region_encoder = _SequenceEncoder(
            in_channels=9, hidden_dim=self.hidden_dim,
            cnn_channels=cnn_channels, dropout=self.dropout,
        )

        # Bidirectional Cross-Attention
        self.cross_attn_c9_to_r9 = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.num_heads,
            dropout=self.dropout, batch_first=True,
        )
        self.cross_attn_r9_to_c9 = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.num_heads,
            dropout=self.dropout, batch_first=True,
        )

        # LayerNorm for residual connections
        self.ln_c9 = nn.LayerNorm(self.hidden_dim)
        self.ln_r9 = nn.LayerNorm(self.hidden_dim)

        # Fusion projection: concat -> proj
        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ELU(inplace=True),
            nn.Dropout(self.dropout),
        )

        # Global pool + MLP head
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
            region: (B, 20, 9) R9 Region-encoded features
            run:    (B, 20, 9) C9 Run-encoded features
            seed_weights: (20,) seed-gradient weights applied to run only
        Returns:
            logits: (B, 1)
        """
        # Apply seed-gradient weighting to C9 Run only
        w = seed_weights.unsqueeze(0).unsqueeze(-1)  # (1, 20, 1)
        run_weighted = run * w  # (B, 20, 9)

        # Encode to per-position features: (B, hidden_dim, 20)
        H_c9 = self.run_encoder(run_weighted.transpose(1, 2))
        H_r9 = self.region_encoder(region.transpose(1, 2))

        # Transpose to (B, seq_len, d_model) for MHA
        H_c9 = H_c9.transpose(1, 2)  # (B, 20, hidden_dim)
        H_r9 = H_r9.transpose(1, 2)  # (B, 20, hidden_dim)

        # Bidirectional Cross-Attention
        # C9 attends to R9
        attn_c9, _ = self.cross_attn_c9_to_r9(H_c9, H_r9, H_r9)
        # R9 attends to C9
        attn_r9, _ = self.cross_attn_r9_to_c9(H_r9, H_c9, H_c9)

        # Residual + LayerNorm
        H_c9_prime = self.ln_c9(H_c9 + attn_c9)
        H_r9_prime = self.ln_r9(H_r9 + attn_r9)

        # Fusion projection
        fused = torch.cat([H_c9_prime, H_r9_prime], dim=-1)  # (B, 20, 2*hidden_dim)
        H_prior = self.fusion_proj(fused)  # (B, 20, hidden_dim)

        # Global pool + MLP
        H_prior_t = H_prior.transpose(1, 2)  # (B, hidden_dim, 20)
        x = self.pool(H_prior_t).squeeze(-1)  # (B, hidden_dim)
        return self.mlp(x)  # (B, 1)
