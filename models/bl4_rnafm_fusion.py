"""BL4 — RNA-FM + Run fusion model.

AGENTS.md compliance: [use_rnafm=True, freeze_rnafm=True,
                       split_mode=sgrna_safe, pos_weight=auto]
确认本文件遵守 AGENTS.md 约束 #15：含 RNA-FM，归属 BL4，禁止叫 BL3。

RNA-FM encodes the sgRNA<sep>off sequence → CLS embedding [640].
Run encoder encodes mismatch runs → CNN → [hidden_dim].
Fusion: concat → MLP → logits.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class RunCNN(nn.Module):
    """1D-CNN on Run-encoded features."""

    def __init__(self, in_channels: int = 9, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )
        self.hidden_dim = hidden_dim

    def forward(self, run_features: torch.Tensor, seed_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            run_features: (B, 20, 9)
            seed_weights: (20,)
        Returns:
            (B, hidden_dim)
        """
        w = seed_weights.unsqueeze(0).unsqueeze(-1)  # (1, 20, 1)
        x = run_features * w  # (B, 20, 9)
        x = x.transpose(1, 2)  # (B, 9, 20)
        x = self.net(x)  # (B, hidden_dim, 20)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, hidden_dim)
        return x


class BL3RNAFMFusion(nn.Module):
    """RNA-FM CLS + Run CNN → fusion MLP."""

    def __init__(
        self,
        rnafm_model: nn.Module | None = None,
        padding_idx: int = 0,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        cfg = config or {}
        self.rnafm_model = rnafm_model
        self.padding_idx = int(padding_idx)
        self.freeze_rnafm = bool(cfg.get("freeze_rnafm", True))
        self.repr_layer = int(cfg.get("repr_layer", 12))

        # Run CNN
        run_cfg = cfg.get("run", {})
        self.run_cnn = RunCNN(
            in_channels=9,
            hidden_dim=run_cfg.get("hidden_dim", 128),
            dropout=run_cfg.get("dropout", 0.2),
        )

        # Fusion MLP
        self.rnafm_dim = int(cfg.get("rnafm_dim", 640))
        fusion_dim = self.rnafm_dim + self.run_cnn.hidden_dim
        mlp_hidden = int(cfg.get("mlp_hidden", 256))
        dropout = float(cfg.get("dropout", 0.2))
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

        if self.rnafm_model is not None and self.freeze_rnafm:
            for p in self.rnafm_model.parameters():
                p.requires_grad = False
            self.rnafm_model.eval()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.rnafm_model is not None and self.freeze_rnafm:
            self.rnafm_model.eval()
        return self

    def forward(
        self,
        tokens_or_emb: torch.Tensor,
        run_features: torch.Tensor,
        seed_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens_or_emb: (B, seq_len) RNA-FM token ids  OR  (B, 640) precomputed embedding
            run_features: (B, 20, 9)
            seed_weights: (20,)
        Returns:
            logits: (B,)
        """
        # RNA-FM branch
        if tokens_or_emb.dim() == 2 and tokens_or_emb.dtype == torch.float32 and tokens_or_emb.shape[1] == self.rnafm_dim:
            # Precomputed embedding
            rnafm_emb = tokens_or_emb  # (B, 640)
        else:
            # Token IDs, compute on-the-fly
            attention_mask = tokens_or_emb.ne(self.padding_idx)
            if self.freeze_rnafm:
                with torch.no_grad():
                    out = self.rnafm_model(
                        tokens_or_emb, repr_layers=[self.repr_layer], return_contacts=False
                    )
            else:
                out = self.rnafm_model(
                    tokens_or_emb, repr_layers=[self.repr_layer], return_contacts=False
                )
            embeddings = out["representations"][self.repr_layer]
            rnafm_emb = embeddings[:, 0, :]  # (B, 640)

        # Run branch
        run_emb = self.run_cnn(run_features, seed_weights)  # (B, hidden_dim)

        # Fusion
        fused = torch.cat([rnafm_emb, run_emb], dim=-1)  # (B, 640 + hidden_dim)
        return self.fusion(fused).squeeze(-1)  # (B,)
