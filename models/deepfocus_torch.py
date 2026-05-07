from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DeepFocusTorchConfig:
    input_dim: int = 9
    hidden_dim: int = 96
    dropout: float = 0.20
    attn_heads: int = 4
    attn_layers: int = 2
    ablation_mode: str = "full"


class RegionAwareInception(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.middle = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dilated = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.compress = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.region_attention = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        channels = x.transpose(1, 2)
        local = self.local(channels)
        middle = self.middle(channels)
        middle = F.interpolate(middle, size=seq_len, mode="linear", align_corners=False)
        dilated = self.dilated(channels)
        fused = self.compress(torch.cat([local, middle, dilated], dim=1))
        attention = self.region_attention(channels)
        return (fused * attention).transpose(1, 2)


class CoordinateAttention(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        reduced_dim = max(8, hidden_dim // 4)
        self.channel_gate = nn.Sequential(
            nn.Linear(hidden_dim, reduced_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.position_gate = nn.Sequential(
            nn.Conv1d(hidden_dim, reduced_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(reduced_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_weight = self.channel_gate(x.mean(dim=1)).unsqueeze(1)
        position_weight = self.position_gate(x.transpose(1, 2)).transpose(1, 2)
        return x * channel_weight * position_weight


class CATransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, attn_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, attn_heads, dropout=dropout, batch_first=True)
        self.coord_attention = CoordinateAttention(hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.coord_attention(x)
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class CATransformerEncoder(nn.Module):
    def __init__(self, config: DeepFocusTorchConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CATransformerBlock(config.hidden_dim, config.attn_heads, config.dropout)
                for _ in range(config.attn_layers)
            ]
        )
        self.register_buffer("position_encoding", self._build_position_encoding(23, config.hidden_dim), persistent=False)

    @staticmethod
    def _build_position_encoding(seq_len: int, hidden_dim: int) -> torch.Tensor:
        positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim))
        encoding = torch.zeros(seq_len, hidden_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div)
        encoding[:, 1::2] = torch.cos(positions * div[: encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.position_encoding[:, : x.shape[1], :].to(dtype=x.dtype, device=x.device)
        for layer in self.layers:
            x = layer(x)
        return x


class DeepFocusTorchModel(nn.Module):
    VALID_ABLATION_MODES = {"full", "inception_only"}

    def __init__(self, config: DeepFocusTorchConfig | None = None):
        super().__init__()
        self.config = config or DeepFocusTorchConfig()
        self.ablation_mode = self.config.ablation_mode.strip().lower().replace("-", "_")
        if self.ablation_mode not in self.VALID_ABLATION_MODES:
            allowed = ", ".join(sorted(self.VALID_ABLATION_MODES))
            raise ValueError(f"unsupported DeepFocus ablation_mode: {self.config.ablation_mode!r}; expected one of {allowed}")
        self.inception = RegionAwareInception(self.config.input_dim, self.config.hidden_dim, self.config.dropout)
        if self.ablation_mode == "full":
            self.transformer = CATransformerEncoder(self.config)
        else:
            self.transformer = None
        self.head = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim * 2),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.inception(x)
        if self.transformer is not None:
            features = self.transformer(features)
        pooled = torch.cat([features.mean(dim=1), features.max(dim=1).values], dim=-1)
        return self.head(pooled).squeeze(-1)
