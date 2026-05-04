from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ConMismatch9TorchConfig:
    hidden_dim: int = 96
    dropout: float = 0.20
    attn_heads: int = 4
    attn_layers: int = 2
    run_base_width: int = 2
    run_state_width: int = 2
    aux_init_scale: float = 0.0
    aux_max_scale: float = 0.50
    ablation_mode: str = "full"


class CNNBackbone(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(4, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x_base: torch.Tensor) -> torch.Tensor:
        x = x_base.transpose(1, 2)
        x = self.input(x)
        residual = x
        x = self.local(x)
        return (x + residual).transpose(1, 2)


class MIModulationModule(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1),
        )

    def forward(self, x_mi: torch.Tensor, backbone: torch.Tensor) -> torch.Tensor:
        params = self.net(x_mi.transpose(1, 2)).transpose(1, 2)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        return backbone * (1.0 + gamma) + beta


class RunBandwidthMaskGenerator(nn.Module):
    """Dynamic Run-Attn mask driven by C9 continuous-mismatch state bits.

    The ConMismatch9 paper specifies a bandwidth-masked Transformer for
    bits 8-9, but does not give a numeric bandwidth formula. This
    implementation uses a monotonic, configurable mapping:
    bandwidth = base_width + state_code * state_width, where state_code is
    the binary value of 00/01/10/11. It is therefore a real local attention
    mask, not standard full attention, while keeping the paper-underspecified
    width rule explicit and tunable.
    """

    def __init__(self, base_width: int = 2, state_width: int = 2):
        super().__init__()
        self.base_width = base_width
        self.state_width = state_width

    def forward(self, x_run: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch_size, seq_len, _ = x_run.shape
        device = x_run.device
        state_code = x_run[:, :, 0] * 2.0 + x_run[:, :, 1]
        bandwidth = self.base_width + state_code.long() * self.state_width
        positions = torch.arange(seq_len, device=device)
        distance = torch.abs(positions[None, :, None] - positions[None, None, :])
        allowed = distance <= bandwidth[:, :, None]
        eye = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)
        allowed = allowed | eye
        blocked = ~allowed
        return blocked.unsqueeze(1).expand(batch_size, num_heads, seq_len, seq_len).reshape(batch_size * num_heads, seq_len, seq_len)


class RunAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class RunAttentionBranch(nn.Module):
    def __init__(self, config: ConMismatch9TorchConfig, use_mask: bool = True):
        super().__init__()
        self.num_heads = config.attn_heads
        self.use_mask = use_mask
        self.input = nn.Linear(2, config.hidden_dim)
        self.mask_generator = RunBandwidthMaskGenerator(
            base_width=config.run_base_width,
            state_width=config.run_state_width,
        )
        self.layers = nn.ModuleList(
            [
                RunAttentionBlock(config.hidden_dim, config.attn_heads, config.dropout)
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

    def forward(self, x_run: torch.Tensor) -> torch.Tensor:
        x = self.input(x_run)
        x = x + self.position_encoding[:, : x.shape[1], :].to(dtype=x.dtype, device=x.device)
        attn_mask = self.mask_generator(x_run, self.num_heads) if self.use_mask else None
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class GatedFusionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.pool_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        return self.pool_projection(torch.cat([mean_pool, max_pool], dim=-1))

    def forward(self, cnn_features: torch.Tensor, run_features: torch.Tensor) -> torch.Tensor:
        cnn_pool = self._pool(cnn_features)
        run_pool = self._pool(run_features)
        gate = self.gate(torch.cat([cnn_pool, run_pool], dim=-1))
        fused = gate * cnn_pool + (1.0 - gate) * run_pool
        return self.head(fused).squeeze(-1)


class NormalizedGatedFusionHead(GatedFusionHead):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__(hidden_dim, dropout)
        self.cnn_norm = nn.LayerNorm(hidden_dim)
        self.run_norm = nn.LayerNorm(hidden_dim)

    def forward(self, cnn_features: torch.Tensor, run_features: torch.Tensor) -> torch.Tensor:
        cnn_pool = self.cnn_norm(self._pool(cnn_features))
        run_pool = self.run_norm(self._pool(run_features))
        gate = self.gate(torch.cat([cnn_pool, run_pool], dim=-1))
        fused = gate * cnn_pool + (1.0 - gate) * run_pool
        return self.head(fused).squeeze(-1)


class ConcatFusionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.pool_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        return self.pool_projection(torch.cat([mean_pool, max_pool], dim=-1))

    def forward(self, cnn_features: torch.Tensor, run_features: torch.Tensor) -> torch.Tensor:
        cnn_pool = self._pool(cnn_features)
        run_pool = self._pool(run_features)
        return self.head(torch.cat([cnn_pool, run_pool], dim=-1)).squeeze(-1)


class SingleBranchHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.pool_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        return self.pool_projection(torch.cat([mean_pool, max_pool], dim=-1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self._pool(features)).squeeze(-1)


class ResidualAuxiliaryFusionHead(nn.Module):
    """Keep the strong CNN path as the main logit and add weak MI/Run residuals."""

    def __init__(self, hidden_dim: int, dropout: float, aux_init: float = 0.0, max_aux_scale: float = 0.50):
        super().__init__()
        if max_aux_scale <= 0.0:
            raise ValueError("max_aux_scale must be positive")
        self.max_aux_scale = float(max_aux_scale)
        raw_init = self._raw_scale_init(aux_init, self.max_aux_scale)
        self.main_head = SingleBranchHead(hidden_dim, dropout)
        self.mi_head = SingleBranchHead(hidden_dim, dropout)
        self.run_head = SingleBranchHead(hidden_dim, dropout)
        self.raw_mi_scale = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        self.raw_run_scale = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    @staticmethod
    def _raw_scale_init(initial_scale: float, max_aux_scale: float) -> float:
        ratio = max(-0.999, min(0.999, float(initial_scale) / float(max_aux_scale)))
        return 0.5 * math.log((1.0 + ratio) / (1.0 - ratio))

    def auxiliary_scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        mi_scale = self.max_aux_scale * torch.tanh(self.raw_mi_scale)
        run_scale = self.max_aux_scale * torch.tanh(self.raw_run_scale)
        return mi_scale, run_scale

    def reset_auxiliary_scales(self, value: float = 0.0) -> None:
        raw_value = self._raw_scale_init(value, self.max_aux_scale)
        with torch.no_grad():
            self.raw_mi_scale.fill_(raw_value)
            self.raw_run_scale.fill_(raw_value)

    def forward(self, backbone_features: torch.Tensor, mi_features: torch.Tensor, run_features: torch.Tensor) -> torch.Tensor:
        main_logit = self.main_head(backbone_features)
        mi_logit = self.mi_head(mi_features)
        run_logit = self.run_head(run_features)
        mi_scale, run_scale = self.auxiliary_scales()
        return main_logit + mi_scale * mi_logit + run_scale * run_logit


def _prefixed_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


class ConMismatch9TorchModel(nn.Module):
    VALID_ABLATION_MODES = {
        "full",
        "legacy_full",
        "run_attn_no_mask",
        "fusion_norm",
        "run_attn_no_mask_fusion_norm",
        "no_mi",
        "no_run_attn",
        "no_fusion",
        "only_cnn",
        "no_mi_no_run_attn",
    }

    def __init__(self, config: ConMismatch9TorchConfig | None = None):
        super().__init__()
        self.config = config or ConMismatch9TorchConfig()
        self.ablation_mode = self.config.ablation_mode.strip().lower().replace("-", "_")
        if self.ablation_mode not in self.VALID_ABLATION_MODES:
            allowed = ", ".join(sorted(self.VALID_ABLATION_MODES))
            raise ValueError(f"unsupported ConMismatch9 ablation_mode: {self.config.ablation_mode!r}; expected one of {allowed}")

        self.use_mi = self.ablation_mode not in {"no_mi", "only_cnn", "no_mi_no_run_attn"}
        self.use_run_attn = self.ablation_mode not in {"no_run_attn", "only_cnn", "no_mi_no_run_attn"}
        self.use_residual_fusion = self.ablation_mode == "full"
        self.use_gated_fusion = self.ablation_mode not in {"no_fusion", "full"}

        self.backbone = CNNBackbone(self.config.hidden_dim, self.config.dropout)
        self.mi = MIModulationModule(self.config.hidden_dim, self.config.dropout) if self.use_mi else None
        self.run_attn = (
            RunAttentionBranch(
                self.config,
                use_mask=self.ablation_mode not in {"run_attn_no_mask", "run_attn_no_mask_fusion_norm"},
            )
            if self.use_run_attn
            else None
        )
        if self.use_run_attn:
            if self.use_residual_fusion:
                self.fusion = ResidualAuxiliaryFusionHead(
                    self.config.hidden_dim,
                    self.config.dropout,
                    aux_init=self.config.aux_init_scale,
                    max_aux_scale=self.config.aux_max_scale,
                )
            elif not self.use_gated_fusion:
                self.fusion = ConcatFusionHead(self.config.hidden_dim, self.config.dropout)
            elif self.ablation_mode in {"fusion_norm", "run_attn_no_mask_fusion_norm"}:
                self.fusion = NormalizedGatedFusionHead(self.config.hidden_dim, self.config.dropout)
            else:
                self.fusion = GatedFusionHead(self.config.hidden_dim, self.config.dropout)
            self.single_head = None
        else:
            self.fusion = None
            self.single_head = SingleBranchHead(self.config.hidden_dim, self.config.dropout)

    def warmstart_main_path(self, state_dict: dict[str, torch.Tensor]) -> None:
        if not isinstance(self.fusion, ResidualAuxiliaryFusionHead):
            raise ValueError("warmstart_main_path is only supported for residual full mode")

        backbone_state = _prefixed_state_dict(state_dict, "backbone.")
        main_head_state = _prefixed_state_dict(state_dict, "single_head.")
        if not main_head_state:
            main_head_state = _prefixed_state_dict(state_dict, "fusion.main_head.")
        if not backbone_state:
            raise ValueError("warmstart checkpoint does not contain a backbone state")
        if not main_head_state:
            raise ValueError("warmstart checkpoint does not contain a compatible main head state")

        self.backbone.load_state_dict(backbone_state, strict=True)
        self.fusion.main_head.load_state_dict(main_head_state, strict=True)
        self.fusion.reset_auxiliary_scales(0.0)

    def warmstart_main_path_from_checkpoint(self, checkpoint: dict[str, object]) -> None:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError("warmstart checkpoint does not contain a state dict")
        self.warmstart_main_path(state_dict)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_split = [x[:, :, 0:4], x[:, :, 4:7], x[:, :, 7:9]]
        x_base, x_mi, x_run = x_split
        backbone_features = self.backbone(x_base)
        cnn_features = self.mi(x_mi, backbone_features) if self.mi is not None else backbone_features
        if self.run_attn is None:
            if self.single_head is None:
                raise RuntimeError("single_branch head is not initialized")
            return self.single_head(cnn_features)
        if self.fusion is None:
            raise RuntimeError("fusion head is not initialized")
        run_features = self.run_attn(x_run)
        if isinstance(self.fusion, ResidualAuxiliaryFusionHead):
            return self.fusion(backbone_features, cnn_features, run_features)
        return self.fusion(cnn_features, run_features)
