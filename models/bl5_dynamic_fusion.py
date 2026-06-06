"""BL5-3 dynamic fusion model: RNA-FM fine-tune + Run Cross-Attn/Gate.

AGENTS.md compliance: [use_rnafm=True, freeze_rnafm=False,
                       split_mode=sgrna_safe, pos_weight=None,
                       focal_loss=True]
确认本文件遵守 AGENTS.md 约束：BL5 使用 token-level Cross-Attn + Softmax
Gated Fusion；Run 编码只覆盖 positions 1-20；test 由训练脚本加载 best.pt。
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from encoders.learnable_run_encoder import LearnableRunEncoder
from encoders.pam_encoder import PAMEncoder
from utils.guardrails import check_model_config


class RunTokenCNN(nn.Module):
    """1D-CNN over C9 Run features, returning per-position tokens."""

    def __init__(self, in_channels: int = 9, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv1d(128, self.hidden_dim, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, run_features: torch.Tensor, seed_weights: torch.Tensor) -> torch.Tensor:
        """Return H_run with shape (B, 20, hidden_dim)."""
        if run_features.dim() != 3 or run_features.shape[1:] != (20, 9):
            raise ValueError(f"run_features must have shape (B, 20, 9), got {tuple(run_features.shape)}")
        if seed_weights.dim() == 1:
            weights = seed_weights.view(1, 20, 1)
        elif seed_weights.dim() == 2:
            weights = seed_weights.unsqueeze(-1)
        else:
            raise ValueError(f"seed_weights must have shape (20,) or (B,20), got {tuple(seed_weights.shape)}")
        if weights.shape[1] != 20:
            raise ValueError(f"seed_weights must cover exactly 20 positions, got {weights.shape[1]}")
        x = run_features * weights
        x = x.transpose(1, 2)
        return self.net(x).transpose(1, 2)


class BL5CrossAttentionLayer(nn.Module):
    """Bidirectional token-level Cross-Attention between Run and RNA-FM tokens."""

    def __init__(self, d_model: int = 128, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.run_to_rna = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.rna_to_run = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_run = nn.LayerNorm(d_model)
        self.norm_rna = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_run: torch.Tensor,
        h_rna: torch.Tensor,
        rna_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        run_input = h_run
        rna_input = h_rna

        run_delta, _ = self.run_to_rna(
            query=run_input,
            key=rna_input,
            value=rna_input,
            key_padding_mask=rna_padding_mask,
            need_weights=False,
        )
        h_run = self.norm_run(run_input + self.dropout(run_delta))

        rna_delta, _ = self.rna_to_run(
            query=rna_input,
            key=run_input,
            value=run_input,
            need_weights=False,
        )
        h_rna = self.norm_rna(rna_input + self.dropout(rna_delta))
        return h_run, h_rna


class BL5FusionBackend(nn.Module):
    """Cross-Attn + Softmax Gated Fusion backend for two views."""

    def __init__(
        self,
        d_model: int = 128,
        rnafm_dim: int = 640,
        num_heads: int = 4,
        num_layers: int = 1,
        gate_hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.rnafm_proj = nn.Linear(rnafm_dim, self.d_model)
        self.layers = nn.ModuleList(
            [
                BL5CrossAttentionLayer(
                    d_model=self.d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.d_model * 2, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 2),
        )

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        if padding_mask is None:
            return tokens.mean(dim=1)
        valid = (~padding_mask).to(dtype=tokens.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (tokens * valid).sum(dim=1) / denom

    def forward(
        self,
        h_run: torch.Tensor,
        h_rnafm: torch.Tensor,
        rna_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_rna = self.rnafm_proj(h_rnafm)
        for layer in self.layers:
            h_run, h_rna = layer(h_run, h_rna, rna_padding_mask)

        z_run = h_run.mean(dim=1)
        z_rna = self._masked_mean(h_rna, rna_padding_mask)
        gate_logits = self.gate_mlp(torch.cat([z_run, z_rna], dim=-1))
        weights = torch.softmax(gate_logits, dim=-1)
        fused = weights[:, 0:1] * z_run + weights[:, 1:2] * z_rna
        return fused, weights


class BL5RunOnlyDynamicFusion(nn.Module):
    """BL5 run-only fusion model with config-selected backend."""

    def __init__(
        self,
        rnafm_model: nn.Module,
        padding_idx: int = 0,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        cfg = config or {}
        check_model_config(cfg)
        model_cfg = cfg.get("model", cfg)
        self.use_rnafm = bool(model_cfg.get("use_rnafm", True))
        if self.use_rnafm and model_cfg.get("freeze_rnafm") is not False:
            raise ValueError("BL5-3 requires model.freeze_rnafm=false for RNA-FM fine-tuning")
        self.use_learnable_run = bool(model_cfg.get("use_learnable_run", False))
        self.fusion_type = str(model_cfg.get("fusion_type", "cross_attn_gate")).lower()
        self.use_pam_encoder = bool(model_cfg.get("use_pam_encoder", False))
        if self.fusion_type not in {"cross_attn_gate", "simple_concat", "pam_gated_fusion", "run_only", "pam_only", "rnafm_pam_concat", "run_pam_concat"}:
            raise ValueError(
                "model.fusion_type must be 'cross_attn_gate', 'simple_concat', 'pam_gated_fusion', 'run_only', 'pam_only', 'rnafm_pam_concat', or 'run_pam_concat', "
                f"got {self.fusion_type!r}"
            )
        if self.fusion_type == "run_only":
            if self.use_rnafm:
                raise ValueError("run_only fusion requires model.use_rnafm=false")
            if self.use_pam_encoder:
                raise ValueError("run_only fusion requires model.use_pam_encoder=false")
        if self.fusion_type == "pam_only":
            if self.use_rnafm:
                raise ValueError("pam_only fusion requires model.use_rnafm=false")
            if model_cfg.get("use_run") is not False or self.use_learnable_run:
                raise ValueError("pam_only fusion requires model.use_run=false and model.use_learnable_run=false")
            if not self.use_pam_encoder:
                raise ValueError("pam_only fusion requires model.use_pam_encoder=true")
        if self.fusion_type == "rnafm_pam_concat":
            if not self.use_rnafm:
                raise ValueError("rnafm_pam_concat requires model.use_rnafm=true")
            if model_cfg.get("freeze_rnafm") is not False:
                raise ValueError("rnafm_pam_concat requires model.freeze_rnafm=false")
            if model_cfg.get("use_run") is not False or self.use_learnable_run:
                raise ValueError("rnafm_pam_concat requires model.use_run=false and model.use_learnable_run=false")
            if not self.use_pam_encoder:
                raise ValueError("rnafm_pam_concat requires model.use_pam_encoder=true")
        if self.fusion_type == "run_pam_concat":
            if self.use_rnafm:
                raise ValueError("run_pam_concat requires model.use_rnafm=false")
            if model_cfg.get("use_run") is not True or not self.use_learnable_run:
                raise ValueError("run_pam_concat requires model.use_run=true and model.use_learnable_run=true")
            if not self.use_pam_encoder:
                raise ValueError("run_pam_concat requires model.use_pam_encoder=true")
            if model_cfg.get("use_region") is True:
                raise ValueError("run_pam_concat requires model.use_region=false")
        if self.use_pam_encoder and self.fusion_type not in {"simple_concat", "pam_gated_fusion", "pam_only", "rnafm_pam_concat", "run_pam_concat"}:
            raise ValueError("Route-B PAM encoder is currently supported only with simple_concat, pam_gated_fusion, pam_only, rnafm_pam_concat, or run_pam_concat fusion")
        if self.fusion_type == "pam_gated_fusion" and not self.use_pam_encoder:
            raise ValueError("pam_gated_fusion requires model.use_pam_encoder=true")
        if self.fusion_type not in {"pam_only", "rnafm_pam_concat", "run_pam_concat"} and model_cfg.get("use_run") is not True and not self.use_learnable_run:
            raise ValueError("BL5-3 run-only dynamic fusion requires model.use_run=true")
        if model_cfg.get("use_region") is True:
            raise ValueError("BL5-3 run-only version requires model.use_region=false")

        self.rnafm_model = rnafm_model
        self.padding_idx = int(padding_idx)
        self.repr_layer = int(model_cfg.get("repr_layer", cfg.get("repr_layer", 12)))
        self.rnafm_dim = int(model_cfg.get("rnafm_dim", cfg.get("rnafm_dim", 640)))
        d_model = int(model_cfg.get("d_model", 128))
        dropout = float(model_cfg.get("dropout", cfg.get("dropout", 0.2)))
        dropout2 = float(model_cfg.get("dropout2", dropout))
        self.rna_pooling = str(model_cfg.get("rna_pooling", "mean")).lower()
        if self.use_rnafm and self.rna_pooling not in {"mean", "cls"}:
            raise ValueError(f"model.rna_pooling must be 'mean' or 'cls' when use_rnafm=true, got {self.rna_pooling!r}")

        if self.fusion_type not in {"pam_only", "rnafm_pam_concat"}:
            if self.use_learnable_run:
                self.run_encoder = LearnableRunEncoder(
                    d_model=d_model,
                    dropout=float(model_cfg.get("run_dropout", dropout)),
                )
                self.run_cnn = None
            else:
                self.run_encoder = RunTokenCNN(
                    in_channels=9,
                    hidden_dim=d_model,
                    dropout=float(model_cfg.get("run_dropout", dropout)),
                )
                self.run_cnn = self.run_encoder
        else:
            self.run_encoder = None
            self.run_cnn = None
        pam_dim = int(model_cfg.get("pam_dim", 16))
        if self.fusion_type == "cross_attn_gate":
            self.fusion_backend: BL5FusionBackend | None = BL5FusionBackend(
                d_model=d_model,
                rnafm_dim=self.rnafm_dim,
                num_heads=int(model_cfg.get("cross_attn_heads", 4)),
                num_layers=int(model_cfg.get("cross_attn_layers", 1)),
                gate_hidden_dim=int(model_cfg.get("gate_hidden_dim", 64)),
                dropout=dropout,
            )
            classifier_input_dim = d_model
            self.rnafm_proj = None
            self.pam_proj = None
            self.gate_mlp = None
        elif self.fusion_type == "pam_gated_fusion":
            self.fusion_backend = None
            self.rnafm_proj = nn.Linear(self.rnafm_dim, d_model)
            self.pam_proj = nn.Linear(pam_dim, d_model)
            gate_input_dim = d_model + d_model + pam_dim
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_input_dim, d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(d_model, 3),
            )
            classifier_input_dim = self.rnafm_dim + d_model + pam_dim + d_model
        elif self.fusion_type == "rnafm_pam_concat":
            self.fusion_backend = None
            self.rnafm_proj = None
            self.pam_proj = None
            self.gate_mlp = None
            classifier_input_dim = self.rnafm_dim + pam_dim
        elif self.fusion_type == "run_pam_concat":
            self.fusion_backend = None
            self.rnafm_proj = None
            self.pam_proj = None
            self.gate_mlp = None
            classifier_input_dim = d_model + pam_dim
        elif self.fusion_type == "pam_only":
            self.fusion_backend = None
            self.rnafm_proj = None
            self.pam_proj = None
            self.gate_mlp = None
            classifier_input_dim = pam_dim
        elif self.fusion_type == "run_only":
            self.fusion_backend = None
            self.rnafm_proj = None
            self.pam_proj = None
            self.gate_mlp = None
            classifier_input_dim = d_model
        else:
            self.fusion_backend = None
            self.rnafm_proj = None
            self.pam_proj = None
            self.gate_mlp = None
            classifier_input_dim = self.rnafm_dim + d_model
        self.pam_encoder = None
        if self.use_pam_encoder:
            self.pam_encoder = PAMEncoder(d_out=pam_dim)
            if self.fusion_type == "simple_concat":
                classifier_input_dim += pam_dim

        mlp_hidden = int(model_cfg.get("mlp_hidden", 128))
        mlp_hidden2 = model_cfg.get("mlp_hidden2")
        activation_name = str(model_cfg.get("mlp_activation", "elu")).lower()
        activation_cls = nn.ReLU if activation_name == "relu" else nn.ELU
        if mlp_hidden2 is None:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, mlp_hidden),
                activation_cls(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, mlp_hidden),
                activation_cls(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, int(mlp_hidden2)),
                activation_cls(inplace=True),
                nn.Dropout(dropout2),
                nn.Linear(int(mlp_hidden2), 1),
            )
        self._init_new_weights()

    def _init_new_weights(self) -> None:
        modules: list[nn.Module] = [self.classifier]
        if self.run_encoder is not None:
            modules.insert(0, self.run_encoder)
        if self.fusion_backend is not None:
            modules.extend([self.fusion_backend.rnafm_proj, self.fusion_backend.gate_mlp])
        if self.rnafm_proj is not None:
            modules.append(self.rnafm_proj)
        if self.pam_proj is not None:
            modules.append(self.pam_proj)
        if self.gate_mlp is not None:
            modules.append(self.gate_mlp)
        if self.pam_encoder is not None:
            modules.append(self.pam_encoder)
        for module in modules:
            for submodule in module.modules():
                if isinstance(submodule, (nn.Linear, nn.Conv1d)):
                    nn.init.kaiming_uniform_(submodule.weight.data)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        run_features: torch.Tensor,
        seed_weights: torch.Tensor,
        pam_features: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.use_rnafm:
            out = self.rnafm_model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
            h_rnafm = out["representations"][self.repr_layer]
            rna_padding_mask = tokens.eq(self.padding_idx)
        else:
            h_rnafm = None
            rna_padding_mask = None

        if self.fusion_type not in {"pam_only", "rnafm_pam_concat"}:
            if self.use_learnable_run:
                h_run = self.run_encoder(run_features)
            else:
                h_run = self.run_encoder(run_features, seed_weights)
        if self.fusion_type == "cross_attn_gate":
            if self.fusion_backend is None:
                raise RuntimeError("fusion_backend is required for cross_attn_gate")
            fused, gate_weights = self.fusion_backend(h_run, h_rnafm, rna_padding_mask)
            aux = {"gate_weights": gate_weights}
        elif self.fusion_type == "pam_gated_fusion":
            if self.rna_pooling == "cls":
                z_rna = h_rnafm[:, 0, :]
            else:
                z_rna = BL5FusionBackend._masked_mean(h_rnafm, rna_padding_mask)
            z_run = h_run.mean(dim=1)
            if self.pam_encoder is None:
                raise RuntimeError("pam_encoder is required when use_pam_encoder=true")
            if pam_features is None or pam_features.numel() == 0:
                raise ValueError("PAM features are required when model.use_pam_encoder=true")
            z_pam = self.pam_encoder(pam_features)
            z_rna_proj = self.rnafm_proj(z_rna)
            z_pam_proj = self.pam_proj(z_pam)
            view_summary = torch.cat([z_rna_proj, z_run, z_pam], dim=-1)
            gate_logits = self.gate_mlp(view_summary)
            gate = torch.softmax(gate_logits, dim=-1)
            z_weighted = gate[:, 0:1] * z_rna_proj + gate[:, 1:2] * z_run + gate[:, 2:3] * z_pam_proj
            fused = torch.cat([z_rna, z_run, z_pam, z_weighted], dim=-1)
            aux = {"gate_weights": gate}
        elif self.fusion_type == "rnafm_pam_concat":
            if self.rna_pooling == "cls":
                z_rna = h_rnafm[:, 0, :]
            else:
                z_rna = BL5FusionBackend._masked_mean(h_rnafm, rna_padding_mask)
            if self.pam_encoder is None:
                raise RuntimeError("pam_encoder is required for rnafm_pam_concat")
            if pam_features is None or pam_features.numel() == 0:
                raise ValueError("PAM features are required for rnafm_pam_concat")
            z_pam = self.pam_encoder(pam_features)
            fused = torch.cat([z_rna, z_pam], dim=-1)
            aux = {}
        elif self.fusion_type == "run_pam_concat":
            if self.run_encoder is None:
                raise RuntimeError("run_encoder is required for run_pam_concat")
            if self.pam_encoder is None:
                raise RuntimeError("pam_encoder is required for run_pam_concat")
            if pam_features is None or pam_features.numel() == 0:
                raise ValueError("PAM features are required for run_pam_concat")
            # h_run already computed in the fusion_type check block above (line ~370);
            # run_pam_concat requires use_learnable_run=true, so it paths through
            # self.run_encoder(run_features) there — no need to recompute.
            z_run = h_run.mean(dim=1)
            z_pam = self.pam_encoder(pam_features)
            fused = torch.cat([z_run, z_pam], dim=-1)
            aux = {}
        elif self.fusion_type == "pam_only":
            if self.pam_encoder is None:
                raise RuntimeError("pam_encoder is required for pam_only")
            if pam_features is None or pam_features.numel() == 0:
                raise ValueError("PAM features are required for pam_only")
            fused = self.pam_encoder(pam_features)
            aux = {}
        elif self.fusion_type == "run_only":
            z_run = h_run.mean(dim=1)
            fused = z_run
            aux = {}
        else:
            if self.rna_pooling == "cls":
                z_rna = h_rnafm[:, 0, :]
            else:
                z_rna = BL5FusionBackend._masked_mean(h_rnafm, rna_padding_mask)
            z_run = h_run.mean(dim=1)
            fused_parts = [z_rna, z_run]
            if self.use_pam_encoder:
                if self.pam_encoder is None:
                    raise RuntimeError("pam_encoder is required when use_pam_encoder=true")
                if pam_features is None or pam_features.numel() == 0:
                    raise ValueError("PAM features are required when model.use_pam_encoder=true")
                fused_parts.append(self.pam_encoder(pam_features))
            fused = torch.cat(fused_parts, dim=-1)
            aux = {}
        logits = self.classifier(fused).squeeze(-1)
        if return_aux:
            return logits, aux
        return logits


__all__ = [
    "BL5FusionBackend",
    "BL5RunOnlyDynamicFusion",
    "RunTokenCNN",
]
