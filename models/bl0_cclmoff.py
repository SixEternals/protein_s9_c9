from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from utils.rnafm import load_rnafm


@dataclass
class BL0CCLMoffConfig:
    input_dim: int = 640
    head_type: str = "official_mlp"
    transformer_dim: int = 640
    transformer_layers: int = 1
    attention_heads: int = 8
    ffn_dim: int = 1024
    mlp_hidden_dim: int = 64
    dropout: float = 0.2
    repr_layer: int = 12
    pool: str = "cls"
    freeze_rnafm: bool = True


class BL0CCLMoffHead(nn.Module):
    """CCLMoff BL0 head.

    Default `official_mlp` follows the public CCLMoff code:
    RNA-FM CLS embedding -> Linear(640,64) -> ELU -> Linear(64,1).
    The optional `transformer_mlp` path is kept for later BL0-plus experiments,
    but it is not the strict BL0 reproduction.
    """

    def __init__(self, config: BL0CCLMoffConfig | None = None):
        super().__init__()
        self.config = config or BL0CCLMoffConfig()
        if self.config.head_type not in {"official_mlp", "transformer_mlp"}:
            raise ValueError("head_type must be 'official_mlp' or 'transformer_mlp'")
        if self.config.pool not in {"cls", "masked_mean"}:
            raise ValueError("pool must be 'cls' or 'masked_mean'")

        if self.config.head_type == "official_mlp":
            self.input_projection = nn.Identity()
            self.transformer = nn.Identity()
            classifier_dim = self.config.input_dim
        else:
            self.input_projection = (
                nn.Identity()
                if self.config.input_dim == self.config.transformer_dim
                else nn.Linear(self.config.input_dim, self.config.transformer_dim)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.config.transformer_dim,
                nhead=self.config.attention_heads,
                dim_feedforward=self.config.ffn_dim,
                dropout=self.config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.transformer_layers)
            classifier_dim = self.config.transformer_dim

        self.dropout = nn.Dropout(self.config.dropout)
        self.dense1 = nn.Linear(classifier_dim, self.config.mlp_hidden_dim)
        self.elu = nn.ELU()
        self.dense2 = nn.Linear(self.config.mlp_hidden_dim, 1)
        for module in (self.dense1, self.dense2):
            nn.init.kaiming_uniform_(module.weight.data)

    def pool_tokens(self, encoded: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.config.pool == "cls":
            return encoded[:, 0, :]
        if attention_mask is None:
            return encoded.mean(dim=1)
        mask = attention_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (encoded * mask).sum(dim=1) / denom

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError(f"expected embeddings with shape [batch, tokens, dim], got {tuple(embeddings.shape)}")
        x = self.input_projection(embeddings)
        if self.config.head_type == "transformer_mlp":
            src_key_padding_mask = None
            if attention_mask is not None:
                src_key_padding_mask = ~attention_mask.to(dtype=torch.bool)
            encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        else:
            encoded = x
        pooled = self.pool_tokens(encoded, attention_mask)
        hidden = self.elu(self.dropout(self.dense1(pooled)))
        logits = self.dense2(self.dropout(hidden))
        return logits.squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self.forward(embeddings, attention_mask))


class BL0CCLMoffModel(nn.Module):
    """Wrapped RNA-FM BL0 model.

    The wrapper expects tokenized RNA-FM input. It deliberately contains no
    region encoder or run encoder so BL0 remains a clean RNA-FM baseline.
    """

    def __init__(
        self,
        rnafm_model: nn.Module,
        padding_idx: int,
        config: BL0CCLMoffConfig | None = None,
    ):
        super().__init__()
        self.rnafm_model = rnafm_model
        self.padding_idx = int(padding_idx)
        self.config = config or BL0CCLMoffConfig()
        self.freeze_rnafm = bool(self.config.freeze_rnafm)
        self.head = BL0CCLMoffHead(self.config)
        if self.freeze_rnafm:
            for parameter in self.rnafm_model.parameters():
                parameter.requires_grad = False
            self.rnafm_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_rnafm:
            self.rnafm_model.eval()
        return self

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attention_mask = tokens.ne(self.padding_idx)
        if self.freeze_rnafm:
            with torch.no_grad():
                rnafm_output = self.rnafm_model(tokens, repr_layers=[self.config.repr_layer], return_contacts=False)
        else:
            rnafm_output = self.rnafm_model(tokens, repr_layers=[self.config.repr_layer], return_contacts=False)
        embeddings = rnafm_output["representations"][self.config.repr_layer]
        return self.head(embeddings, attention_mask)

    @torch.no_grad()
    def predict_proba(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(tokens))


def load_rnafm_from_package(checkpoint_path: str | Path | None = None, allow_download: bool = False):
    return load_rnafm(checkpoint_path, allow_download=allow_download, trust_local_checkpoint=True)


def build_bl0_with_rnafm(
    checkpoint_path: str | Path | None = None,
    allow_download: bool = False,
    config: BL0CCLMoffConfig | None = None,
) -> tuple[BL0CCLMoffModel, Any]:
    config = config or BL0CCLMoffConfig()
    rnafm_model, alphabet = load_rnafm_from_package(checkpoint_path, allow_download)
    embed_dim = int(getattr(getattr(rnafm_model, "args", None), "embed_dim", config.input_dim))
    if embed_dim != config.input_dim:
        config = replace(config, input_dim=embed_dim, transformer_dim=embed_dim)
    model = BL0CCLMoffModel(rnafm_model=rnafm_model, padding_idx=alphabet.padding_idx, config=config)
    return model, alphabet
