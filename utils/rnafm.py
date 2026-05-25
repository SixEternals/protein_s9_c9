from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch


DEFAULT_RNAFM_CHECKPOINT = Path("data/rnafm/checkpoints/RNA-FM_pretrained.pth")


def trusted_rnafm_load_context(trust_local_checkpoint: bool = True):
    """Allow-list the training args stored in the official RNA-FM checkpoint.

    The local RNA-FM checkpoint was already checksum-audited in this project.
    PyTorch 2.6+ uses weights-only loading by default, while rna-fm 0.2.2
    expects to deserialize an argparse.Namespace from the checkpoint.
    """
    if not trust_local_checkpoint:
        return nullcontext()
    try:
        return torch.serialization.safe_globals([Namespace])
    except AttributeError:  # pragma: no cover - older torch fallback.
        return nullcontext()


def resolve_rnafm_checkpoint(checkpoint_path: str | Path | None = None) -> Path:
    """Resolve the RNA-FM checkpoint path used by this project."""
    if checkpoint_path:
        return Path(checkpoint_path)
    return DEFAULT_RNAFM_CHECKPOINT


def load_rnafm(
    checkpoint_path: str | Path | None = None,
    *,
    allow_download: bool = False,
    trust_local_checkpoint: bool = True,
):
    """Load RNA-FM with the project's audited local checkpoint by default."""
    import fm

    checkpoint = resolve_rnafm_checkpoint(checkpoint_path)
    with trusted_rnafm_load_context(trust_local_checkpoint):
        if checkpoint.exists():
            return fm.pretrained.rna_fm_t12(str(checkpoint))
        if allow_download:
            return fm.pretrained.rna_fm_t12()
    raise FileNotFoundError(f"RNA-FM checkpoint not found: {checkpoint}")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return int(sum(param.numel() for param in params))


def rnafm_model_specs(model: torch.nn.Module) -> dict[str, Any]:
    args = getattr(model, "args", None)
    layers = len(getattr(model, "layers", []))
    if hasattr(model, "num_layers"):
        value = getattr(model, "num_layers")
        layers = int(value() if callable(value) else value)
    embed_dim = int(getattr(args, "embed_dim", 0) or getattr(model, "embed_dim", 0) or 0)
    return {
        "model_class": model.__class__.__name__,
        "layers": layers,
        "embed_dim": embed_dim,
        "params": count_parameters(model),
    }


def split_special_tokens(sequence: str, special_tokens: Sequence[str] = ("<sep>",)) -> list[str]:
    """Split a CCLMoff pair string while preserving special tokens.

    RNA-FM's default BatchConverter tokenizes by fixed-width character chunks.
    It does not preserve the literal string "<sep>" as a special token, so
    pair inputs need this project-level tokenizer.
    """
    tokens: list[str] = []
    index = 0
    while index < len(sequence):
        matched = None
        for token in special_tokens:
            if sequence.startswith(token, index):
                matched = token
                break
        if matched is not None:
            tokens.append(matched)
            index += len(matched)
        else:
            tokens.append(sequence[index])
            index += 1
    return tokens


def tokenize_rnafm_sequences(alphabet: Any, sequences: Sequence[str]) -> torch.Tensor:
    """Tokenize RNA-FM sequences and preserve CCLMoff's <sep> pair marker."""
    if not sequences:
        raise ValueError("sequences must be non-empty")

    tokenized = [split_special_tokens(seq) for seq in sequences]
    max_len = max(len(items) for items in tokenized)
    width = max_len + int(getattr(alphabet, "prepend_bos", False)) + int(getattr(alphabet, "append_eos", False))
    tokens = torch.empty((len(tokenized), width), dtype=torch.long)
    tokens.fill_(int(alphabet.padding_idx))

    offset = int(getattr(alphabet, "prepend_bos", False))
    if offset:
        tokens[:, 0] = int(alphabet.cls_idx)
    for row, items in enumerate(tokenized):
        ids = [int(alphabet.get_idx(item)) for item in items]
        tokens[row, offset : offset + len(ids)] = torch.tensor(ids, dtype=torch.long)
        if getattr(alphabet, "append_eos", False):
            tokens[row, offset + len(ids)] = int(alphabet.eos_idx)
    return tokens


def normalize_pair_sequence(sgrna_seq: str, off_seq: str, *, replace_t_with_u: bool = True) -> str:
    sgrna = str(sgrna_seq).upper().replace("_", "-")
    off = str(off_seq).upper().replace("_", "-")
    seq = f"{sgrna}<sep>{off}"
    if replace_t_with_u:
        seq = seq.replace("T", "U")
    return seq
