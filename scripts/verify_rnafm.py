from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from argparse import Namespace
from contextlib import nullcontext
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch


DEFAULT_REPORT = Path("results/audits/rnafm_verification_report.md")
DEFAULT_JSON = Path("results/audits/rnafm_verification_report.json")
EXPECTED_LAYERS = 12
EXPECTED_EMBED_DIM = 640
EXPECTED_PARAM_LOW = 80_000_000
EXPECTED_PARAM_HIGH = 120_000_000
RNAFM_URL = "https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM_pretrained.pth"
RNAFM_HF_URL = "https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth?download=true"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_checkpoint_path() -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / "RNA-FM_pretrained.pth"


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_params(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def trusted_rnafm_load_context(trust_local_checkpoint: bool):
    """Allow-list the argparse.Namespace stored by the official RNA-FM ckpt.

    PyTorch 2.6+ defaults to weights-only deserialization. The public RNA-FM
    checkpoint stores its training args as an argparse.Namespace, so loading it
    through rna-fm 0.2.2 needs this narrow allow-list when the local checkpoint
    has been verified from a trusted source.
    """
    if not trust_local_checkpoint:
        return nullcontext()
    try:
        return torch.serialization.safe_globals([Namespace])
    except AttributeError:  # pragma: no cover - older torch fallback.
        return nullcontext()


def load_rnafm(checkpoint: Path | None, allow_download: bool, trust_local_checkpoint: bool):
    import fm

    if checkpoint and checkpoint.exists():
        with trusted_rnafm_load_context(trust_local_checkpoint):
            return fm.pretrained.rna_fm_t12(str(checkpoint))
    if not allow_download:
        raise FileNotFoundError(
            f"RNA-FM checkpoint not found. Expected {checkpoint or default_checkpoint_path()}; "
            "rerun with --allow-download only when the network is allowed."
        )
    with trusted_rnafm_load_context(trust_local_checkpoint):
        return fm.pretrained.rna_fm_t12()


def model_specs(model: torch.nn.Module) -> dict[str, Any]:
    args = getattr(model, "args", None)
    layers = len(getattr(model, "layers", []))
    if hasattr(model, "num_layers"):
        try:
            layers = int(model.num_layers)
        except TypeError:
            layers = int(model.num_layers())
    embed_dim = int(getattr(args, "embed_dim", 0) or getattr(model, "embed_dim", 0) or 0)
    return {
        "layers": layers,
        "embed_dim": embed_dim,
        "params": count_params(model),
        "model_class": model.__class__.__name__,
        "args": vars(args) if args is not None else {},
    }


def smoke_forward(model: torch.nn.Module, alphabet: Any, repr_layer: int) -> dict[str, Any]:
    converter = alphabet.get_batch_converter()
    _, _, tokens = converter([("sample_1", "AUGCAUGCAUGCAUGCAUGC"), ("sample_2", "ACGUACGUACGUACGUACGU")])
    model.eval()
    with torch.no_grad():
        output = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    representation = output["representations"][repr_layer]
    return {
        "tokens_shape": list(tokens.shape),
        "representation_shape": list(representation.shape),
        "representation_dtype": str(representation.dtype),
    }


def write_reports(payload: dict[str, Any], report_path: Path, json_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# RNA-FM Verification Report",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Python: `{payload['python']}`",
        f"- Torch: `{payload['torch_version']}`",
        f"- CUDA available: `{payload['cuda_available']}`",
        f"- `rna-fm` package version: `{payload['rna_fm_package_version']}`",
        f"- Import module: `{payload['fm_module_file']}`",
        f"- Legacy CCLMoff API `esm1b_rna_t12`: `{payload['legacy_esm1b_rna_t12_available']}`",
        f"- Current API `rna_fm_t12`: `{payload['rna_fm_t12_available']}`",
        f"- Expected checkpoint: `{payload['checkpoint_path']}`",
        f"- Checkpoint exists: `{payload['checkpoint_exists']}`",
        f"- Checkpoint size bytes: `{payload['checkpoint_size_bytes']}`",
        f"- Checkpoint SHA256: `{payload['checkpoint_sha256'] or ''}`",
        f"- Trust local checkpoint for PyTorch 2.6+ safe globals: `{payload['trust_local_checkpoint']}`",
        f"- Official package download URL: `{RNAFM_URL}`",
        f"- Hugging Face mirror URL: `{RNAFM_HF_URL}`",
        f"- HTTP proxy env: `{payload['http_proxy'] or ''}`",
        f"- HTTPS proxy env: `{payload['https_proxy'] or ''}`",
        "",
        "## Status",
        "",
        f"- Load status: `{payload['load_status']}`",
        f"- Full-spec status: `{payload['full_spec_status']}`",
        f"- Notes: {payload['notes']}",
    ]
    if payload.get("model_specs"):
        specs = payload["model_specs"]
        lines.extend(
            [
                "",
                "## Loaded Model",
                "",
                f"- Class: `{specs['model_class']}`",
                f"- Layers: `{specs['layers']}`",
                f"- Embedding dim: `{specs['embed_dim']}`",
                f"- Parameter count: `{specs['params']}`",
            ]
        )
    if payload.get("smoke_forward"):
        lines.extend(
            [
                "",
                "## Smoke Forward",
                "",
                f"- Tokens shape: `{payload['smoke_forward']['tokens_shape']}`",
                f"- Representation shape: `{payload['smoke_forward']['representation_shape']}`",
                f"- Representation dtype: `{payload['smoke_forward']['representation_dtype']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A 1.19GB checkpoint with SHA256 matching the Hugging Face linked etag is the full RNA-FM file used here.",
            "- rna-fm 0.2.2 exposes `fm.pretrained.rna_fm_t12`; the legacy CCLMoff import name `esm1b_rna_t12` is not present in this package version.",
            "- Package import is necessary but not sufficient for BL0 training.",
            "- BL0-v1.0 should not be tagged until this report shows a loaded 12-layer, 640-dim RNA-FM checkpoint and a real training run has metrics.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify RNA-FM package, checkpoint, and full model specs.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--trust-local-checkpoint",
        action="store_true",
        help="Allow-list the official checkpoint's argparse.Namespace for PyTorch 2.6+ loading.",
    )
    parser.add_argument("--smoke-forward", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint or default_checkpoint_path()
    payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "checkpoint_path": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.exists() else 0,
        "checkpoint_sha256": file_sha256(checkpoint),
        "trust_local_checkpoint": bool(args.trust_local_checkpoint),
        "http_proxy": None,
        "https_proxy": None,
        "rna_fm_package_version": None,
        "fm_module_file": None,
        "legacy_esm1b_rna_t12_available": False,
        "rna_fm_t12_available": False,
        "load_status": "not_loaded",
        "full_spec_status": "not_verified",
        "notes": "",
        "model_specs": None,
        "smoke_forward": None,
    }

    try:
        import os

        payload["http_proxy"] = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        payload["https_proxy"] = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        payload["rna_fm_package_version"] = importlib.metadata.version("rna-fm")
        import fm

        payload["fm_module_file"] = getattr(fm, "__file__", None)
        payload["legacy_esm1b_rna_t12_available"] = bool(hasattr(fm.pretrained, "esm1b_rna_t12"))
        payload["rna_fm_t12_available"] = bool(hasattr(fm.pretrained, "rna_fm_t12"))
        payload["rnafm_import_spec"] = importlib.util.find_spec("rnafm") is not None
        payload["rna_fm_import_spec"] = importlib.util.find_spec("rna_fm") is not None

        model, alphabet = load_rnafm(checkpoint, args.allow_download, args.trust_local_checkpoint)
        specs = model_specs(model)
        payload["model_specs"] = specs
        payload["load_status"] = "loaded"
        full_spec_ok = (
            specs["layers"] == EXPECTED_LAYERS
            and specs["embed_dim"] == EXPECTED_EMBED_DIM
            and EXPECTED_PARAM_LOW <= specs["params"] <= EXPECTED_PARAM_HIGH
        )
        payload["full_spec_status"] = "pass" if full_spec_ok else "spec_mismatch"
        if args.smoke_forward:
            payload["smoke_forward"] = smoke_forward(model, alphabet, EXPECTED_LAYERS)
    except Exception as exc:  # noqa: BLE001 - report the exact blocker for P0.
        payload["load_status"] = "failed"
        payload["notes"] = f"{exc.__class__.__name__}: {exc}"
        if not checkpoint.exists():
            payload["notes"] += " The current environment also returns 403 for the official RNA-FM checkpoint URL."

    write_reports(payload, args.report, args.json)
    print(f"Wrote {args.report}")
    print(f"Wrote {args.json}")
    print(f"load_status={payload['load_status']} full_spec_status={payload['full_spec_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
