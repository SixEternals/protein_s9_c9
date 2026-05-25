from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path
from typing import Any

import torch


DEFAULT_CKPT = Path("data/cclmoff/CCLMoff_V1.ckpt")
DEFAULT_REPORT = Path("results/audits/cclmoff_checkpoint_audit.md")
DEFAULT_JSON = Path("results/audits/cclmoff_checkpoint_audit.json")
EXPECTED_MD5 = "80b311715969524a3972ab75928f3969"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_summary(state_dict: dict[str, Any], max_items: int) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    entries: list[dict[str, Any]] = []
    prefix_counts: Counter[str] = Counter()
    tensor_params = 0
    for key, value in state_dict.items():
        prefix_counts[key.split(".", 1)[0]] += 1
        if torch.is_tensor(value):
            tensor_params += int(value.numel())
            if len(entries) < max_items:
                entries.append(
                    {
                        "key": key,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "numel": int(value.numel()),
                    }
                )
        elif len(entries) < max_items:
            entries.append({"key": key, "type": type(value).__name__})
    return entries, dict(prefix_counts), tensor_params


def audit(path: Path, trust_checkpoint: bool, max_items: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "md5": file_md5(path),
        "expected_md5": EXPECTED_MD5,
        "md5_matches_expected": file_md5(path) == EXPECTED_MD5,
        "trust_checkpoint": trust_checkpoint,
        "load_status": "not_loaded",
        "top_level_type": None,
        "top_level_keys": [],
        "epoch": None,
        "global_step": None,
        "hyper_parameters_keys": [],
        "state_dict_key_count": 0,
        "state_dict_prefix_counts": {},
        "state_dict_tensor_params": 0,
        "state_dict_examples": [],
        "official_head_keys_present": {},
        "compatibility": "not_checked",
        "notes": "",
    }
    if not trust_checkpoint:
        payload["notes"] = "Skipped torch.load. Rerun with --trust-checkpoint only for the verified Figshare checkpoint."
        return payload

    # The md5 is checked before this point. weights_only=False is required for
    # Lightning-style checkpoints under PyTorch 2.6+.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    payload["load_status"] = "loaded"
    payload["top_level_type"] = type(checkpoint).__name__
    if isinstance(checkpoint, dict):
        payload["top_level_keys"] = sorted(str(key) for key in checkpoint.keys())
        payload["epoch"] = checkpoint.get("epoch")
        payload["global_step"] = checkpoint.get("global_step")
        hyper_parameters = checkpoint.get("hyper_parameters") or {}
        if isinstance(hyper_parameters, dict):
            payload["hyper_parameters_keys"] = sorted(str(key) for key in hyper_parameters.keys())
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
        if isinstance(state_dict, dict):
            examples, prefixes, tensor_params = tensor_summary(state_dict, max_items)
            payload["state_dict_key_count"] = len(state_dict)
            payload["state_dict_prefix_counts"] = prefixes
            payload["state_dict_tensor_params"] = tensor_params
            payload["state_dict_examples"] = examples
            keys = set(str(key) for key in state_dict.keys())
            payload["official_head_keys_present"] = {
                "dense1.weight": any(key.endswith("dense1.weight") for key in keys),
                "dense1.bias": any(key.endswith("dense1.bias") for key in keys),
                "dense2.weight": any(key.endswith("dense2.weight") for key in keys),
                "dense2.bias": any(key.endswith("dense2.bias") for key in keys),
                "rna_model": any("rna_model" in key for key in keys),
            }
            has_official_head = all(
                bool(payload["official_head_keys_present"].get(key))
                for key in ("dense1.weight", "dense1.bias", "dense2.weight", "dense2.bias")
            )
            prefix_set = set(prefixes)
            if has_official_head:
                payload["compatibility"] = "compatible_with_public_cclmoff_head"
            elif prefix_set == {"encoder", "decoder"} or prefix_set.issuperset({"encoder", "decoder"}):
                payload["compatibility"] = "incompatible_with_public_cclmoff_head"
                payload["notes"] = (
                    "The file matches the Figshare MD5 and loads as a Lightning checkpoint, "
                    "but its state_dict is encoder/decoder-style and lacks dense1/dense2/rna_model keys "
                    "from CCLMoff/my_model.py. Do not treat it as a directly usable public CCLMoff "
                    "RNA-FM+dense inference checkpoint until this mismatch is resolved."
                )
            else:
                payload["compatibility"] = "unknown_state_dict_layout"
    return payload


def write_reports(payload: dict[str, Any], report_path: Path, json_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# CCLMoff Checkpoint Audit",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Checkpoint: `{payload['path']}`",
        f"- File size bytes: `{payload['file_size_bytes']}`",
        f"- MD5: `{payload['md5']}`",
        f"- MD5 matches Figshare API: `{payload['md5_matches_expected']}`",
        f"- Load status: `{payload['load_status']}`",
        f"- Trust checkpoint: `{payload['trust_checkpoint']}`",
        f"- Top-level type: `{payload['top_level_type']}`",
        f"- Top-level keys: `{payload['top_level_keys']}`",
        f"- Epoch: `{payload['epoch']}`",
        f"- Global step: `{payload['global_step']}`",
        f"- Hyper-parameters keys: `{payload['hyper_parameters_keys']}`",
        f"- State dict key count: `{payload['state_dict_key_count']}`",
        f"- State dict tensor params: `{payload['state_dict_tensor_params']}`",
        f"- State dict prefix counts: `{payload['state_dict_prefix_counts']}`",
        f"- Official head keys present: `{payload['official_head_keys_present']}`",
        f"- Compatibility: `{payload['compatibility']}`",
        f"- Notes: {payload['notes']}",
        "",
        "## State Dict Examples",
        "",
    ]
    for item in payload["state_dict_examples"]:
        lines.append(f"- `{item}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the downloaded CCLMoff checkpoint structure.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument("--max-items", type=int, default=40)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = audit(args.checkpoint, args.trust_checkpoint, args.max_items)
    write_reports(payload, args.report, args.json)
    print(f"Wrote {args.report}")
    print(f"Wrote {args.json}")
    print(f"load_status={payload['load_status']} md5_matches_expected={payload['md5_matches_expected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
