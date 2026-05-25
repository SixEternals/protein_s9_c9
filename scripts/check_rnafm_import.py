from __future__ import annotations

import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from utils.rnafm import (
    DEFAULT_RNAFM_CHECKPOINT,
    count_parameters,
    load_rnafm,
    normalize_pair_sequence,
    rnafm_model_specs,
    split_special_tokens,
    tokenize_rnafm_sequences,
)


REPORT_PATH = Path("results/audits/rnafm_import_check.txt")
JSON_PATH = Path("results/audits/rnafm_import_check.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_reports(lines: list[str], payload: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def check() -> bool:
    lines: list[str] = []
    payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "checkpoint_path": str(DEFAULT_RNAFM_CHECKPOINT),
        "checkpoint_exists": DEFAULT_RNAFM_CHECKPOINT.exists(),
        "checkpoint_size_bytes": DEFAULT_RNAFM_CHECKPOINT.stat().st_size if DEFAULT_RNAFM_CHECKPOINT.exists() else 0,
        "status": "not_started",
    }

    def log(message: str) -> None:
        print(message)
        lines.append(message)

    try:
        log("Step 1: importing fm...")
        import fm

        payload["fm_module_file"] = getattr(fm, "__file__", None)
        payload["rna_fm_package_version"] = importlib.metadata.version("rna-fm")
        payload["rna_fm_t12_available"] = bool(hasattr(fm.pretrained, "rna_fm_t12"))
        log(f"  OK: fm module imported from {payload['fm_module_file']}")
        log(f"  OK: rna-fm package version {payload['rna_fm_package_version']}")

        log("Step 2: loading RNA-FM from audited local checkpoint...")
        model, alphabet = load_rnafm(DEFAULT_RNAFM_CHECKPOINT, allow_download=False)
        specs = rnafm_model_specs(model)
        payload["model_specs"] = specs
        log("  OK: model loaded")
        log(f"  Layers: {specs['layers']}")
        log(f"  Embed dim: {specs['embed_dim']}")
        log(f"  Params: {count_parameters(model) / 1e6:.1f}M")

        log("Step 3: checking alphabet...")
        payload["alphabet"] = {
            "padding_idx": int(alphabet.padding_idx),
            "cls_idx": int(alphabet.cls_idx),
            "sep_idx": int(alphabet.get_idx("<sep>")),
            "unk_idx": int(alphabet.unk_idx),
            "native_sep_available": "<sep>" in getattr(alphabet, "tok_to_idx", {}),
            "prepend_bos": bool(getattr(alphabet, "prepend_bos", False)),
            "append_eos": bool(getattr(alphabet, "append_eos", False)),
            "all_toks": list(getattr(alphabet, "all_toks", [])),
        }
        payload["alphabet"]["separator_policy"] = (
            "native_<sep>" if payload["alphabet"]["native_sep_available"] else "single_<unk>_delimiter"
        )
        log(f"  padding_idx: {payload['alphabet']['padding_idx']}")
        log(f"  cls_idx: {payload['alphabet']['cls_idx']}")
        log(f"  sep_idx: {payload['alphabet']['sep_idx']}")
        log(f"  unk_idx: {payload['alphabet']['unk_idx']}")
        log(f"  native <sep> available: {payload['alphabet']['native_sep_available']}")
        log(f"  separator policy: {payload['alphabet']['separator_policy']}")

        log("Step 4: checking project tokenizer with <sep> preservation...")
        test_seq = normalize_pair_sequence(
            "GAGTCCGAGCAGAAGAAGAA",
            "GAGTCCGAGCAGAAGAAGAA",
            replace_t_with_u=True,
        )
        split_tokens = split_special_tokens(test_seq)
        tokens = tokenize_rnafm_sequences(alphabet, [test_seq])
        sep_positions = (tokens[0] == alphabet.get_idx("<sep>")).nonzero(as_tuple=False).flatten().tolist()
        unk_count = int((tokens[0] == alphabet.unk_idx).sum().item())
        expected_unk_count = 0 if payload["alphabet"]["native_sep_available"] else split_tokens.count("<sep>")
        payload["tokenizer"] = {
            "sequence": test_seq,
            "split_length": len(split_tokens),
            "tokens_shape": list(tokens.shape),
            "sep_positions": sep_positions,
            "unknown_token_count": unk_count,
            "expected_unknown_token_count": expected_unk_count,
            "first_20_token_ids": tokens[0, :20].tolist(),
        }
        log(f"  Sequence: {test_seq}")
        log(f"  Split length: {len(split_tokens)}")
        log(f"  Tokens shape: {list(tokens.shape)}")
        log(f"  <sep> positions: {sep_positions}")
        log(f"  Unknown token count: {unk_count}")
        log(f"  Expected unknown token count under policy: {expected_unk_count}")

        if specs["layers"] != 12 or specs["embed_dim"] != 640:
            raise RuntimeError(f"RNA-FM spec mismatch: {specs}")
        if not sep_positions:
            raise RuntimeError("project tokenizer did not preserve <sep> as a special token")
        if unk_count != expected_unk_count:
            raise RuntimeError(
                "project tokenizer produced unexpected unknown tokens "
                f"(actual={unk_count}, expected={expected_unk_count})"
            )

        payload["status"] = "passed"
        log("")
        log("ALL CHECKS PASSED")
        return True
    except Exception as exc:  # noqa: BLE001 - this is a durable audit script.
        payload["status"] = "failed"
        payload["error"] = f"{exc.__class__.__name__}: {exc}"
        log("")
        log(f"CHECK FAILED: {payload['error']}")
        return False
    finally:
        write_reports(lines, payload)


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
