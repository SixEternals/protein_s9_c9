from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import torch.nn.functional as F

from models.bl0_cclmoff import BL0CCLMoffConfig, build_bl0_with_rnafm
from utils.rnafm import (
    count_parameters,
    normalize_pair_sequence,
    rnafm_model_specs,
    tokenize_rnafm_sequences,
)


CHECKPOINT_PATH = Path("data/rnafm/checkpoints/RNA-FM_pretrained.pth")
CCLMOFF_CSV_PATH = Path("data/cclmoff/09212024_CCLMoff_dataset.csv")
REPORT_PATH = Path("results/smoke_tests/bl0a_smoke_test.md")
JSON_PATH = Path("results/smoke_tests/bl0a_smoke_test.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_reports(lines: list[str], payload: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def smoke_test() -> bool:
    lines: list[str] = []
    payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "cclmoff_csv_path": str(CCLMOFF_CSV_PATH),
        "cclmoff_csv_exists": CCLMOFF_CSV_PATH.exists(),
        "status": "not_started",
    }

    def log(message: str) -> None:
        print(message)
        lines.append(message)

    try:
        log("# BL0a Smoke Test")
        log("")
        log("## 1. Initialize RNA-FM + official MLP head")
        device = choose_device()
        torch.manual_seed(42)
        config = BL0CCLMoffConfig(head_type="official_mlp", freeze_rnafm=True, dropout=0.2)
        model, alphabet = build_bl0_with_rnafm(
            checkpoint_path=CHECKPOINT_PATH,
            allow_download=False,
            config=config,
        )
        model.to(device)
        specs = rnafm_model_specs(model.rnafm_model)
        total_params = count_parameters(model)
        trainable_params = count_parameters(model, trainable_only=True)
        payload["device"] = str(device)
        payload["rnafm_specs"] = specs
        payload["total_params"] = total_params
        payload["trainable_params"] = trainable_params
        log(f"- Device: `{device}`")
        log(f"- RNA-FM specs: `{specs}`")
        log(f"- Total params: `{total_params}`")
        log(f"- Trainable params: `{trainable_params}`")
        if specs["layers"] != 12 or specs["embed_dim"] != 640:
            raise RuntimeError(f"RNA-FM spec mismatch: {specs}")

        log("")
        log("## 2. Tokenize dummy CCLMoff-style pairs")
        dummy_seqs = [
            normalize_pair_sequence("GAGTCCGAGCAGAAGAAGAA", "GAGTCCGAGCAGAAGAAGAA"),
            normalize_pair_sequence("ATGCATGCATGCATGCATGC", "ATGCATGCATGCATGCATGC"),
        ]
        dummy_tokens = tokenize_rnafm_sequences(alphabet, dummy_seqs).to(device)
        sep_idx = int(alphabet.get_idx("<sep>"))
        native_sep_available = "<sep>" in getattr(alphabet, "tok_to_idx", {})
        expected_delimiter_unknown_count = 0 if native_sep_available else 1
        sep_counts = (dummy_tokens == sep_idx).sum(dim=1).detach().cpu().tolist()
        unk_counts = (dummy_tokens == alphabet.unk_idx).sum(dim=1).detach().cpu().tolist()
        payload["dummy_tokenization"] = {
            "tokens_shape": list(dummy_tokens.shape),
            "native_sep_available": native_sep_available,
            "separator_policy": "native_<sep>" if native_sep_available else "single_<unk>_delimiter",
            "sep_counts": sep_counts,
            "unknown_counts": unk_counts,
            "expected_unknown_counts": [expected_delimiter_unknown_count, expected_delimiter_unknown_count],
        }
        log(f"- Tokens shape: `{list(dummy_tokens.shape)}`")
        log(f"- Native <sep> available: `{native_sep_available}`")
        log(f"- Separator policy: `{payload['dummy_tokenization']['separator_policy']}`")
        log(f"- <sep> counts per sample: `{sep_counts}`")
        log(f"- Unknown token counts per sample: `{unk_counts}`")
        if sep_counts != [1, 1]:
            raise RuntimeError(f"expected one <sep> token per sample, got {sep_counts}")
        if unk_counts != [expected_delimiter_unknown_count, expected_delimiter_unknown_count]:
            raise RuntimeError(
                "unexpected unknown-token count in dummy sequences: "
                f"actual={unk_counts}, expected={[expected_delimiter_unknown_count, expected_delimiter_unknown_count]}"
            )

        log("")
        log("## 3. Forward pass")
        model.eval()
        with torch.no_grad():
            logits = model(dummy_tokens)
            probs = torch.sigmoid(logits)
        payload["forward"] = {
            "logits_shape": list(logits.shape),
            "probabilities": probs.detach().cpu().tolist(),
        }
        log(f"- Logits shape: `{list(logits.shape)}`")
        log(f"- Probabilities: `{payload['forward']['probabilities']}`")
        if list(logits.shape) != [2]:
            raise RuntimeError(f"expected logits shape [2], got {list(logits.shape)}")
        if not bool(torch.all((probs >= 0) & (probs <= 1)).item()):
            raise RuntimeError("sigmoid probabilities are outside [0, 1]")

        log("")
        log("## 4. Backward pass")
        model.train()
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=1e-4)
        labels = torch.tensor([1.0, 0.0], device=device)
        logits = model(dummy_tokens)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        dense1_grad = model.head.dense1.weight.grad
        dense2_grad = model.head.dense2.weight.grad
        payload["backward"] = {
            "loss": float(loss.item()),
            "dense1_grad_norm": float(dense1_grad.norm().item()) if dense1_grad is not None else None,
            "dense2_grad_norm": float(dense2_grad.norm().item()) if dense2_grad is not None else None,
        }
        log(f"- Loss: `{loss.item():.6f}`")
        log(f"- dense1 grad norm: `{payload['backward']['dense1_grad_norm']}`")
        log(f"- dense2 grad norm: `{payload['backward']['dense2_grad_norm']}`")
        if dense1_grad is None or dense2_grad is None:
            raise RuntimeError("MLP head did not receive gradients")

        log("")
        log("## 5. Real CCLMoff CSV sample forward pass")
        if not CCLMOFF_CSV_PATH.exists():
            raise FileNotFoundError(f"CCLMoff CSV not found: {CCLMOFF_CSV_PATH}")
        df = pd.read_csv(CCLMOFF_CSV_PATH, nrows=1)
        sgrna = str(df.iloc[0]["sgRNA_seq"])
        off = str(df.iloc[0]["off_seq"])
        label = int(df.iloc[0]["label"])
        real_seq = normalize_pair_sequence(sgrna, off)
        real_tokens = tokenize_rnafm_sequences(alphabet, [real_seq]).to(device)
        real_sep_count = int((real_tokens == sep_idx).sum().item())
        real_unk_count = int((real_tokens == alphabet.unk_idx).sum().item())
        expected_real_unk_count = 0 if native_sep_available else 1
        model.eval()
        with torch.no_grad():
            real_logit = model(real_tokens)
            real_prob = torch.sigmoid(real_logit)
        payload["real_sample"] = {
            "sgRNA_seq": sgrna,
            "off_seq": off,
            "label": label,
            "tokens_shape": list(real_tokens.shape),
            "sep_count": real_sep_count,
            "unknown_count": real_unk_count,
            "expected_unknown_count": expected_real_unk_count,
            "probability": float(real_prob.item()),
        }
        log(f"- sgRNA_seq: `{sgrna}`")
        log(f"- off_seq: `{off}`")
        log(f"- label semantic: `observed_positive` if 1 else `unobserved_candidate`; value=`{label}`")
        log(f"- Tokens shape: `{list(real_tokens.shape)}`")
        log(f"- <sep> count: `{real_sep_count}`")
        log(f"- Unknown token count: `{real_unk_count}`")
        log(f"- Expected unknown token count under policy: `{expected_real_unk_count}`")
        log(f"- Probability: `{real_prob.item():.6f}`")
        if real_sep_count != 1:
            raise RuntimeError(f"expected one <sep> in real sample, got {real_sep_count}")
        if real_unk_count != expected_real_unk_count:
            raise RuntimeError(
                "unexpected unknown-token count in real sample: "
                f"actual={real_unk_count}, expected={expected_real_unk_count}"
            )

        if device.type == "cuda":
            payload["gpu_memory"] = {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
            log("")
            log("## 6. GPU memory")
            log(f"- Allocated GB: `{payload['gpu_memory']['allocated_gb']:.3f}`")
            log(f"- Reserved GB: `{payload['gpu_memory']['reserved_gb']:.3f}`")
            log(f"- Max allocated GB: `{payload['gpu_memory']['max_allocated_gb']:.3f}`")
        else:
            payload["gpu_memory"] = None
            log("")
            log("## 6. GPU memory")
            log("- CPU mode; CUDA unavailable in this session.")

        payload["status"] = "passed"
        log("")
        log("ALL SMOKE TESTS PASSED")
        return True
    except Exception as exc:  # noqa: BLE001 - durable smoke test report.
        payload["status"] = "failed"
        payload["error"] = f"{exc.__class__.__name__}: {exc}"
        log("")
        log(f"SMOKE TEST FAILED: {payload['error']}")
        return False
    finally:
        write_reports(lines, payload)


if __name__ == "__main__":
    raise SystemExit(0 if smoke_test() else 1)
