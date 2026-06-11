#!/usr/bin/env python3
"""BL6-1 eval-only gate weight export.

AGENTS.md compliance: [use_rnafm=True, freeze_rnafm=False,
                       split_mode=sgrna_safe, pos_weight=None,
                       eval_only=True]
确认本文件遵守 AGENTS.md 约束：仅加载 BL6-1 best.pt 做 eval-only gate export；
Run 编码只覆盖 positions 1-20；PAM 使用 off_seq[20:23]。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.bl5_dynamic_fusion import BL5RunOnlyDynamicFusion
from scripts.train_bl5 import (
    BL5Arrays,
    BL5Dataset,
    formal_group_json_split,
    make_live_collate,
    to_device,
)
from utils.config import load_config
from utils.guardrails import check_eval_procedure, check_model_config
from utils.rnafm import load_rnafm


def build_test_loader(config: dict[str, Any], batch_size: int, num_workers: int) -> tuple[DataLoader, np.ndarray, int, Any]:
    """Build a single-GPU test DataLoader from config using the formal BL5 split."""
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    split_cfg = config.get("split", {})

    use_rnafm = bool(model_cfg.get("use_rnafm", True))
    use_learnable_run = bool(model_cfg.get("use_learnable_run", False))
    use_run = bool(model_cfg.get("use_run", True))
    use_pam_encoder = bool(model_cfg.get("use_pam_encoder", False))

    arrays = BL5Arrays(config)

    # Load group labels for formal split
    csv_path = data_cfg.get("cclmoff_csv")
    group_col = data_cfg.get("group_column", "sgRNA_type")
    if not csv_path:
        raise ValueError("cclmoff_csv is required for formal split")
    group_labels = pd.read_csv(csv_path, usecols=[group_col])[group_col].values
    if len(group_labels) != len(arrays.labels):
        raise ValueError(f"group label count mismatch: {len(group_labels)} vs {len(arrays.labels)}")

    split_result = formal_group_json_split(
        arrays.labels.astype(np.int64),
        group_labels,
        split_cfg,
    )
    test_indices = split_result["test"]

    test_dataset = BL5Dataset(arrays, test_indices, None)
    test_len = len(test_dataset)

    # Load RNA-FM for tokenization
    rnafm_cfg = config.get("rnafm", {})
    _, alphabet = load_rnafm(rnafm_cfg.get("checkpoint_path"), trust_local_checkpoint=True)

    collate_fn = make_live_collate(
        alphabet,
        use_rnafm=use_rnafm,
        use_run=use_run,
        use_learnable_run=use_learnable_run,
        use_pam_encoder=use_pam_encoder,
        shuffle_pam=False,
        shuffle_pam_mode="batch",
    )

    pin_memory = torch.cuda.is_available()
    dataloader_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 4

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )

    return test_loader, test_indices, test_len, alphabet


def gate_entropy_np(gate: np.ndarray) -> np.ndarray:
    """Compute per-row entropy: -sum(g * log(g + 1e-12))."""
    eps = 1e-12
    return -np.sum(gate * np.log(gate + eps), axis=1)


def gate_argmax_np(gate: np.ndarray) -> np.ndarray:
    """Return string array of argmax view name."""
    mapping = np.array(["rnafm", "run", "pam"])
    idx = np.argmax(gate, axis=1)
    return mapping[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="BL6-1 eval-only gate weight export")
    parser.add_argument(
        "--config",
        default="configs/bl6_1_pam_gated_fusion.yaml",
        help="Path to BL6-1 YAML config (default: configs/bl6_1_pam_gated_fusion.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        default="results/bl6_1_pam_gated_fusion/checkpoints/best.pt",
        help="Path to BL6-1 best.pt checkpoint",
    )
    parser.add_argument(
        "--output",
        default="results/bl6_1_pam_gated_fusion/gate_predictions.csv",
        help="Output CSV path (default: results/bl6_1_pam_gated_fusion/gate_predictions.csv)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda:0 if available, else cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: 2x config training.batch_size)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop after N batches (debug only, default: None = full test set)",
    )
    parser.add_argument(
        "--validate-existing-predictions",
        default="results/bl6_1_pam_gated_fusion/test_predictions.csv",
        help="Path to original test_predictions.csv for alignment validation",
    )
    args = parser.parse_args()

    # ---- 0. Enforce single-process / no DDP ----
    for env_var in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        val = os.environ.get(env_var, "")
        if val and val not in ("0", "1") and env_var == "WORLD_SIZE" and val == "1":
            continue
        if val and val not in ("0",):
            raise RuntimeError(
                f"export_bl6_1_gate_predictions.py must run in single-process mode. "
                f"Found {env_var}={val!r}. Unset it and re-run without torchrun."
            )

    # ---- 1. Load config and validate ----
    config = load_config(args.config)
    check_model_config(config)

    model_cfg = config.get("model", {})
    fusion_type = str(model_cfg.get("fusion_type", "")).lower()
    use_pam_encoder = bool(model_cfg.get("use_pam_encoder", False))
    use_rnafm = bool(model_cfg.get("use_rnafm", False))
    freeze_rnafm = bool(model_cfg.get("freeze_rnafm", False))
    split_mode = config.get("split_mode", "")
    split_strategy = config.get("split", {}).get("strategy", "")

    if fusion_type != "pam_gated_fusion":
        raise ValueError(f"Expected fusion_type='pam_gated_fusion', got {fusion_type!r}")
    if not use_pam_encoder:
        raise ValueError("Expected model.use_pam_encoder=true")
    if not use_rnafm:
        raise ValueError("Expected model.use_rnafm=true")
    if freeze_rnafm:
        raise ValueError("Expected model.freeze_rnafm=false")
    if split_mode != "sgrna_safe":
        raise ValueError(f"Expected split_mode='sgrna_safe', got {split_mode!r}")
    if split_strategy != "formal_group_json":
        raise ValueError(f"Expected split.strategy='formal_group_json', got {split_strategy!r}")

    # ---- 2. Device ----
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # AMP deliberately disabled — train_bl5.py::predict_probabilities() does not use
    # autocast.  Keeping the forward path identical ensures probability alignment with
    # the original test_predictions.csv.
    use_amp = False

    # ---- 3. Batch size ----
    training_cfg = config.get("training", {})
    batch_size = args.batch_size or int(training_cfg.get("batch_size", 1024)) * 2
    num_workers = int(training_cfg.get("num_workers", 0))

    print(f"[gate_export] device={device} batch_size={batch_size} num_workers={num_workers}")
    print(f"[gate_export] config={args.config}")
    print(f"[gate_export] checkpoint={args.checkpoint}")
    print(f"[gate_export] output={args.output}")

    # ---- 4. Build test loader ----
    test_loader, test_indices, test_len, alphabet = build_test_loader(
        config, batch_size, num_workers
    )
    print(f"[gate_export] test dataset size = {test_len} (expected 954326)")

    if test_len != 954326:
        print(f"[gate_export] WARNING: test size {test_len} != expected 954326", flush=True)

    # ---- 5. Load RNA-FM and model ----
    rnafm_cfg = config.get("rnafm", {})
    rnafm_model, _ = load_rnafm(rnafm_cfg.get("checkpoint_path"), trust_local_checkpoint=True)
    rnafm_model = rnafm_model.to(device)
    # Disable unused head gradients
    for name, param in rnafm_model.named_parameters():
        if "contact_head" in name or "lm_head" in name:
            param.requires_grad = False

    model = BL5RunOnlyDynamicFusion(
        rnafm_model=rnafm_model,
        padding_idx=alphabet.padding_idx if alphabet else 0,
        config=config,
    ).to(device)

    # ---- 6. Load checkpoint ----
    ckpt_path = Path(args.checkpoint)
    check_eval_procedure(ckpt_path, checkpoint_type="best", require_exists=True)
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    print(f"[gate_export] loaded best.pt epoch={best_ckpt.get('epoch', '?')}")

    # ---- 7. Forward pass: collect probs + gate weights ----
    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    gates_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            tokens_or_emb, runs, sw, pam_input, labels = to_device(batch, device)

            # No autocast — match train_bl5.py::predict_probabilities() inference path
            logits, aux = model(tokens_or_emb, runs, sw, pam_input, return_aux=True)

            probs = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
            gate = aux["gate_weights"].detach().cpu().numpy()

            if gate.shape != (len(probs), 3):
                raise RuntimeError(
                    f"Unexpected gate shape: {gate.shape}, expected ({len(probs)}, 3)"
                )

            labels_all.append(labels.detach().cpu().numpy())
            probs_all.append(probs)
            gates_all.append(gate)

            if (batch_index + 1) % 50 == 0:
                print(f"[gate_export] batch {batch_index + 1}/{len(test_loader)}", flush=True)

    labels_np = np.concatenate(labels_all)
    probs_np = np.concatenate(probs_all)
    gates_np = np.concatenate(gates_all)

    if args.max_batches is not None:
        labels_np = labels_np[:args.max_batches * batch_size]
        probs_np = probs_np[:args.max_batches * batch_size]
        gates_np = gates_np[:args.max_batches * batch_size]

    n_exported = len(probs_np)
    print(f"[gate_export] exported {n_exported} rows")

    # ---- 8. Build output DataFrame ----
    data_cfg = config.get("data", {})
    csv_path = data_cfg.get("cclmoff_csv")
    cols = ["sgRNA_type", "sgRNA_seq", "off_seq", "label"]
    df = pd.read_csv(csv_path, usecols=lambda c: c in cols)
    test_df = df.iloc[test_indices[:n_exported]].reset_index(drop=True)

    off_seqs = test_df["off_seq"].astype(str)
    pam_original = off_seqs.str.slice(20, 23)

    gate_rnafm = gates_np[:, 0].astype(np.float32)
    gate_run = gates_np[:, 1].astype(np.float32)
    gate_pam = gates_np[:, 2].astype(np.float32)
    gate_sum = gate_rnafm + gate_run + gate_pam
    gate_entropy = gate_entropy_np(gates_np).astype(np.float32)
    gate_max = np.max(gates_np, axis=1).astype(np.float32)
    gate_argmax = gate_argmax_np(gates_np)
    pam_family = np.where(
        (pam_original.str.len() == 3) & (pam_original.str[1:] == "GG"),
        "NGG",
        "non-NGG",
    )

    out = pd.DataFrame({
        "sample_index": test_indices[:n_exported].astype(np.int64),
        "sgRNA_type": test_df["sgRNA_type"].astype(str),
        "on_seq": test_df["sgRNA_seq"].astype(str),
        "off_seq": off_seqs,
        "PAM_original": pam_original,
        "label": test_df["label"].astype(int),
        "probability": probs_np.astype(np.float32),
        "gate_rnafm": gate_rnafm,
        "gate_run": gate_run,
        "gate_pam": gate_pam,
        "gate_sum": gate_sum,
        "gate_entropy": gate_entropy,
        "gate_max": gate_max,
        "gate_argmax": gate_argmax,
        "pam_family": pam_family,
        "split": "test",
    })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[gate_export] wrote {len(out)} rows to {output_path}")

    # ---- 9. Validation ----
    validation: dict[str, Any] = {
        "output_path": str(output_path),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_type": "best",
        "checkpoint_epoch": int(best_ckpt.get("epoch", -1)),
        "rows": int(len(out)),
        "expected_rows": 954326,
        "rows_match": bool(len(out) == 954326),
        "label_counts": {
            "observed_positive": int((out["label"] == 1).sum()),
            "unobserved_candidate": int((out["label"] == 0).sum()),
        },
        "expected_label_counts": {
            "observed_positive": 3057,
            "unobserved_candidate": 951269,
        },
        "label_counts_match": bool(
            (out["label"] == 1).sum() == 3057 and (out["label"] == 0).sum() == 951269
        ),
        "probability_range": [float(out["probability"].min()), float(out["probability"].max())],
        "prob_in_01": bool(
            0.0 <= float(out["probability"].min()) and float(out["probability"].max()) <= 1.0
        ),
        "gate_ranges": {
            "gate_rnafm": [float(gate_rnafm.min()), float(gate_rnafm.max())],
            "gate_run": [float(gate_run.min()), float(gate_run.max())],
            "gate_pam": [float(gate_pam.min()), float(gate_pam.max())],
        },
        "gates_in_01": bool(
            0.0 <= float(gate_rnafm.min()) and float(gate_rnafm.max()) <= 1.0
            and 0.0 <= float(gate_run.min()) and float(gate_run.max()) <= 1.0
            and 0.0 <= float(gate_pam.min()) and float(gate_pam.max()) <= 1.0
        ),
        "gate_sum_max_error": float(np.abs(out["gate_sum"] - 1.0).max()),
        "gate_sum_within_tolerance": bool(np.abs(out["gate_sum"] - 1.0).max() <= 1e-5),
        "pam_coord_ok": bool(
            (out["PAM_original"] == out["off_seq"].astype(str).str.slice(20, 23)).all()
        ),
        "gate_argmax_counts": out["gate_argmax"].value_counts().to_dict(),
        "pam_family_counts": out["pam_family"].value_counts().to_dict(),
        "no_experiment_log_written": True,
    }

    # Alignment with existing test_predictions.csv
    existing_path = Path(args.validate_existing_predictions)
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        existing = existing.iloc[:n_exported]
        alignment = {
            "existing_rows": int(len(existing)),
            "sample_index_match": bool((existing["sample_index"].values == out["sample_index"].values).all()),
            "sgRNA_type_match": bool((existing["sgRNA_type"].values == out["sgRNA_type"].values).all()),
            "off_seq_match": bool((existing["off_seq"].values == out["off_seq"].values).all()),
            "label_match": bool((existing["label"].values == out["label"].values).all()),
        }
        prob_diff = np.abs(out["probability"].values - existing["probability"].values)
        alignment["probability_max_abs_diff"] = float(prob_diff.max())
        alignment["probability_within_tolerance"] = bool(prob_diff.max() <= 1e-5)
        validation["alignment_with_existing_predictions"] = alignment
    else:
        validation["alignment_with_existing_predictions"] = "skipped (file not found)"

    # Write validation JSON
    val_json_path = output_path.parent / "gate_export_validation.json"
    val_json_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[gate_export] validation written to {val_json_path}")

    # ---- 10. Print summary ----
    print("\n=== Gate Export Validation ===")
    print(f"  rows: {validation['rows']} (match={validation['rows_match']})")
    print(f"  labels: {validation['label_counts']} (match={validation['label_counts_match']})")
    print(f"  prob range: {validation['probability_range']} (in [0,1]={validation['prob_in_01']})")
    print(f"  gates in [0,1]: {validation['gates_in_01']}")
    print(f"  max|gate_sum - 1|: {validation['gate_sum_max_error']:.2e} (<=1e-5={validation['gate_sum_within_tolerance']})")
    print(f"  PAM coord ok: {validation['pam_coord_ok']}")
    print(f"  gate_argmax: {validation['gate_argmax_counts']}")
    print(f"  pam_family: {validation['pam_family_counts']}")
    if "alignment_with_existing_predictions" in validation and isinstance(validation["alignment_with_existing_predictions"], dict):
        a = validation["alignment_with_existing_predictions"]
        print(f"  alignment: sample_index={a['sample_index_match']} sgRNA={a['sgRNA_type_match']} off_seq={a['off_seq_match']} label={a['label_match']}")
        print(f"  prob max_abs_diff: {a['probability_max_abs_diff']:.2e} (<=1e-5={a['probability_within_tolerance']})")

    print("\n[gate_export] Done. Part 2 complete — no gate audit interpretation yet (Part 3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
