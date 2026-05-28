#!/usr/bin/env python3
"""In-silico perturbation analysis for biological plausibility.

Selects representative test samples and perturbs PAM, seed mismatch,
and consecutive mismatch patterns. Reports probability shifts.

Usage:
    python scripts/in_silico_perturbation.py \
        --config configs/bl5_v4_pam.yaml \
        --checkpoint results/bl5_v4_pam/checkpoints/best.pt \
        --samples 20 \
        --output results/in_silico_perturbation_examples.csv
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Import BL5 training module for model construction and inference
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_bl5", ROOT / "scripts" / "train_bl5.py")
train_bl5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_bl5)


def perturb_pam(off_seq: str, new_pam: str) -> str:
    assert len(new_pam) == 3
    return off_seq[:-3] + new_pam


def perturb_seed_mismatch(off_seq: str, on_seq: str, n_mismatches: int) -> str:
    """Introduce mismatches in seed region (positions 16-20, 0-indexed 15-19)."""
    off_list = list(off_seq)
    seed_positions = list(range(15, 20))
    random.shuffle(seed_positions)
    for pos in seed_positions[:n_mismatches]:
        # Pick a different base
        original = off_list[pos]
        alternatives = [b for b in "ACGT" if b != original]
        off_list[pos] = random.choice(alternatives)
    return "".join(off_list)


def perturb_consecutive_mismatch(off_seq: str, on_seq: str, run_len: int, start_pos: int = 10) -> str:
    """Introduce a run of `run_len` consecutive mismatches starting at `start_pos`."""
    off_list = list(off_seq)
    for pos in range(start_pos, min(start_pos + run_len, len(off_seq))):
        original = off_list[pos]
        alternatives = [b for b in "ACGT" if b != original]
        off_list[pos] = random.choice(alternatives)
    return "".join(off_list)


def predict_batch(model, alphabet, samples: list[dict], device: torch.device) -> np.ndarray:
    """Run BL5 model inference on a list of (on_seq, off_seq) dicts."""
    from utils.rnafm import normalize_pair_sequence, tokenize_rnafm_sequences
    from encoders.learnable_run_encoder import encode_base_pair_indices
    from encoders.pam_encoder import encode_pam_onehot

    sequences = [normalize_pair_sequence(s["on_seq"], s["off_seq"]) for s in samples]
    tokens = tokenize_rnafm_sequences(alphabet, sequences)
    run_input = encode_base_pair_indices([s["on_seq"] for s in samples], [s["off_seq"] for s in samples])
    pam_input = encode_pam_onehot([s["off_seq"] for s in samples])
    seed_weights = torch.empty(0, dtype=torch.float32)

    tokens = tokens.to(device)
    run_input = run_input.to(device)
    seed_weights = seed_weights.to(device)
    pam_input = pam_input.to(device)

    with torch.no_grad():
        logits = model(tokens, run_input, seed_weights, pam_input)
    return torch.sigmoid(logits.squeeze(-1)).cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bl5_v4_pam.yaml")
    parser.add_argument("--checkpoint", default="results/bl5_v4_pam/checkpoints/best.pt")
    parser.add_argument("--csv", default="data/cclmoff/09212024_CCLMoff_dataset.csv")
    parser.add_argument("--split", default="formal_split_bl5_seed42.json")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", default="results/in_silico_perturbation_examples.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = train_bl5.load_config(args.config)

    # Load model
    rnafm_cfg = dict(config.get("rnafm", {}))
    rnafm_model, alphabet = train_bl5.load_rnafm(rnafm_cfg.get("checkpoint_path"), trust_local_checkpoint=True)
    rnafm_model = rnafm_model.to(device)
    model = train_bl5.BL5RunOnlyDynamicFusion(rnafm_model=rnafm_model, alphabet=alphabet, model_cfg=config["model"])
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test data
    import json
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    test_groups = set(split["splits"]["test"]["sgRNA_types"])
    df = pd.read_csv(args.csv, usecols=["sgRNA_seq", "off_seq", "label", "sgRNA_type", "Direction"])
    df = df[df["sgRNA_type"].astype(str).isin(test_groups)].reset_index(drop=True)

    # Stratified sample: pick from positive and negative, NGG and non-NGG
    df["PAM"] = df["off_seq"].astype(str).str[-3:]
    df["is_NGG"] = df["PAM"].str[1:3] == "GG"

    selected = []
    for (is_ngg, label), group in df.groupby(["is_NGG", "label"]):
        n = min(len(group), args.samples // 4 + 2)
        selected.append(group.sample(n, random_state=args.seed))
    sample_df = pd.concat(selected).reset_index(drop=True)

    rows = []
    for _, row in sample_df.iterrows():
        base = {
            "original_on_seq": row["sgRNA_seq"],
            "original_off_seq": row["off_seq"],
            "PAM": row["PAM"],
            "label": row["label"],
            "is_NGG": row["is_NGG"],
            "Direction": row.get("Direction", ""),
        }

        # Base prediction
        probs = predict_batch(model, alphabet, [{"on_seq": row["sgRNA_seq"], "off_seq": row["off_seq"]}], device)
        base_prob = float(probs[0])
        base["original_prob"] = base_prob

        # PAM perturbations
        for new_pam in ["AGG", "TGG", "GGG", "CGG", "NAG", "NGA", "NCG"]:
            perturbed_off = perturb_pam(row["off_seq"], new_pam)
            p = float(predict_batch(model, alphabet, [{"on_seq": row["sgRNA_seq"], "off_seq": perturbed_off}], device)[0])
            r = dict(base)
            r["perturbation_type"] = f"PAM_to_{new_pam}"
            r["perturbed_off_seq"] = perturbed_off
            r["perturbed_prob"] = p
            r["delta_prob"] = p - base_prob
            rows.append(r)

        # Seed mismatch perturbations
        for n_mm in [1, 2, 3]:
            perturbed_off = perturb_seed_mismatch(row["off_seq"], row["sgRNA_seq"], n_mm)
            p = float(predict_batch(model, alphabet, [{"on_seq": row["sgRNA_seq"], "off_seq": perturbed_off}], device)[0])
            r = dict(base)
            r["perturbation_type"] = f"seed_{n_mm}mm"
            r["perturbed_off_seq"] = perturbed_off
            r["perturbed_prob"] = p
            r["delta_prob"] = p - base_prob
            rows.append(r)

        # Consecutive mismatch perturbations
        for run_len in [2, 3]:
            for start in [5, 10, 15]:
                perturbed_off = perturb_consecutive_mismatch(row["off_seq"], row["sgRNA_seq"], run_len, start)
                p = float(predict_batch(model, alphabet, [{"on_seq": row["sgRNA_seq"], "off_seq": perturbed_off}], device)[0])
                r = dict(base)
                r["perturbation_type"] = f"consecutive_run{run_len}_pos{start}"
                r["perturbed_off_seq"] = perturbed_off
                r["perturbed_prob"] = p
                r["delta_prob"] = p - base_prob
                rows.append(r)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Summary report
    md = ["# In-silico Perturbation Report", ""]
    md.append(f"- Base samples evaluated: {len(sample_df)}")
    md.append(f"- Total perturbations: {len(out_df)}")
    md.append("")

    # Aggregate by perturbation type
    agg = out_df.groupby("perturbation_type")["delta_prob"].agg(["mean", "median", "std", "count"]).reset_index()
    md.append("## Mean Delta Probability by Perturbation Type")
    md.append(agg.to_markdown(index=False))
    md.append("")

    md_path = out_path.with_suffix(".md")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
