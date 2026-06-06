#!/usr/bin/env python3
"""Load a BL5 checkpoint and generate test_predictions.csv for ablation analysis."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from train_bl5 directly to reuse dataset logic
from scripts.train_bl5 import (
    BL5Arrays,
    BL5Dataset,
    make_live_collate,
    make_split,
)
from models.bl5_dynamic_fusion import BL5RunOnlyDynamicFusion
from utils.guardrails import check_model_config
from utils.rnafm import load_rnafm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--output", required=True, help="Path to output CSV")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    check_model_config(config)

    model_cfg = config.get("model", config)
    data_cfg = config.get("data", config)
    training_cfg = config.get("training", config)
    rnafm_cfg = config.get("rnafm", {})

    use_rnafm = bool(model_cfg.get("use_rnafm", True))
    use_run = bool(model_cfg.get("use_run", True))
    use_learnable_run = bool(model_cfg.get("use_learnable_run", False))
    use_pam_encoder = bool(model_cfg.get("use_pam_encoder", False))
    batch_size = int(training_cfg.get("batch_size", 256))
    num_workers = int(training_cfg.get("num_workers", 4))
    seed = int(config.get("seed", 42))

    # Load arrays
    arrays = BL5Arrays(config)

    # Load group labels for split
    csv_path = data_cfg.get("cclmoff_csv")
    df_meta = pd.read_csv(csv_path, usecols=[data_cfg.get("group_column", "sgRNA_type")])
    group_labels = df_meta[data_cfg.get("group_column", "sgRNA_type")].values

    # Make split
    split_indices = make_split(
        arrays.labels.astype(np.int64),
        seed,
        group_labels,
    )
    test_indices = split_indices["test"]

    # Load RNA-FM
    alphabet = None
    rnafm_model = None
    if use_rnafm:
        ckpt = rnafm_cfg.get("checkpoint_path", "data/rnafm/checkpoints/RNA-FM_pretrained.pth")
        rnafm_model, alphabet = load_rnafm(ckpt)
        rnafm_model.eval()

    # Dataset
    test_dataset = BL5Dataset(arrays, test_indices, pam_shuffle_indices=None)

    collate_fn = make_live_collate(
        alphabet,
        use_rnafm=use_rnafm,
        use_run=use_run,
        use_learnable_run=use_learnable_run,
        use_pam_encoder=use_pam_encoder,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BL5RunOnlyDynamicFusion(rnafm_model=rnafm_model, config=config)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    # Predict
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            tokens, run_features, seed_weights, pam_features, labels = batch
            tokens = tokens.to(device)
            run_features = run_features.to(device) if use_run else torch.empty(0, device=device)
            seed_weights = seed_weights.to(device) if use_run else torch.empty(0, device=device)
            pam_features = pam_features.to(device) if use_pam_encoder else None

            logits = model(tokens, run_features, seed_weights, pam_features)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    probs_np = np.concatenate(all_probs).astype(np.float64)

    # Write CSV
    df = pd.read_csv(csv_path, usecols=["sgRNA_seq", "off_seq", "sgRNA_type", "label"])
    df_test = df.iloc[test_indices].copy()
    df_test["probability"] = probs_np
    df_test["sample_index"] = test_indices
    df_test.to_csv(args.output, index=False)
    print(f"Saved {len(df_test)} predictions to {args.output}")


if __name__ == "__main__":
    main()
