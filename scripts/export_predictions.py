"""Export per-sample predictions from a BL0 checkpoint for visualization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.cclmoff_dataset import CCLMoffFrameDataset, load_cclmoff_dataframe
from models.bl0_cclmoff import BL0CCLMoffConfig, build_bl0_with_rnafm
from scripts.train_bl0a_formal import make_collate_fn, make_split
from utils.config import load_config
from utils.rnafm import tokenize_rnafm_sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="output CSV path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    csv_path = config["dataset"]["csv_path"]
    replace_t_with_u = config["dataset"].get("replace_t_with_u", True)
    df = load_cclmoff_dataframe(csv_path)

    # Reconstruct split (must use same seed and strategy)
    split_cfg = config.get("split", {})
    seed = config.get("seed", 42)
    split_indices = make_split(df, split_cfg, seed)

    split_name = args.split
    indices = split_indices[split_name]
    subset_df = df.iloc[indices].reset_index(drop=True)

    dataset = CCLMoffFrameDataset(subset_df, replace_t_with_u=replace_t_with_u)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Load model
    model_cfg = BL0CCLMoffConfig(**config.get("model_config", {}))
    model, alphabet = build_bl0_with_rnafm(
        checkpoint_path=config["rnafm"]["checkpoint_path"],
        allow_download=config["rnafm"].get("allow_download", False),
        config=model_cfg,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "head_state_dict" in ckpt:
        model.head.load_state_dict(ckpt["head_state_dict"])
    else:
        raise ValueError(f"checkpoint has no recognizable state dict; keys: {list(ckpt.keys())}")
    model.to(device)

    # Rebuild loader with proper collate_fn
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=make_collate_fn(alphabet),
    )

    # Run inference and collect per-sample predictions
    model.eval()
    labels_all = []
    probs_all = []
    precision = config.get("training", {}).get("precision", "bf16")
    autocast_enabled = device.type == "cuda" and precision in {"fp16", "bf16"}
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    with torch.no_grad():
        for tokens, labels in loader:
            tokens = tokens.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                logits = model(tokens)
            probs = torch.sigmoid(logits.float()).detach().cpu().numpy().flatten()
            labels_all.append(labels.detach().cpu().numpy().flatten())
            probs_all.append(probs)

    labels_np = np.concatenate(labels_all)
    probs_np = np.concatenate(probs_all)

    # Save
    out_df = pd.DataFrame({
        "label": labels_np.astype(int),
        "predicted_probability": probs_np.astype(np.float32),
    })
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(out_df)} rows to {args.output}")
    print(f"  label=1 (observed_positive): {(out_df['label'] == 1).sum()}")
    print(f"  label=0 (unobserved_candidate): {(out_df['label'] == 0).sum()}")


if __name__ == "__main__":
    main()
