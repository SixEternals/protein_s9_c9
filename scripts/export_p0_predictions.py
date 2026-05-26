"""Export P0 (R9/C9 weighted average) predictions on CCLMoff test set."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.cclmoff_dataset import load_cclmoff_dataframe
from encoders.c9_encoder import C9Encoder
from encoders.r9_encoder import R9Encoder
from models.bl0_cclmoff import build_bl0_with_rnafm
from scripts.train_bl0a_formal import make_split
from train import _checkpoint_payload, _instantiate_model_from_payload
from utils.config import load_config


def encode_batch(encoder, sequences_on, sequences_off):
    encoded = np.empty((len(sequences_on), 23, 9), dtype=np.uint8)
    for idx, (on, off) in enumerate(zip(sequences_on, sequences_off)):
        encoded[idx] = np.asarray(encoder.encode_pair(str(on), str(off)), dtype=np.uint8)
    return encoded


@torch.no_grad()
def predict(model, features, batch_size, device):
    model.eval()
    model.to(device)
    probs = []
    for i in range(0, len(features), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(features[i:i+batch_size], dtype=np.float32)).to(device)
        logits = model(batch)
        probs.append(torch.sigmoid(logits).cpu().numpy().flatten())
    return np.concatenate(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bl0b-config", default="configs/bl0b_finetune.yaml")
    parser.add_argument("--r9-checkpoint", default="artifacts/full_upgrade_guide_seq_deepfocus/seed_43/full/deepfocus_r9_GUIDE_seq.pt")
    parser.add_argument("--c9-checkpoint", default="artifacts/full_upgrade_guide_seq_conmismatch9/seed_43/full/conmismatch9_c9_GUIDE_seq.pt")
    parser.add_argument("--weight-r9", type=float, default=0.16, help="R9 weight in weighted average")
    parser.add_argument("--output", default="results/bl0b_finetune/p0_test_predictions.csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bl0b_config = load_config(args.bl0b_config)

    # Load CCLMoff data and reconstruct same test split as BL0b
    csv_path = bl0b_config["dataset"]["csv_path"]
    df = load_cclmoff_dataframe(csv_path)
    split_cfg = bl0b_config.get("split", {})
    seed = bl0b_config.get("seed", 42)
    split_indices = make_split(df, split_cfg, seed)
    test_df = df.iloc[split_indices["test"]].reset_index(drop=True)

    on_seq = test_df["sgRNA_seq"].to_numpy()
    off_seq = test_df["off_seq"].to_numpy()
    labels = test_df["label"].to_numpy().astype(np.float32)

    # Encode
    r9_features = encode_batch(R9Encoder(), on_seq, off_seq)
    c9_features = encode_batch(C9Encoder(), on_seq, off_seq)

    # Load R9/DeepFocus
    r9_payload = _checkpoint_payload(Path(args.r9_checkpoint))
    _, _, r9_model = _instantiate_model_from_payload(r9_payload)
    r9_probs = predict(r9_model, r9_features, args.batch_size, device)

    # Load C9/ConMismatch9
    c9_payload = _checkpoint_payload(Path(args.c9_checkpoint))
    _, _, c9_model = _instantiate_model_from_payload(c9_payload)
    c9_probs = predict(c9_model, c9_features, args.batch_size, device)

    # Weighted average
    weight_r9 = args.weight_r9
    weight_c9 = 1.0 - weight_r9
    fused_probs = weight_r9 * r9_probs + weight_c9 * c9_probs

    out_df = pd.DataFrame({
        "label": labels.astype(int),
        "p0_predicted_probability": fused_probs.astype(np.float32),
    })
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(out_df)} rows to {args.output}")
    print(f"  weight_r9={weight_r9:.2f}, weight_c9={weight_c9:.2f}")
    print(f"  label=1: {(out_df['label']==1).sum()}, label=0: {(out_df['label']==0).sum()}")


if __name__ == "__main__":
    main()
