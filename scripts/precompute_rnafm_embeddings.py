#!/usr/bin/env python3
"""Precompute RNA-FM CLS embeddings for all sequences in NPZ."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.rnafm import load_rnafm, normalize_pair_sequence, tokenize_rnafm_sequences


class SequenceDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=False)
        self.on_seqs = data["on_seq"]
        self.off_seqs = data["off_seq"]

    def __len__(self):
        return len(self.on_seqs)

    def __getitem__(self, idx):
        return str(self.on_seqs[idx]), str(self.off_seqs[idx])


def precompute(
    npz_path: str,
    output_path: str,
    checkpoint_path: str | None = None,
    batch_size: int = 2048,
    num_workers: int = 8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading RNA-FM on {device}...")
    rnafm_model, alphabet = load_rnafm(checkpoint_path, trust_local_checkpoint=True)
    rnafm_model = rnafm_model.to(device)
    rnafm_model.eval()
    repr_layer = 12

    dataset = SequenceDataset(npz_path)
    print(f"Total sequences: {len(dataset):,}")

    def collate_fn(batch):
        on_seqs, off_seqs = zip(*batch)
        sequences = [normalize_pair_sequence(on, off) for on, off in zip(on_seqs, off_seqs)]
        tokens = tokenize_rnafm_sequences(alphabet, sequences)
        return tokens

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    embeddings_list = []
    with torch.no_grad():
        for i, tokens in enumerate(loader):
            tokens = tokens.to(device)
            out = rnafm_model(tokens, repr_layers=[repr_layer], return_contacts=False)
            emb = out["representations"][repr_layer][:, 0, :]  # CLS, (B, 640)
            embeddings_list.append(emb.cpu().numpy())
            if (i + 1) % 10 == 0:
                print(f"  Processed {(i + 1) * batch_size:,} / {len(dataset):,}")

    embeddings = np.concatenate(embeddings_list, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")

    np.savez_compressed(output_path, rnafm_embeddings=embeddings)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/cclmoff/cclmoff_9bit.npz")
    parser.add_argument("--output", default="data/cclmoff/cclmoff_rnafm_embeddings.npz")
    parser.add_argument("--checkpoint", default="data/rnafm/checkpoints/RNA-FM_pretrained.pth")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    precompute(args.npz, args.output, args.checkpoint, args.batch_size, args.num_workers)
