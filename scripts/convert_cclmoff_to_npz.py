#!/usr/bin/env python3
"""Convert CCLMoff CSV to NPZ with pre-encoded Region + Run features."""

import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoders.run_encoder import RegionEncoder, RunEncoder


def _encode_chunk(args):
    """Worker function: encode a chunk of (on_seq, off_seq) pairs."""
    chunk_idx, on_seqs, off_seqs, weight_mode, tau = args
    region_enc = RegionEncoder(length=20)
    run_enc = RunEncoder(length=20, tau=tau, weight_mode=weight_mode)
    pairs = [(str(on), str(off)) for on, off in zip(on_seqs, off_seqs)]
    region = region_enc.encode_batch(pairs)
    run = run_enc.encode_batch(pairs)
    return chunk_idx, region, run


def convert_cclmoff_to_npz(
    csv_path: str,
    output_path: str,
    weight_mode: str = "soft",
    tau: float = 4.0,
    num_workers: int = 8,
):
    print(f"Reading {csv_path}...")
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=500_000, on_bad_lines="skip"):
        cols = ["sgRNA_seq", "off_seq", "read", "label"]
        chunk = chunk[cols].copy()
        chunk.rename(columns={"sgRNA_seq": "on_seq"}, inplace=True)
        chunks.append(chunk)
        print(f"  Read {len(chunk)} rows, total: {sum(len(c) for c in chunks)}")

    df = pd.concat(chunks, ignore_index=True)
    print(f"Total rows: {len(df)}")
    df.dropna(subset=["on_seq", "off_seq", "label"], inplace=True)
    print(f"After NA drop: {len(df)}")

    n = len(df)
    X = np.zeros((n, 23, 9), dtype=np.float32)
    y = df["label"].values.astype(np.float32)
    reads = df["read"].fillna(0).values.astype(np.float32)
    on_seq = np.array(df["on_seq"].values, dtype="U")
    off_seq = np.array(df["off_seq"].values, dtype="U")

    # Parallel pre-encoding
    print(f"Pre-encoding Region + Run features with {num_workers} workers...")
    chunk_size = (n + num_workers - 1) // num_workers
    tasks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        tasks.append((
            i,
            on_seq[start:end],
            off_seq[start:end],
            weight_mode,
            tau,
        ))

    with mp.Pool(num_workers) as pool:
        results = pool.map(_encode_chunk, tasks)

    # Merge results in order
    results.sort(key=lambda x: x[0])
    region_features = np.concatenate([r[1] for r in results], axis=0)
    run_features = np.concatenate([r[2] for r in results], axis=0)
    seed_weights = RunEncoder(length=20, tau=tau, weight_mode=weight_mode).seed_weights()

    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        reads=reads,
        on_seq=on_seq,
        off_seq=off_seq,
        region_features=region_features,
        run_features=run_features,
        seed_weights=seed_weights,
    )
    print("Done.")
    print(f"  region_features: {region_features.shape}")
    print(f"  run_features:    {run_features.shape}")
    print(f"  seed_weights:    {seed_weights.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/cclmoff/09212024_CCLMoff_dataset.csv")
    parser.add_argument("--output", default="data/cclmoff/cclmoff_9bit.npz")
    parser.add_argument("--weight-mode", default="soft")
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    convert_cclmoff_to_npz(
        args.csv, args.output, args.weight_mode, args.tau, args.num_workers
    )
