#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from server import _load_torch_predictor
from utils.config import load_config, resolve_dataset_files
from utils.io import iter_sequence_records


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _dataset_name_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    return stem[:-5] if stem.endswith("_9bit") else stem


def _checkpoint_path_for_dataset(dataset_name: str) -> Path:
    safe_name = dataset_name.replace("-", "_")
    slug = dataset_name.lower().replace("-", "_")
    return Path("artifacts") / f"full_{slug}_conmismatch9" / f"conmismatch9_c9_{safe_name}.pt"


def _predict_batch(predictor, pairs: list[tuple[str, str]]) -> np.ndarray:
    encoded = np.asarray(predictor.encoder.encode_batch(pairs), dtype=np.float32)
    x = torch.from_numpy(encoded).to(device=predictor.device, dtype=torch.float32)
    with torch.no_grad():
        context = torch.amp.autocast(device_type="cuda") if predictor.device.type == "cuda" else nullcontext()
        with context:
            logits = predictor.model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    return probs


def _risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "high"
    if prob >= 0.3:
        return "medium"
    return "low"


def _prediction_payload(
    dataset_name: str,
    item: dict[str, Any],
    prob: float,
    checkpoint: Path,
    model_backend: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "row": item["row"],
        "label": item["label"],
        "reads": item["reads"],
        "sgRNA": item["sgRNA"],
        "dna": item["dna"],
        "off_target_prob": prob,
        "risk_level": _risk_level(prob),
        "model_used": "conmismatch9",
        "encoder_used": "c9",
        "model_backend": model_backend,
        "checkpoint": str(checkpoint),
    }


def _metric_summary(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    if labels.size == 0:
        return {
            "auroc": None,
            "aupr": None,
            "acc": None,
            "mean_prob": None,
            "min_prob": None,
            "max_prob": None,
        }

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        auroc = 0.5
        aupr = 0.0
    else:
        auroc = float(roc_auc_score(labels, probs))
        aupr = float(average_precision_score(labels, probs))

    preds = (probs >= 0.5).astype(np.int64)
    return {
        "auroc": auroc,
        "aupr": aupr,
        "acc": float(accuracy_score(labels.astype(np.int64), preds)),
        "mean_prob": float(probs.mean()),
        "min_prob": float(probs.min()),
        "max_prob": float(probs.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export five-dataset ConMismatch9 predictions to JSON/CSV")
    parser.add_argument("--config", default="configs/c9_conmismatch9.yaml", help="Dataset config to resolve the five datasets")
    parser.add_argument("--device", default="cuda", help="torch device to use, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--batch-size", type=int, default=4096, help="Inference batch size")
    parser.add_argument("--output-root", default="output", help="Base output directory")
    parser.add_argument("--package-name", default=None, help="Optional package folder name")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_files = resolve_dataset_files(config)
    if not dataset_files:
        raise ValueError(f"no dataset_files resolved from config: {args.config}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    package_name = args.package_name or f"crispr_dualpred_five_dataset_{_timestamp()}"
    package_dir = output_root / package_name
    if package_dir.exists():
        raise FileExistsError(f"output package already exists: {package_dir}")
    package_dir.mkdir(parents=True)

    csv_path = package_dir / "predictions.csv"
    predictions_json_path = package_dir / "predictions.json"
    json_path = package_dir / "summary.json"
    manifest_path = package_dir / "manifest.json"
    note_path = package_dir / "PACKAGE_README.md"

    note_path.write_text(
        "\n".join(
            [
                "# CRISPR-DualPred export package",
                "",
                "This package contains ConMismatch9 predictions for the five datasets resolved from `configs/c9_conmismatch9.yaml`.",
                "",
                f"- CSV: {csv_path.name}",
                f"- Prediction JSON: {predictions_json_path.name}",
                f"- Summary JSON: {json_path.name}",
                "",
                "The rows were generated with the dataset-specific `.pt` checkpoints under `artifacts/full_*_conmismatch9/`.",
                "CSV rows include dataset name, row index, label, sequences, probability, and backend metadata.",
                "Prediction JSON contains the same per-row records as the CSV.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    datasets: list[dict[str, Any]] = []
    overall_labels: list[np.ndarray] = []
    overall_probs: list[np.ndarray] = []

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file, predictions_json_path.open(
        "w",
        encoding="utf-8",
    ) as predictions_json_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "dataset",
                "row",
                "label",
                "reads",
                "sgRNA",
                "dna",
                "off_target_prob",
                "risk_level",
                "model_used",
                "encoder_used",
                "model_backend",
                "checkpoint",
            ]
        )
        predictions_json_file.write("[\n")
        first_json_record = True

        for dataset_file in dataset_files:
            dataset_name = _dataset_name_from_path(dataset_file)
            checkpoint = _checkpoint_path_for_dataset(dataset_name)
            if not checkpoint.exists():
                raise FileNotFoundError(f"checkpoint not found for {dataset_name}: {checkpoint}")

            predictor = _load_torch_predictor("conmismatch9", checkpoint, args.device)

            rows = []
            labels_batches: list[np.ndarray] = []
            probs_batches: list[np.ndarray] = []
            pairs: list[tuple[str, str]] = []
            meta: list[dict[str, Any]] = []

            for record in iter_sequence_records(dataset_file, dataset=dataset_name):
                pairs.append((record.on_seq, record.off_seq))
                meta.append(
                    {
                        "row": int(record.index or 0) + 1,
                        "label": int(record.label),
                        "reads": record.reads,
                        "sgRNA": record.on_seq,
                        "dna": record.off_seq,
                    }
                )
                if len(pairs) >= args.batch_size:
                    probs = _predict_batch(predictor, pairs)
                    label_arr = np.asarray([item["label"] for item in meta], dtype=np.int64)
                    probs_arr = probs.astype(np.float32, copy=False)
                    labels_batches.append(label_arr)
                    probs_batches.append(probs_arr)
                    for item, prob in zip(meta, probs_arr.tolist(), strict=True):
                        payload = _prediction_payload(dataset_name, item, prob, checkpoint, predictor.model_backend)
                        writer.writerow(
                            [
                                payload["dataset"],
                                payload["row"],
                                payload["label"],
                                "" if payload["reads"] is None else payload["reads"],
                                payload["sgRNA"],
                                payload["dna"],
                                f"{prob:.10f}",
                                payload["risk_level"],
                                payload["model_used"],
                                payload["encoder_used"],
                                payload["model_backend"],
                                payload["checkpoint"],
                            ]
                        )
                        if not first_json_record:
                            predictions_json_file.write(",\n")
                        predictions_json_file.write(json.dumps(payload, ensure_ascii=False))
                        first_json_record = False
                    pairs.clear()
                    meta.clear()

            if pairs:
                probs = _predict_batch(predictor, pairs)
                label_arr = np.asarray([item["label"] for item in meta], dtype=np.int64)
                probs_arr = probs.astype(np.float32, copy=False)
                labels_batches.append(label_arr)
                probs_batches.append(probs_arr)
                for item, prob in zip(meta, probs_arr.tolist(), strict=True):
                    payload = _prediction_payload(dataset_name, item, prob, checkpoint, predictor.model_backend)
                    writer.writerow(
                        [
                            payload["dataset"],
                            payload["row"],
                            payload["label"],
                            "" if payload["reads"] is None else payload["reads"],
                            payload["sgRNA"],
                            payload["dna"],
                            f"{prob:.10f}",
                            payload["risk_level"],
                            payload["model_used"],
                            payload["encoder_used"],
                            payload["model_backend"],
                            payload["checkpoint"],
                        ]
                    )
                    if not first_json_record:
                        predictions_json_file.write(",\n")
                    predictions_json_file.write(json.dumps(payload, ensure_ascii=False))
                    first_json_record = False

            labels = np.concatenate(labels_batches) if labels_batches else np.asarray([], dtype=np.int64)
            probs = np.concatenate(probs_batches) if probs_batches else np.asarray([], dtype=np.float32)
            overall_labels.append(labels)
            overall_probs.append(probs)

            metrics = _metric_summary(labels, probs)
            datasets.append(
                {
                    "dataset": dataset_name,
                    "dataset_file": str(dataset_file),
                    "checkpoint": str(checkpoint),
                    "num_rows": int(labels.size),
                    "positives": int(labels.sum()) if labels.size else 0,
                    "negatives": int((labels == 0).sum()) if labels.size else 0,
                    "metrics": metrics,
                    "backend": predictor.model_backend,
                }
            )
            print(f"[done] {dataset_name}: {labels.size} rows")
        predictions_json_file.write("\n]\n")

    all_labels = np.concatenate(overall_labels) if overall_labels else np.asarray([], dtype=np.int64)
    all_probs = np.concatenate(overall_probs) if overall_probs else np.asarray([], dtype=np.float32)

    summary = {
        "project": "CRISPR-DualPred",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "device": args.device,
        "batch_size": args.batch_size,
        "datasets": datasets,
        "overall": {
            "num_rows": int(all_labels.size),
            "positives": int(all_labels.sum()) if all_labels.size else 0,
            "negatives": int((all_labels == 0).sum()) if all_labels.size else 0,
            "metrics": _metric_summary(all_labels, all_probs),
        },
        "files": {
            "csv": str(csv_path.name),
            "predictions_json": str(predictions_json_path.name),
            "summary_json": str(json_path.name),
            "manifest": str(manifest_path.name),
            "note": str(note_path.name),
        },
        "notes": [
            "CSV contains every record from the five datasets resolved by the config.",
            "Predictions are generated with the dataset-specific ConMismatch9 .pt checkpoints.",
            "The output package is intended for sharing and is written under output/.",
        ],
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "package": package_name,
        "csv": csv_path.name,
        "predictions_json": predictions_json_path.name,
        "summary_json": json_path.name,
        "note": note_path.name,
        "datasets": [item["dataset"] for item in datasets],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    archive_base = output_root / package_name
    shutil.make_archive(str(archive_base), "zip", root_dir=output_root, base_dir=package_name)

    print(package_dir)
    print(f"{archive_base}.zip")
    print(predictions_json_path)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
