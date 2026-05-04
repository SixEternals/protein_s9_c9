from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.conmismatch9 import ConMismatch9Model
from models.deepfocus import DeepFocusModel
from utils.config import load_config, resolve_dataset_files
from utils.io import collect_balanced_records
from utils.training import evaluate_model, fit_linear_model, serialize_history, train_test_split_71515


def _build_model(name: str):
    name = name.lower()
    if name == "deepfocus":
        return DeepFocusModel()
    if name == "conmismatch9":
        return ConMismatch9Model()
    raise ValueError(f"unknown model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRISPR-DualPred lightweight baseline models")
    parser.add_argument("--config", required=True, help="Path to a JSON-compatible YAML config")
    parser.add_argument("--output", default=None, help="Optional path to write the training summary")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = str(config.get("model", "deepfocus"))
    model = _build_model(model_name)

    dataset_files = resolve_dataset_files(config)
    if not dataset_files:
        raise ValueError("config must define dataset_files")

    sampling = config.get("sampling", {})
    positive_cap = sampling.get("positive_cap", 5000)
    negative_cap = sampling.get("negative_cap", 15000)
    seed = int(config.get("seed", 42))

    records = []
    for dataset_file in dataset_files:
        dataset_name = Path(dataset_file).stem
        records.extend(
            collect_balanced_records(
                dataset_file,
                dataset=dataset_name,
                positive_cap=positive_cap,
                negative_cap=negative_cap,
                seed=seed,
            )
        )

    train_records, val_records, test_records = train_test_split_71515(records, seed=seed)
    training = config.get("training", {})
    model, history = fit_linear_model(
        model,
        train_records,
        val_records,
        epochs=int(training.get("epochs", 8)),
        learning_rate=float(training.get("learning_rate", 0.05)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        patience=int(training.get("patience", 5)),
        seed=seed,
        pos_weight=training.get("pos_weight"),
    )

    test_metrics = evaluate_model(model, test_records)
    weights_path = config.get("weights_path")
    if weights_path:
        model.save(weights_path)

    summary = {
        "model": model_name,
        "encoder": config.get("encoder"),
        "dataset_files": dataset_files,
        "train_size": len(train_records),
        "val_size": len(val_records),
        "test_size": len(test_records),
        "test_metrics": test_metrics,
        "history": serialize_history(history),
        "weights_path": weights_path,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

