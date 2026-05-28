#!/usr/bin/env python3
"""Eval-only: load best.pt and run test evaluation + prediction export."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import from train_bl0a_formal without running its main block
import importlib.util
spec = importlib.util.spec_from_file_location("train_bl0a_formal", ROOT / "scripts" / "train_bl0a_formal.py")
bl0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bl0)


def main() -> int:
    config_path = Path("configs/bl0b_on_bl5split.yaml")
    output_dir = Path("results/bl0b_on_bl5split")
    best_checkpoint = output_dir / "checkpoints" / "best.pt"

    if not best_checkpoint.exists():
        print(f"best.pt not found: {best_checkpoint}", file=sys.stderr)
        return 1

    config = bl0.load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist_info = {"distributed": False, "rank": 0, "local_rank": 0, "world_size": 1}

    rnafm_cfg = dict(config.get("rnafm", {}))
    model_cfg = bl0.make_model_config(config)
    model, alphabet = bl0.build_bl0_with_rnafm(
        checkpoint_path=rnafm_cfg.get("checkpoint_path"),
        allow_download=bool(rnafm_cfg.get("allow_download", False)),
        config=model_cfg,
    )
    model = model.to(device)

    # Load best.pt
    ckpt = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    best_epoch = ckpt.get("epoch", 0)
    print(f"Loaded best.pt from epoch {best_epoch}")

    # Data loading
    dataset_cfg = dict(config.get("dataset", {}))
    required_columns = tuple(dataset_cfg.get("required_columns", ("sgRNA_seq", "off_seq", "label", "sgRNA_type", "id")))
    df = bl0.load_cclmoff_dataframe(
        config["dataset"]["csv_path"],
        max_rows=dataset_cfg.get("max_rows"),
        required_columns=required_columns,
    )
    split_info = bl0.make_split(df, dict(config.get("split", {})), int(config.get("seed", 42)))
    split_indices = {name: split_info[name] for name in ("train", "val", "test")}

    precision = str(config.get("training", {}).get("precision", "fp32"))
    eval_batch_size = int(config.get("training", {}).get("eval_batch_size", 2048))
    replace_t_with_u = bool(dataset_cfg.get("replace_t_with_u", True))
    test_dataset = bl0.CCLMoffFrameDataset(
        df.iloc[split_indices["test"]].reset_index(drop=True),
        replace_t_with_u=replace_t_with_u,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=bl0.make_collate_fn(alphabet),
    )

    # Evaluate
    test_metrics = bl0.evaluate(bl0.unwrap_model(model), test_loader, device, precision, max_batches=None)
    bl0.report_metrics(test_metrics.get("auroc"), test_metrics.get("auprc"), config.get("split_mode", split_info["strategy"]))

    # Export predictions
    test_predictions_path = output_dir / "test_predictions.csv"
    probabilities = bl0.predict_probabilities(bl0.unwrap_model(model), test_loader, device, precision)
    bl0.write_test_predictions(test_predictions_path, df, split_indices["test"], probabilities)
    print(f"Exported test predictions to {test_predictions_path}")

    # Write summary
    summary = {
        "generated_at": bl0.utc_now(),
        "version": str(config.get("version", "BL0b-on-BL5split")),
        "status": "completed_eval_only",
        "config_path": str(config_path),
        "commit_hash": bl0.git_hash(),
        "device": str(device),
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "artifacts": {
            "best_checkpoint": str(best_checkpoint),
            "test_predictions": str(test_predictions_path),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
