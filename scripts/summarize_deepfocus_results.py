#!/usr/bin/env python3
"""Summarize DeepFocus multiseed experiment results."""

import json
import sys
from pathlib import Path


def extract_metrics(summary_path: Path) -> dict | None:
    try:
        data = json.loads(summary_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return {
        "best_val_aupr": data.get("best_val_aupr"),
        "test_auroc": data.get("test_metrics", {}).get("auroc"),
        "test_aupr": data.get("test_metrics", {}).get("aupr"),
        "test_acc": data.get("test_metrics", {}).get("acc"),
        "best_epoch": data.get("best_epoch"),
    }


def main() -> None:
    run_base = Path("runs")
    datasets = ["K562", "SITE", "Tasi", "CHANGE-seq", "GUIDE-seq"]
    seeds = [42, 43, 44]

    rows = []
    for dataset in datasets:
        slug = dataset.lower().replace("-", "_")
        safe_name = dataset.replace("-", "_")
        base = run_base / f"full_upgrade_{slug}_deepfocus"
        for seed in seeds:
            for mode in ["inception_only", "full"]:
                summary = base / f"seed_{seed}" / mode / "train_summaries" / f"deepfocus_r9_{safe_name}.json"
                metrics = extract_metrics(summary)
                if metrics:
                    rows.append({
                        "dataset": dataset,
                        "seed": seed,
                        "mode": mode,
                        **metrics,
                    })

    if not rows:
        print("No results found yet.")
        sys.exit(0)

    # Print table
    print(f"{'Dataset':<12} {'Seed':>4} {'Mode':<16} {'Val_AUPR':>10} {'Test_AUPR':>10} {'Test_AUROC':>10} {'Test_Acc':>10} {'Epoch':>5}")
    print("-" * 90)
    for r in rows:
        print(f"{r['dataset']:<12} {r['seed']:>4} {r['mode']:<16} {r['best_val_aupr']:>10.4f} {r['test_aupr']:>10.4f} {r['test_auroc']:>10.4f} {r['test_acc']:>10.4f} {r['best_epoch']:>5}")

    # Grouped summary
    print("\n" + "=" * 90)
    print("Grouped by dataset + mode (mean ± std)")
    print("=" * 90)
    from collections import defaultdict
    import statistics

    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["mode"])].append(r["test_aupr"])

    for (dataset, mode), values in sorted(groups.items()):
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{dataset:<12} {mode:<16} test_aupr = {mean:.4f} ± {std:.4f} (n={len(values)})")


if __name__ == "__main__":
    main()
