#!/usr/bin/env python3
"""Audit formal_split_bl5_seed42.json for leakage, counts, and ratios.

Outputs:
    results/formal_split_bl5_seed42_audit.json
    results/formal_split_bl5_seed42_audit.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    split_path = Path("formal_split_bl5_seed42.json")
    if not split_path.exists():
        print(f"Missing {split_path}", file=sys.stderr)
        return 1

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    data = payload.get("data", {})
    splits = payload.get("splits", {})
    leakage = payload.get("leakage_check", {})

    # Build audit payload
    audit = {
        "audit_version": "1.0",
        "split_file_path": str(split_path),
        "data_file_path": data.get("cclmoff_csv"),
        "npz_path": data.get("npz_path"),
        "seed": payload.get("seed"),
        "split_algorithm": payload.get("split_logic"),
        "group_column": data.get("group_column"),
        "unique_groups_total": data.get("unique_groups"),
        "rows_total": data.get("rows"),
        "splits": {},
        "leakage_check": leakage,
        "leakage_passed": not any(leakage.values()),
    }

    for name in ("train", "val", "test"):
        sp = splits.get(name, {})
        audit["splits"][name] = {
            "sgRNA_type_count": sp.get("sgRNA_type_count"),
            "sgRNA_types": sp.get("sgRNA_types"),
            "samples": sp.get("samples"),
            "positive": sp.get("observed_positive"),
            "negative": sp.get("unobserved_candidate"),
            "positive_ratio": sp.get("positive_ratio"),
        }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "formal_split_bl5_seed42_audit.json"
    md_path = out_dir / "formal_split_bl5_seed42_audit.md"

    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Formal Split Audit Report",
        "",
        f"- **Split file**: `{audit['split_file_path']}`",
        f"- **Data file**: `{audit['data_file_path']}`",
        f"- **NPZ file**: `{audit['npz_path']}`",
        f"- **Seed**: {audit['seed']}",
        f"- **Algorithm**: {audit['split_algorithm']}",
        f"- **Group column**: {audit['group_column']}",
        f"- **Total unique groups**: {audit['unique_groups_total']}",
        f"- **Total rows**: {audit['rows_total']}",
        "",
        "## Leakage Check",
        "",
    ]
    for k, v in leakage.items():
        status = "✅ PASS" if not v else "❌ FAIL"
        md_lines.append(f"- **{k}**: {status} ({v if v else 'empty'})")
    md_lines.append(f"- **Overall leakage passed**: {'✅ YES' if audit['leakage_passed'] else '❌ NO'}")
    md_lines.append("")

    for name in ("train", "val", "test"):
        s = audit["splits"][name]
        md_lines.extend([
            f"## {name.upper()}",
            f"- sgRNA_type count: {s['sgRNA_type_count']}",
            f"- samples: {s['samples']}",
            f"- positive: {s['positive']}",
            f"- negative: {s['negative']}",
            f"- positive_ratio: {s['positive_ratio']:.6f}",
            "",
        ])

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
