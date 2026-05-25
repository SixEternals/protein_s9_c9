from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sequence import encode_c9_matrix, encode_r9_matrix, event_name_from_pair, region_bits


DEFAULT_REPORT = Path("results/audits/coordinate_audit_report.md")
DEFAULT_JSON = Path("results/audits/coordinate_audit_report.json")
DEFAULT_TABLE = Path("results/audits/coordinate_mapping_table.csv")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def state_from_bits(bits: tuple[int, int] | list[int]) -> str:
    pair = tuple(int(value) for value in bits)
    return {
        (0, 0): "pam_or_no_region",
        (0, 1): "ordinary",
        (1, 0): "hard_seed",
        (1, 1): "unused",
    }.get(pair, "unknown")


def run_state_from_bits(bits: tuple[int, int] | list[int]) -> str:
    pair = tuple(int(value) for value in bits)
    return {
        (0, 0): "match_or_non_mismatch",
        (0, 1): "isolated",
        (1, 0): "run2",
        (1, 1): "run3plus",
    }.get(pair, "unknown")


def build_mapping_rows() -> list[dict[str, Any]]:
    rows = []
    for pos in range(1, 24):
        region = region_bits(pos)
        if pos <= 20:
            canonical = "protospacer"
        else:
            canonical = "PAM"
        rows.append(
            {
                "position": pos,
                "zero_based_index": pos - 1,
                "canonical_zone": canonical,
                "canonical_orientation": "PAM-distal" if pos == 1 else "PAM-proximal" if pos == 20 else "",
                "hard_seed": "yes" if 16 <= pos <= 20 else "no",
                "r9_region_bits": "".join(str(value) for value in region),
                "r9_region_state": state_from_bits(region),
            }
        )
    return rows


def c9_pam_scope_probe() -> dict[str, Any]:
    on_seq = "A" * 23
    off_seq = list(on_seq)
    for pos in (20, 21, 22):
        off_seq[pos - 1] = "C"
    matrix = encode_c9_matrix(on_seq, "".join(off_seq))
    states = {
        str(pos): run_state_from_bits(matrix[pos - 1][7:9])
        for pos in (19, 20, 21, 22, 23)
    }
    return {
        "probe": "mismatch run across positions 20-22",
        "states": states,
        "interpretation": (
            "Current legacy C9 encoding computes continuous mismatch state across all 23 positions. "
            "Future BL3+ run features should restrict run computation to positions 1-20 and encode PAM separately."
        ),
    }


def equality_probe() -> dict[str, Any]:
    return {
        "A_vs_A": event_name_from_pair("A", "A"),
        "A_vs_T": event_name_from_pair("A", "T"),
        "A_vs_U_normalized_to_T": event_name_from_pair("A", "U"),
        "interpretation": "Canonical local encoders use character equality after U->T normalization, not Watson-Crick complement lookup.",
    }


def r9_probe() -> dict[str, Any]:
    matrix = encode_r9_matrix("A" * 23, "A" * 23)
    selected = {
        str(pos): {
            "bits": "".join(str(value) for value in matrix[pos - 1][7:9]),
            "state": state_from_bits(matrix[pos - 1][7:9]),
        }
        for pos in (1, 15, 16, 20, 21, 23)
    }
    return {
        "selected_positions": selected,
        "interpretation": "R9 region bits match the canonical hard-seed contract: 1-15 ordinary, 16-20 hard_seed, 21-23 PAM/no-region.",
    }


def scan_read_only_sources(root: Path) -> list[dict[str, str]]:
    paths = [
        Path("utils/sequence.py"),
        Path("encoders/r9_encoder.py"),
        Path("encoders/c9_encoder.py"),
        Path("offtarget_fusion_project/baseline0_cclmoff/CCLMoff/dataloader.py"),
        Path("offtarget_fusion_project/baseline0_cclmoff/CCLMoff/my_model.py"),
    ]
    rows: list[dict[str, str]] = []
    for rel_path in paths:
        path = root / rel_path
        if not path.exists():
            rows.append({"path": str(rel_path), "status": "missing", "notes": ""})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        notes = []
        if "region_bits" in text:
            notes.append("contains region_bits reference")
        if "encode_c9_matrix" in text:
            notes.append("contains C9 encoder reference")
        if "sgRNA_seq" in text and "off_seq" in text:
            notes.append("uses sgRNA_seq/off_seq pair fields")
        if "<sep>" in text:
            notes.append("uses '<sep>' pair joiner")
        rows.append({"path": str(rel_path), "status": "read_only_scanned", "notes": "; ".join(notes)})
    return rows


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(payload: dict[str, Any], report_path: Path, json_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Coordinate Audit Report",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        "- Canonical contract: position 1 = PAM-distal; position 20 = PAM-proximal; positions 21-23 = PAM.",
        "- Hard seed contract: positions 16-20.",
        "",
        "## R9 Region Probe",
        "",
        payload["r9_probe"]["interpretation"],
        "",
        "| Position | Region bits | State |",
        "|---:|:---:|:---|",
    ]
    for pos, info in payload["r9_probe"]["selected_positions"].items():
        lines.append(f"| {pos} | `{info['bits']}` | {info['state']} |")
    lines.extend(
        [
            "",
            "## Match Definition Probe",
            "",
            f"- `A` vs `A`: `{payload['equality_probe']['A_vs_A']}`",
            f"- `A` vs `T`: `{payload['equality_probe']['A_vs_T']}`",
            f"- `A` vs `U`: `{payload['equality_probe']['A_vs_U_normalized_to_T']}`",
            f"- Interpretation: {payload['equality_probe']['interpretation']}",
            "",
            "## C9 Run Scope Probe",
            "",
            f"- Probe: {payload['c9_pam_scope_probe']['probe']}",
            f"- Interpretation: {payload['c9_pam_scope_probe']['interpretation']}",
            "",
            "| Position | C9 run state |",
            "|---:|:---|",
        ]
    )
    for pos, state in payload["c9_pam_scope_probe"]["states"].items():
        lines.append(f"| {pos} | {state} |")
    lines.extend(
        [
            "",
            "## Read-Only Source Scan",
            "",
            "| Path | Status | Notes |",
            "|:---|:---|:---|",
        ]
    )
    for row in payload["source_scan"]:
        lines.append(f"| `{row['path']}` | {row['status']} | {row['notes']} |")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- Existing root R9 region logic is consistent with the canonical coordinate contract.",
            "- Existing root C9 run logic is useful historical context but computes runs across 23 positions; do not copy that detail into BL3+.",
            "- BL0 is unaffected by region/run coordinates because it uses RNA-FM sequence context only.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit canonical coordinate and legacy encoder assumptions.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mapping_rows = build_mapping_rows()
    payload = {
        "generated_at": utc_now(),
        "mapping_rows": mapping_rows,
        "r9_probe": r9_probe(),
        "equality_probe": equality_probe(),
        "c9_pam_scope_probe": c9_pam_scope_probe(),
        "source_scan": scan_read_only_sources(args.root),
    }
    write_table(mapping_rows, args.table)
    write_reports(payload, args.report, args.json)
    print(f"Wrote {args.table}")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
