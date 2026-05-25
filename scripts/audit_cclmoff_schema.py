from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CSV = Path("data/cclmoff/09212024_CCLMoff_dataset.csv")
DEFAULT_DOWNLOAD_LOG = Path("results/audits/cclmoff_download_log.txt")
DEFAULT_REPORT = Path("results/audits/cclmoff_schema_report.md")
DEFAULT_JSON = Path("results/audits/cclmoff_schema_report.json")
EXPECTED_FIELDS = ["sgRNA_seq", "off_seq", "label", "sgRNA_type", "id"]
MISSING_METADATA_FIELDS = [
    "detection_method",
    "source_study",
    "cell_line",
    "species",
    "bulge_type",
    "bulge_size",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_download_tail(path: Path, max_lines: int = 80) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def audit_csv(path: Path, sample_rows: int) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "file_size_bytes": 0,
            "columns": [],
            "sample_rows": 0,
            "sample_label_counts": {},
            "sample_sgrna_type_unique": None,
            "missing_expected_fields": EXPECTED_FIELDS,
            "missing_metadata_fields": MISSING_METADATA_FIELDS,
            "status": "cclmoff_csv_unavailable_using_local_data",
        }

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        label_counts: dict[str, int] = {}
        sgrna_types = set()
        row_count = 0
        for row in reader:
            row_count += 1
            if "label" in row:
                label = str(row.get("label", ""))
                label_counts[label] = label_counts.get(label, 0) + 1
            if "sgRNA_type" in row:
                sgrna_types.add(str(row.get("sgRNA_type", "")))
            if row_count >= sample_rows:
                break

    missing_expected = [field for field in EXPECTED_FIELDS if field not in columns]
    metadata_present = [field for field in MISSING_METADATA_FIELDS if field in columns]
    status = "schema_warning" if missing_expected else "metadata_limited"
    return {
        "exists": True,
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "columns": columns,
        "sample_rows": row_count,
        "sample_label_counts": label_counts,
        "sample_sgrna_type_unique": len(sgrna_types) if sgrna_types else 0,
        "missing_expected_fields": missing_expected,
        "metadata_present": metadata_present,
        "missing_metadata_fields": [field for field in MISSING_METADATA_FIELDS if field not in columns],
        "status": status,
    }


def write_reports(payload: dict[str, Any], report_path: Path, json_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_info = payload["csv"]
    lines = [
        "# CCLMoff Schema Report",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Expected CSV: `{csv_info['path']}`",
        f"- CSV exists: `{csv_info['exists']}`",
        f"- Status: `{csv_info['status']}`",
        "- Proxy note: current working Clash proxy is `127.0.0.1:7897`; direct Figshare article pages may still return an AWS WAF HTML challenge, so prefer Figshare API/direct file IDs.",
        "",
        "## Expected Public Fields",
        "",
        "| Field | Meaning |",
        "|:---|:---|",
        "| `sgRNA_seq` | sgRNA sequence |",
        "| `off_seq` | off-target candidate sequence; positions 21-23 contain NGG PAM |",
        "| `label` | `1` = observed_positive, `0` = unobserved_candidate |",
        "| `sgRNA_type` | sgRNA identifier, usable for Leave-One-sgRNA-Out splitting |",
        "| `id` | sample identifier |",
        "",
        "## Current CSV Audit",
        "",
        f"- File size bytes: `{csv_info['file_size_bytes']}`",
        f"- Columns: `{csv_info['columns']}`",
        f"- Missing expected fields: `{csv_info['missing_expected_fields']}`",
        f"- Missing metadata fields: `{csv_info['missing_metadata_fields']}`",
        f"- Sample rows read: `{csv_info['sample_rows']}`",
        f"- Sample label counts: `{csv_info['sample_label_counts']}`",
        f"- Sample sgRNA_type unique count: `{csv_info['sample_sgrna_type_unique']}`",
        "",
        "## Known Limitation",
        "",
        "The public CCLMoff CSV is known to lack detection-method, source-study, cell-line, species, and bulge metadata. "
        "Tier-aware training therefore requires an external `sgRNA_type -> detection_method/source_study/Tier` mapping table. "
        "Without that table, BL0 should use overall evaluation first and defer Tier-aware analysis.",
        "",
        "## Download Evidence Tail",
        "",
    ]
    if payload["download_log_tail"]:
        lines.extend([f"    {line}" for line in payload["download_log_tail"]])
    else:
        lines.append("No download log found.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CCLMoff CSV schema if available and document known metadata limits.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--download-log", type=Path, default=DEFAULT_DOWNLOAD_LOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--sample-rows", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "generated_at": utc_now(),
        "csv": audit_csv(args.csv, args.sample_rows),
        "download_log_tail": read_download_tail(args.download_log),
    }
    write_reports(payload, args.report, args.json)
    print(f"Wrote {args.report}")
    print(f"Wrote {args.json}")
    print(f"status={payload['csv']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
