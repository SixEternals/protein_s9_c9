from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LOCAL_ROOT = Path("/data/zwf/project/zhb/data")
DEFAULT_CCLMOFF_ROOT = Path("data/cclmoff")
DEFAULT_TIER_CSV = Path("data/tier_labels.csv")
DEFAULT_REPORT = Path("results/audits/schema_audit_report.md")
DEFAULT_JSON = Path("results/audits/schema_audit_report.json")

TIER_BY_METHOD = {
    "DISCOVER-seq": (1, 1.0),
    "DISCOVER-seq+": (1, 1.0),
    "GUIDE-seq": (1, 1.0),
    "HTGTS": (1, 1.0),
    "IDLV": (2, 0.7),
    "BLESS": (2, 0.7),
    "BLISS": (2, 0.7),
    "PEM-seq": (2, 0.7),
    "VIVO": (2, 0.7),
    "CIRCLE-seq": (3, 0.4),
    "CHANGE-seq": (3, 0.4),
    "Digenome-seq": (3, 0.4),
    "SITE-seq": (3, 0.4),
}

LOCAL_METHOD_BY_DATASET = {
    "CHANGE-seq": "CHANGE-seq",
    "GUIDE-seq": "GUIDE-seq",
    "SITE": "SITE-seq",
}

REQUIRED_PAIR_FIELDS = {"on_seq", "off_seq", "y"}
CCLMOFF_SCHEMA_FIELDS = {"detection_method", "source_study", "bulge_type", "bulge_size"}
CCLMOFF_PUBLIC_FIELDS = {"sgRNA_seq", "off_seq", "label", "sgRNA_type", "id"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_method(value: str | None) -> str:
    if not value:
        return "unknown"
    value = value.strip()
    aliases = {
        "SITE": "SITE-seq",
        "CHANGE": "CHANGE-seq",
        "GUIDE": "GUIDE-seq",
        "CIRCLE": "CIRCLE-seq",
    }
    return aliases.get(value, value)


def tier_for_method(method: str) -> tuple[str, str]:
    method = normalize_method(method)
    if method in TIER_BY_METHOD:
        tier, weight = TIER_BY_METHOD[method]
        return str(tier), f"{weight:.1f}"
    return "unknown", ""


def label_counts(data: np.lib.npyio.NpzFile) -> tuple[int | None, int | None]:
    if "y" not in data.files:
        return None, None
    labels = np.asarray(data["y"])
    observed = int((labels == 1).sum())
    unobserved = int((labels == 0).sum())
    return observed, unobserved


def inspect_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        row_count = int(np.asarray(data[keys[0]]).shape[0]) if keys else 0
        observed, unobserved = label_counts(data)
        shapes = {key: list(np.asarray(data[key]).shape) for key in keys}
        missing = sorted(REQUIRED_PAIR_FIELDS - set(keys))
    return {
        "path": str(path),
        "kind": "npz",
        "keys": keys,
        "shapes": shapes,
        "row_count": row_count,
        "observed_positive_count": observed,
        "unobserved_candidate_count": unobserved,
        "missing_required_fields": missing,
    }


def inspect_delimited_header(path: Path) -> dict[str, Any]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, [])
        sample_rows = []
        for _ in range(1000):
            try:
                sample_rows.append(next(reader))
            except StopIteration:
                break
    sample_labels = []
    if "label" in header:
        label_idx = header.index("label")
        for row in sample_rows:
            if len(row) > label_idx:
                sample_labels.append(row[label_idx])
    return {
        "path": str(path),
        "kind": path.suffix.lower().lstrip("."),
        "keys": header,
        "file_size_bytes": path.stat().st_size,
        "row_count": None,
        "sampled_rows": len(sample_rows),
        "observed_positive_count": None,
        "unobserved_candidate_count": None,
        "sample_label_counts": {value: sample_labels.count(value) for value in sorted(set(sample_labels))},
        "missing_cclmoff_fields": sorted(CCLMOFF_SCHEMA_FIELDS - set(header)),
        "missing_public_fields": sorted(CCLMOFF_PUBLIC_FIELDS - set(header)),
    }


def local_rows(local_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(local_root.glob("*/*_9bit.npz")):
        dataset_name = path.parent.name
        method = normalize_method(LOCAL_METHOD_BY_DATASET.get(dataset_name, dataset_name))
        tier, weight = tier_for_method(method)
        info = inspect_npz(path)
        status = "ok" if not info["missing_required_fields"] else "schema_warning"
        if tier == "unknown":
            status = "tier_unknown" if status == "ok" else status
        rows.append(
            {
                "dataset_name": dataset_name,
                "source": "local_npz",
                "local_path": str(path),
                "detection_method": method,
                "tier": tier,
                "tier_weight": weight,
                "total_samples": info["row_count"],
                "observed_positive_count": info["observed_positive_count"],
                "unobserved_candidate_count": info["unobserved_candidate_count"],
                "status": status,
                "notes": "; ".join(info["missing_required_fields"]) if info["missing_required_fields"] else "",
                "schema": info,
            }
        )
    return rows


def cclmoff_rows(cclmoff_root: Path) -> list[dict[str, Any]]:
    if not cclmoff_root.exists():
        return [
            {
                "dataset_name": "CCLMoff",
                "source": "cclmoff_release_or_figshare",
                "local_path": str(cclmoff_root),
                "detection_method": "unknown",
                "tier": "unknown",
                "tier_weight": "",
                "total_samples": "",
                "observed_positive_count": "",
                "unobserved_candidate_count": "",
                "status": "missing",
                "notes": "No local CCLMoff dataset found. GitHub releases are empty; retry Figshare with proxy and browser User-Agent.",
                "schema": {},
            }
        ]

    candidates = sorted(
        [
            path
            for pattern in ("*.csv", "*.tsv", "*.tab", "*.npz")
            for path in cclmoff_root.rglob(pattern)
            if path.is_file()
        ]
    )
    if not candidates:
        return [
            {
                "dataset_name": "CCLMoff",
                "source": "cclmoff_local",
                "local_path": str(cclmoff_root),
                "detection_method": "unknown",
                "tier": "unknown",
                "tier_weight": "",
                "total_samples": "",
                "observed_positive_count": "",
                "unobserved_candidate_count": "",
                "status": "empty",
                "notes": "Directory exists but no csv/tsv/npz files were discovered.",
                "schema": {},
            }
        ]

    rows: list[dict[str, Any]] = []
    for path in candidates:
        info = inspect_npz(path) if path.suffix.lower() == ".npz" else inspect_delimited_header(path)
        keys = set(info.get("keys", []))
        missing_metadata = sorted(CCLMOFF_SCHEMA_FIELDS - keys)
        missing_public = sorted(CCLMOFF_PUBLIC_FIELDS - keys)
        status = "metadata_limited" if not missing_public else "schema_warning"
        notes = []
        if missing_public:
            notes.append(f"missing public fields: {', '.join(missing_public)}")
        if missing_metadata:
            notes.append(f"missing metadata fields: {', '.join(missing_metadata)}")
        rows.append(
            {
                "dataset_name": f"CCLMoff:{path.stem}",
                "source": "cclmoff_local",
                "local_path": str(path),
                "detection_method": "unknown",
                "tier": "unknown",
                "tier_weight": "",
                "total_samples": info.get("row_count", ""),
                "observed_positive_count": info.get("observed_positive_count", ""),
                "unobserved_candidate_count": info.get("unobserved_candidate_count", ""),
                "status": status,
                "notes": "; ".join(notes),
                "schema": info,
            }
        )
    return rows


def write_tier_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_name",
        "source",
        "local_path",
        "detection_method",
        "tier",
        "tier_weight",
        "total_samples",
        "observed_positive_count",
        "unobserved_candidate_count",
        "status",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_report(rows: list[dict[str, Any]], report_path: Path, json_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = {"generated_at": utc_now(), "rows": rows}
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Schema Audit Report",
        "",
        f"- Generated at: `{json_payload['generated_at']}`",
        f"- Local datasets audited: `{sum(1 for row in rows if row['source'] == 'local_npz')}`",
        f"- CCLMoff entries audited: `{sum(1 for row in rows if row['source'].startswith('cclmoff'))}`",
        "",
        "## Tier Labels",
        "",
        "| Dataset | Source | Method | Tier | Weight | Rows | Observed positives | Unobserved candidates | Status | Notes |",
        "|:---|:---|:---|:---:|---:|---:|---:|---:|:---|:---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset_name} | {source} | {detection_method} | {tier} | {tier_weight} | {total_samples} | "
            "{observed_positive_count} | {unobserved_candidate_count} | {status} | {notes} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `label=1` is treated as experimentally observed off-target signal.",
            "- `label=0` is treated as `unobserved_candidate`, not as a verified safe site.",
            "- K562 and Tasi do not map cleanly to the current detection-method Tier table and remain `tier_unknown` until provenance is confirmed.",
            "- CCLMoff GitHub releases have no assets. The expected public CSV is `data/cclmoff/09212024_CCLMoff_dataset.csv` from Figshare.",
            "- The public CCLMoff CSV is known to lack detection-method/source-study/cell-line metadata; Tier-aware training requires an external `sgRNA_type` mapping table.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local and CCLMoff dataset schema for P0 startup.")
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--cclmoff-root", type=Path, default=DEFAULT_CCLMOFF_ROOT)
    parser.add_argument("--tier-csv", type=Path, default=DEFAULT_TIER_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = local_rows(args.local_root) + cclmoff_rows(args.cclmoff_root)
    write_tier_csv(rows, args.tier_csv)
    write_report(rows, args.report, args.json)
    print(f"Wrote {args.tier_csv}")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
