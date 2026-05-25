from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path
from typing import Any


DEFAULT_CSV = Path("data/cclmoff/09212024_CCLMoff_dataset.csv")
DEFAULT_REPORT = Path("results/audits/cclmoff_label_audit.md")
DEFAULT_JSON = Path("results/audits/cclmoff_label_audit.json")
EXPECTED_MD5 = "2a9be5c69a89c8eee3fdef0c03efae3a"
VALID_SEQUENCE_CHARS = set("ATCGUN-")
STATUS_BY_LABEL = {
    "1.0": "observed_positive",
    "1": "observed_positive",
    "0.0": "unobserved_candidate",
    "0": "unobserved_candidate",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def update_read_bucket(bucket: dict[str, Any], value: str) -> None:
    numeric = read_float(value)
    if numeric is None:
        bucket["missing"] += 1
        return
    if numeric == 0:
        bucket["zero"] += 1
    elif numeric > 0:
        bucket["positive"] += 1
    else:
        bucket["below_zero"] += 1
    bucket["min"] = numeric if bucket["min"] is None else min(bucket["min"], numeric)
    bucket["max"] = numeric if bucket["max"] is None else max(bucket["max"], numeric)


def default_read_bucket() -> dict[str, Any]:
    return {
        "missing": 0,
        "zero": 0,
        "positive": 0,
        "below_zero": 0,
        "min": None,
        "max": None,
    }


def audit(path: Path, max_examples: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    label_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    label_by_method: dict[str, Counter[str]] = defaultdict(Counter)
    status_by_method: dict[str, Counter[str]] = defaultdict(Counter)
    sgRNA_types: set[str] = set()
    sgRNA_seq_values: set[str] = set()
    sgRNA_seq_length_counts: Counter[str] = Counter()
    off_seq_length_counts: Counter[str] = Counter()
    length_field_counts: Counter[str] = Counter()
    top_pam_counts: Counter[str] = Counter()
    blank_counts: Counter[str] = Counter()
    sequence_characters: Counter[str] = Counter()
    invalid_sequence_examples: list[dict[str, Any]] = []
    read_by_status: dict[str, dict[str, Any]] = defaultdict(default_read_bucket)
    rows = 0
    columns: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            label = str(row.get("label", "")).strip()
            status = STATUS_BY_LABEL.get(label, f"unexpected_label:{label}")
            method = (row.get("Method") or "").strip() or "<blank>"
            sgRNA_type = (row.get("sgRNA_type") or "").strip()
            sgRNA_seq = (row.get("sgRNA_seq") or "").strip().upper()
            off_seq = (row.get("off_seq") or "").strip().upper()
            length_field = (row.get("Length") or "").strip()

            label_counts[label] += 1
            status_counts[status] += 1
            method_counts[method] += 1
            label_by_method[method][label] += 1
            status_by_method[method][status] += 1
            if sgRNA_type:
                sgRNA_types.add(sgRNA_type)
            if sgRNA_seq:
                sgRNA_seq_values.add(sgRNA_seq)
            sgRNA_seq_length_counts[str(len(sgRNA_seq))] += 1
            off_seq_length_counts[str(len(off_seq))] += 1
            length_field_counts[length_field] += 1
            if len(off_seq) >= 3:
                top_pam_counts[off_seq[-3:]] += 1
            for field in ("Method", "Length"):
                if not (row.get(field) or "").strip():
                    blank_counts[field] += 1

            for sequence_field, sequence in (("sgRNA_seq", sgRNA_seq), ("off_seq", off_seq)):
                sequence_characters.update(sequence)
                invalid = sorted(set(sequence) - VALID_SEQUENCE_CHARS)
                if invalid and len(invalid_sequence_examples) < max_examples:
                    invalid_sequence_examples.append(
                        {
                            "row": rows,
                            "field": sequence_field,
                            "invalid": invalid,
                            "value": sequence[:80],
                        }
                    )

            update_read_bucket(read_by_status[status], row.get("read") or "")

    non_binary_labels = {
        label: count
        for label, count in label_counts.items()
        if label not in {"0", "0.0", "1", "1.0"}
    }
    return {
        "generated_at": utc_now(),
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "md5": file_md5(path),
        "expected_md5": EXPECTED_MD5,
        "md5_matches_expected": file_md5(path) == EXPECTED_MD5,
        "columns": columns,
        "rows": rows,
        "label_counts": dict(label_counts),
        "status_counts": dict(status_counts),
        "non_binary_labels": non_binary_labels,
        "method_counts": dict(method_counts),
        "label_by_method": {method: dict(counts) for method, counts in label_by_method.items()},
        "status_by_method": {method: dict(counts) for method, counts in status_by_method.items()},
        "sgRNA_type_unique": len(sgRNA_types),
        "sgRNA_seq_unique": len(sgRNA_seq_values),
        "sgRNA_seq_length_counts": dict(sgRNA_seq_length_counts),
        "off_seq_length_counts": dict(off_seq_length_counts),
        "Length_field_counts": dict(length_field_counts),
        "top_pam_counts": dict(top_pam_counts.most_common(30)),
        "blank_counts": dict(blank_counts),
        "read_by_status": dict(read_by_status),
        "sequence_characters": dict(sequence_characters),
        "invalid_sequence_examples": invalid_sequence_examples,
    }


def write_reports(payload: dict[str, Any], report_path: Path, json_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    method_lines = []
    for method, count in sorted(payload["method_counts"].items(), key=lambda item: item[1], reverse=True):
        statuses = payload["status_by_method"].get(method, {})
        method_lines.append(
            f"| `{method}` | {count} | {statuses.get('observed_positive', 0)} | {statuses.get('unobserved_candidate', 0)} |"
        )

    lines = [
        "# CCLMoff 数据标注审计",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- CSV: `{payload['path']}`",
        f"- Rows: `{payload['rows']}`",
        f"- Columns: `{payload['columns']}`",
        f"- File size bytes: `{payload['file_size_bytes']}`",
        f"- MD5: `{payload['md5']}`",
        f"- MD5 matches Figshare API: `{payload['md5_matches_expected']}`",
        "",
        "## 标注结论",
        "",
        f"- `observed_positive`: `{payload['status_counts'].get('observed_positive', 0)}`",
        f"- `unobserved_candidate`: `{payload['status_counts'].get('unobserved_candidate', 0)}`",
        f"- Non-binary labels: `{payload['non_binary_labels']}`",
        f"- `label=0` 在本项目中只表示 `unobserved_candidate`，不能解释为实验验证安全。",
        "",
        "## read 字段一致性",
        "",
        f"- `observed_positive`: `{payload['read_by_status'].get('observed_positive', {})}`",
        f"- `unobserved_candidate`: `{payload['read_by_status'].get('unobserved_candidate', {})}`",
        "",
        "## 方法字段",
        "",
        "| Method | rows | observed_positive | unobserved_candidate |",
        "|:---|---:|---:|---:|",
        *method_lines,
        "",
        "## 序列字段",
        "",
        f"- sgRNA_seq length counts: `{payload['sgRNA_seq_length_counts']}`",
        f"- off_seq length counts: `{payload['off_seq_length_counts']}`",
        f"- Length field counts: `{payload['Length_field_counts']}`",
        f"- Blank counts: `{payload['blank_counts']}`",
        f"- Unique sgRNA_type: `{payload['sgRNA_type_unique']}`",
        f"- Unique sgRNA_seq: `{payload['sgRNA_seq_unique']}`",
        f"- Invalid sequence examples: `{payload['invalid_sequence_examples']}`",
        "",
        "## 需要注意",
        "",
        "- Figshare v2 CSV 实际有 11 个字段，不是早期记录中的 5 个字段。",
        "- `Method` 和 `Length` 均有大量空值；Tier-aware 训练只能对有明确 Method 或已补映射的样本使用。",
        "- 序列长度混有 23nt 和 24nt，并包含 `-`；旧 R9/C9 模块的 P0 加权平均先使用本地 23nt NPZ 数据，不直接吃这个混合 CSV。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CCLMoff CSV labels and annotation consistency.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = audit(args.csv, args.max_examples)
    write_reports(payload, args.report, args.json)
    print(f"Wrote {args.report}")
    print(f"Wrote {args.json}")
    print(f"rows={payload['rows']} md5_matches_expected={payload['md5_matches_expected']}")
    print(f"status_counts={payload['status_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
