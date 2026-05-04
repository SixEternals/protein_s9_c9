from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def save_config(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_dataset_files(config: dict[str, Any]) -> list[str]:
    dataset_files = config.get("dataset_files")
    if dataset_files:
        return [str(item) for item in dataset_files]

    specs = config.get("dataset_specs") or []
    if not specs:
        return []

    variant_digit = str(config.get("data_variant_digit", "9"))
    suffix = variant_digit + "bit"
    resolved: list[str] = []
    for spec in specs:
        directory = Path(spec["dir"])
        name = str(spec.get("name") or directory.name)
        resolved.append(str(directory / f"{name}_{suffix}.npz"))
    return resolved
