from __future__ import annotations

import json
from pathlib import Path
from typing import Any


try:
    import yaml

    def load_config(path: str | Path) -> dict[str, Any]:
        text = Path(path).read_text(encoding="utf-8")
        loaded = yaml.safe_load(text)
        if loaded is None:
            raise ValueError(f"config file is empty or invalid: {path}")
        if not isinstance(loaded, dict):
            raise ValueError(f"config file must contain a top-level mapping, got {type(loaded).__name__}: {path}")
        return loaded

except Exception:

    def load_config(path: str | Path) -> dict[str, Any]:
        text = Path(path).read_text(encoding="utf-8")
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"config file must contain a top-level object, got {type(loaded).__name__}: {path}")
        return loaded


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


def validate_config(config: dict[str, Any]) -> list[str]:
    """Early validation of training/inference config.

    Returns a list of human-readable error strings. An empty list means valid.
    """
    errors: list[str] = []

    model = str(config.get("model", "")).lower()
    encoder = str(config.get("encoder", "")).lower()

    if not model:
        errors.append("missing required field: model")
    elif model not in {"deepfocus", "conmismatch9"}:
        errors.append(f"invalid model: {model!r}; expected 'deepfocus' or 'conmismatch9'")

    if not encoder:
        errors.append("missing required field: encoder")
    elif encoder not in {"r9", "c9"}:
        errors.append(f"invalid encoder: {encoder!r}; expected 'r9' or 'c9'")

    expected = {"deepfocus": "r9", "conmismatch9": "c9"}
    if model in expected and encoder and encoder != expected[model]:
        errors.append(f"model '{model}' expects encoder '{expected[model]}', got '{encoder}'")

    datasets = resolve_dataset_files(config)
    if not datasets:
        errors.append("config must define dataset_files or dataset_specs with valid dir/name")
    else:
        for ds in datasets:
            if not Path(ds).exists():
                errors.append(f"dataset file not found: {ds}")

    weights_path = config.get("weights_path")
    if weights_path is not None and not isinstance(weights_path, str):
        errors.append(f"weights_path must be a string or null, got {type(weights_path).__name__}")

    sampling = config.get("sampling", {})
    if not isinstance(sampling, dict):
        errors.append("sampling must be a mapping")
    else:
        for key in ("positive_cap", "negative_cap"):
            val = sampling.get(key)
            if val is not None and val != "all" and not isinstance(val, int):
                errors.append(f"sampling.{key} must be an integer, 'all', or null")

    training = config.get("training", {})
    if not isinstance(training, dict):
        errors.append("training must be a mapping")
    else:
        for key, expected_type in (
            ("epochs", int),
            ("batch_size", int),
            ("patience", int),
            ("learning_rate", (int, float)),
            ("weight_decay", (int, float)),
            ("dropout", (int, float)),
        ):
            val = training.get(key)
            if val is not None and not isinstance(val, expected_type):
                errors.append(f"training.{key} must be a number, got {type(val).__name__}")

    return errors
