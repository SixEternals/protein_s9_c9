#!/usr/bin/env python3
"""
AGENTS.md compliance: [external_guardrails=True, internal_code_changes=False]
确认本文件遵守 AGENTS.md 约束

External preflight guardrails for job manifests.

This script is intentionally non-invasive: it reads a manifest and optional
config, validates AGENTS.md constraints, and exits before any training command
starts. It does not import model code or modify files.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.guardrails import GuardrailsViolation, check_eval_procedure, check_model_config


VALID_STAGES = {"train", "eval", "test", "precompute", "export", "audit"}
VALID_LAYERS = {"frontend", "middleware", "backend", "evaluation", "data", "audit"}
VALID_DYNAMIC_FUSIONS = {"attn", "cross_attn", "gate", "gated", "full", "bidirectional", "bi_attn"}
CONCAT_FUSIONS = {"concat", "simple_concat", "cat", "none"}
METRIC_AUROC = {"AUROC", "ROC_AUC", "AUC"}
METRIC_AUPRC = {"AUPRC", "PR_AUC", "AVERAGE_PRECISION", "AP"}
DISALLOWED_EXECUTABLES = {"rm", "sudo"}
SHELL_CONTROL_TOKENS = {";", "&&", "||", "|", ">", ">>", "<", "$(", "`"}


class PreflightError(Exception):
    """Raised when preflight cannot continue."""


@dataclass
class Finding:
    severity: str
    message: str


@dataclass
class PreflightResult:
    manifest_path: Path
    findings: list[Finding]

    @property
    def errors(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == "ERROR"]

    @property
    def warnings(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == "WARNING"]

    @property
    def ok(self) -> bool:
        return not self.errors


def run_preflight(manifest_path: str | Path, *, root: str | Path | None = None) -> PreflightResult:
    project_root = Path(root).resolve() if root else ROOT
    manifest_file = _resolve_path(manifest_path, project_root)
    checker = ManifestPreflight(project_root)
    checker.check(manifest_file)
    return PreflightResult(manifest_file, checker.findings)


class ManifestPreflight:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.findings: list[Finding] = []
        self.manifest: dict[str, Any] = {}
        self.config: dict[str, Any] = {}

    def check(self, manifest_path: Path) -> None:
        self.manifest = self._load_mapping(manifest_path, "manifest")

        config_path = self._optional_path("config_path")
        if config_path is not None:
            if config_path.exists():
                self.config = self._load_mapping(config_path, "config")
            else:
                self._error(f"config_path does not exist: {self._display(config_path)}")

        self._check_required_fields()
        self._check_paths()
        self._check_policy()
        self._check_bl_ownership()
        self._check_midware()
        self._check_eval()
        self._check_command()

    def _load_mapping(self, path: Path, label: str) -> dict[str, Any]:
        try:
            loaded = load_mapping_file(path)
        except Exception as exc:
            raise PreflightError(f"failed to load {label} {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise PreflightError(f"{label} must be a mapping: {path}")
        return loaded

    def _check_required_fields(self) -> None:
        for key in ("task_id", "agent", "stage", "bl_version", "architecture_layer"):
            if self._get(key) is None:
                self._error(f"manifest missing required field: {key}")

        stage = self._str("stage").lower()
        if stage and stage not in VALID_STAGES:
            self._error(f"invalid stage={stage!r}; expected one of {sorted(VALID_STAGES)}")

        layer = self._str("architecture_layer").lower()
        if layer and layer not in VALID_LAYERS:
            self._error(f"invalid architecture_layer={layer!r}; expected one of {sorted(VALID_LAYERS)}")

        if stage in {"train", "eval", "test"} and self._get("config_path") is None:
            self._error("train/eval/test manifests must declare config_path")

    def _check_paths(self) -> None:
        for key in ("model_entry",):
            path = self._optional_path(key)
            if path is not None and not path.exists():
                self._error(f"{key} does not exist: {self._display(path)}")

        working_dir = self._optional_path("working_dir", default=".")
        if working_dir is not None and not working_dir.exists():
            self._error(f"working_dir does not exist: {self._display(working_dir)}")

        if self._str("stage").lower() == "train" and self._get("model_entry") is None:
            self._warning("train manifest does not declare model_entry")

    def _check_policy(self) -> None:
        guard_cfg = {
            "use_rnafm": self._policy("use_rnafm"),
            "freeze_rnafm": self._policy("freeze_rnafm"),
            "split_mode": self._policy("split_mode"),
            "pos_weight": self._policy("pos_weight", default=None),
        }
        n_pos = self._policy("n_pos", default=None)
        n_neg = self._policy("n_neg", default=None)
        if n_pos is not None:
            guard_cfg["n_pos"] = n_pos
        if n_neg is not None:
            guard_cfg["n_neg"] = n_neg

        missing = [key for key in ("use_rnafm", "split_mode") if guard_cfg[key] is None]
        for key in missing:
            self._error(f"policy missing required field: {key}")
        if missing:
            return

        try:
            check_model_config(guard_cfg)
        except GuardrailsViolation as exc:
            self._error(str(exc))

    def _check_bl_ownership(self) -> None:
        bl_version = self._str("bl_version")
        norm = _normalize_version(bl_version)
        use_rnafm = self._policy("use_rnafm")
        layer = self._str("architecture_layer").lower()

        if _is_bl3(norm) and use_rnafm is True:
            self._error("BL3/BL3.5 manifests must set policy.use_rnafm=false")

        if _is_bl35(norm):
            if layer != "middleware":
                self._error("BL3.5 must set architecture_layer=middleware")
            if use_rnafm is not False:
                self._error("BL3.5 is middleware-only and must set policy.use_rnafm=false")

        if _is_bl4plus(norm) and use_rnafm is False and not self._allow_rnafm_absent():
            self._warning("BL4/BL5/BL6 normally include RNA-FM; set allow_rnafm_absent=true only for explicit ablation")

    def _check_midware(self) -> None:
        midware = self._mapping("midware")
        use_c9 = _as_bool(midware.get("use_c9"))
        use_r9 = _as_bool(midware.get("use_r9"))
        fusion_mode = _normalize_token(str(midware.get("fusion_mode", "")))
        allow_concat = _as_bool(midware.get("allow_concat_only")) is True
        purpose = self._str("purpose").lower()
        norm = _normalize_version(self._str("bl_version"))

        if _is_bl35(norm):
            if use_c9 is not True or use_r9 is not True:
                self._error("BL3.5 must declare midware.use_c9=true and midware.use_r9=true")
            if fusion_mode not in VALID_DYNAMIC_FUSIONS:
                self._error("BL3.5 must use dynamic fusion: attn, gate, or full")
            if allow_concat:
                self._error("BL3.5 cannot set midware.allow_concat_only=true")

        if use_c9 and use_r9 and fusion_mode in CONCAT_FUSIONS:
            if allow_concat and purpose == "ablation":
                self._warning("C9+R9 concat is allowed only as explicit ablation, not as main-line middleware")
            else:
                self._error("C9+R9 concat is blocked for main-line middleware; use attn, gate, or full")

    def _check_eval(self) -> None:
        stage = self._str("stage").lower()
        eval_cfg = self._mapping("eval")
        metrics = _normalize_metrics(eval_cfg.get("report_metrics"))

        if stage in {"eval", "test"}:
            checkpoint_type = str(eval_cfg.get("checkpoint_type", "")).strip() or None
            checkpoint_path = eval_cfg.get("checkpoint_path")
            if checkpoint_type is None:
                self._error("eval/test manifest must declare eval.checkpoint_type=best")
            elif checkpoint_type.lower() != "best":
                self._error("eval/test manifest must use eval.checkpoint_type=best")

            if checkpoint_path is None:
                self._error("eval/test manifest must declare eval.checkpoint_path")
            else:
                resolved = _resolve_path(checkpoint_path, self.root)
                try:
                    check_eval_procedure(resolved, checkpoint_type or "best", require_exists=False)
                except GuardrailsViolation as exc:
                    self._error(str(exc))

        if stage in {"eval", "test"}:
            if not _has_metric(metrics, METRIC_AUROC) or not _has_metric(metrics, METRIC_AUPRC):
                self._error("eval/test manifest must report both AUROC and AUPRC")
        elif stage == "train":
            if not _has_metric(metrics, METRIC_AUROC) or not _has_metric(metrics, METRIC_AUPRC):
                self._warning("train manifest should declare eval.report_metrics with both AUROC and AUPRC")

    def _check_command(self) -> None:
        command = self._get("command")
        if command is None:
            self._warning("manifest has no command; preflight can validate policy only")
            return

        try:
            tokens = command_tokens(command)
            validate_command_tokens(tokens)
        except PreflightError as exc:
            self._error(str(exc))

    def _policy(self, key: str, default: Any = None) -> Any:
        policy = self._mapping("policy")
        if key in policy:
            return policy[key]
        if key in self.manifest:
            return self.manifest[key]
        found = _deep_find(self.config, key)
        return default if found is None else found

    def _mapping(self, key: str) -> dict[str, Any]:
        value = self._get(key)
        return value if isinstance(value, dict) else {}

    def _get(self, key: str, default: Any = None) -> Any:
        return self.manifest.get(key, default)

    def _str(self, key: str) -> str:
        value = self._get(key)
        return "" if value is None else str(value)

    def _optional_path(self, key: str, default: str | None = None) -> Path | None:
        value = self._get(key, default)
        if value is None:
            return None
        return _resolve_path(value, self.root)

    def _allow_rnafm_absent(self) -> bool:
        value = self._get("allow_rnafm_absent")
        if value is None:
            value = self._mapping("policy").get("allow_rnafm_absent")
        return _as_bool(value) is True

    def _display(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def _error(self, message: str) -> None:
        self.findings.append(Finding("ERROR", message))

    def _warning(self, message: str) -> None:
        self.findings.append(Finding("WARNING", message))


def command_tokens(command: Any) -> list[str]:
    if isinstance(command, str):
        tokens = shlex.split(command)
    elif isinstance(command, list):
        tokens = [str(item) for item in command]
    else:
        raise PreflightError("command must be a string or list of strings")
    if not tokens:
        raise PreflightError("command is empty")
    return tokens


def load_mapping_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
    except Exception:
        stripped = text.lstrip()
        if stripped.startswith("{"):
            loaded = json.loads(text)
        else:
            loaded = parse_yaml_subset(text)

    if loaded is None:
        raise ValueError(f"file is empty or invalid: {path}")
    if not isinstance(loaded, dict):
        raise ValueError(f"file must contain a top-level mapping, got {type(loaded).__name__}: {path}")
    return loaded


def parse_yaml_subset(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by job manifests and project configs.

    This is a dependency-free fallback for environments without PyYAML. It
    supports nested mappings, simple lists, inline lists, booleans, null, and
    numeric scalars. It is not a general YAML parser.
    """
    prepared = _prepare_yaml_lines(text)
    if not prepared:
        return {}
    parsed, index = _parse_yaml_block(prepared, 0, prepared[0][0])
    if index < len(prepared):
        line_no = prepared[index][2]
        raise ValueError(f"unexpected YAML content at line {line_no}")
    if not isinstance(parsed, dict):
        raise ValueError("top-level YAML value must be a mapping")
    return parsed


def _prepare_yaml_lines(text: str) -> list[tuple[int, str, int]]:
    lines: list[tuple[int, str, int]] = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        if raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = _strip_yaml_comment(raw[indent:]).rstrip()
        if stripped:
            lines.append((indent, stripped, line_no))
    return lines


def _parse_yaml_block(
    lines: list[tuple[int, str, int]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index

    current_indent, content, line_no = lines[index]
    if current_indent < indent:
        return {}, index
    if current_indent > indent:
        raise ValueError(f"unexpected indentation at line {line_no}")

    if content.startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_mapping(lines, index, indent)


def _parse_yaml_mapping(
    lines: list[tuple[int, str, int]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    while index < len(lines):
        current_indent, content, line_no = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"unexpected indentation at line {line_no}")
        if content.startswith("- "):
            break
        if ":" not in content:
            raise ValueError(f"expected key: value at line {line_no}")

        key, value_text = content.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        if not key:
            raise ValueError(f"empty key at line {line_no}")

        index += 1
        if value_text in {">", "|"}:
            value, index = _parse_yaml_block_scalar(lines, index, indent, literal=value_text == "|")
        elif value_text:
            value = _parse_yaml_scalar(value_text)
        elif index < len(lines) and lines[index][0] > indent:
            value, index = _parse_yaml_block(lines, index, lines[index][0])
        else:
            value = None
        mapping[key] = value
    return mapping, index


def _parse_yaml_list(
    lines: list[tuple[int, str, int]],
    index: int,
    indent: int,
) -> tuple[list[Any], int]:
    values: list[Any] = []
    while index < len(lines):
        current_indent, content, line_no = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"unexpected indentation at line {line_no}")
        if not content.startswith("- "):
            break

        item_text = content[2:].strip()
        index += 1
        if not item_text:
            if index < len(lines) and lines[index][0] > indent:
                value, index = _parse_yaml_block(lines, index, lines[index][0])
            else:
                value = None
        elif _looks_like_inline_mapping(item_text):
            key, value_text = item_text.split(":", 1)
            value = {key.strip(): _parse_yaml_scalar(value_text.strip())}
        else:
            value = _parse_yaml_scalar(item_text)
        values.append(value)
    return values, index


def _parse_yaml_block_scalar(
    lines: list[tuple[int, str, int]],
    index: int,
    parent_indent: int,
    *,
    literal: bool,
) -> tuple[str, int]:
    chunks: list[str] = []
    child_indent: int | None = None
    while index < len(lines) and lines[index][0] > parent_indent:
        indent, content, _line_no = lines[index]
        if child_indent is None:
            child_indent = indent
        chunks.append(content)
        index += 1
    return ("\n".join(chunks) if literal else " ".join(chunks)).strip(), index


def _parse_yaml_scalar(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(part.strip()) for part in _split_inline_list(inner)]
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+|\d+)([eE][-+]?\d+)?", value):
        return float(value)
    return value


def _strip_yaml_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        if char == "#" and quote is None:
            if index == 0 or value[index - 1].isspace():
                return value[:index]
    return value


def _split_inline_list(value: str) -> list[str]:
    items: list[str] = []
    start = 0
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == "," and quote is None:
            items.append(value[start:index])
            start = index + 1
    items.append(value[start:])
    return items


def _looks_like_inline_mapping(value: str) -> bool:
    if ":" not in value:
        return False
    if value.startswith(("http://", "https://")):
        return False
    key, _rest = value.split(":", 1)
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key.strip()))


def validate_command_tokens(tokens: list[str]) -> None:
    executable = Path(tokens[0]).name
    if executable in DISALLOWED_EXECUTABLES:
        raise PreflightError(f"guarded_run refuses executable: {executable}")

    for token in tokens:
        if token in SHELL_CONTROL_TOKENS or any(marker in token for marker in ("$(", "`")):
            raise PreflightError(f"guarded_run refuses shell control token in command: {token!r}")

    if executable == "git":
        operation = tokens[1] if len(tokens) > 1 else ""
        if operation in {"commit", "push", "reset", "rebase", "clean"}:
            raise PreflightError(f"guarded_run refuses git {operation}")


def _resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PreflightError(f"path is outside project root: {path}") from exc
    return resolved


def _deep_find(mapping: Any, key: str) -> Any:
    if isinstance(mapping, dict):
        if key in mapping:
            return mapping[key]
        for value in mapping.values():
            found = _deep_find(value, key)
            if found is not None:
                return found
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0", "none", "null"}:
            return False
    return None


def _normalize_version(value: str) -> str:
    return re.sub(r"[^A-Z0-9.]+", "-", value.upper()).strip("-")


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _is_bl3(version: str) -> bool:
    return version.startswith("BL3")


def _is_bl35(version: str) -> bool:
    return version.startswith("BL3.5") or version.startswith("BL35")


def _is_bl4plus(version: str) -> bool:
    return version.startswith(("BL4", "BL5", "BL6"))


def _normalize_metrics(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = value
    else:
        return set()
    return {_normalize_metric(str(item)) for item in raw}


def _normalize_metric(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _has_metric(metrics: set[str], accepted: set[str]) -> bool:
    return any(metric in accepted for metric in metrics)


def print_result(result: PreflightResult) -> None:
    print("=" * 60)
    print(f"Preflight: {result.manifest_path}")
    print("=" * 60)

    if not result.findings:
        print("PASS: no findings")
        return

    for finding in result.findings:
        print(f"[{finding.severity}] {finding.message}")

    print("-" * 60)
    print(f"ERROR: {len(result.errors)}")
    print(f"WARNING: {len(result.warnings)}")
    print("PASS" if result.ok else "FAIL")


def main() -> int:
    parser = argparse.ArgumentParser(description="External manifest preflight guardrails")
    parser.add_argument("manifest", help="Path to job manifest YAML/JSON")
    parser.add_argument("--root", default=None, help="Project root; defaults to this repository")
    args = parser.parse_args()

    try:
        result = run_preflight(args.manifest, root=args.root)
    except PreflightError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
