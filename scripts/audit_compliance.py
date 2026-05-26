#!/usr/bin/env python3
"""
AGENTS.md compliance: [audit=True, target=all_python_files]
确认本文件遵守 AGENTS.md 约束

scripts/audit_compliance.py — 自动审计 AGENTS.md 合规性

用法：
    python scripts/audit_compliance.py
    python scripts/audit_compliance.py models/bl0_cclmoff.py
    python scripts/audit_compliance.py --strict

返回码：
    0 = 没有 ERROR 级违规
    1 = 存在 ERROR 级违规，或 --strict 下存在 WARNING
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VALID_SPLIT_MODES = {"random", "sgrna_safe", "loo"}
COMPLIANCE_MARKER = "AGENTS.md compliance"
EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "venv",
}
EXCLUDED_TOP_LEVEL = {
    "artifacts",
    "data",
    "offtarget_fusion_project",
    "output",
    "reference",
    "runs",
}
ACTIVE_CODE_DIRS = {"models", "encoders", "scripts", "utils"}
SELF_FILES = {
    Path("scripts/audit_compliance.py"),
    Path("scripts/check_before_commit.py"),
    Path("utils/guardrails.py"),
}


@dataclass(frozen=True)
class Finding:
    file: Path
    line: int
    constraint: str
    message: str
    severity: str = "ERROR"


class ComplianceAuditor:
    """AGENTS.md 合规性审计器。"""

    def __init__(
        self,
        project_root: str | Path | None = None,
        *,
        strict: bool = False,
        require_marker: bool = False,
    ) -> None:
        self.root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        self.root = self.root.resolve()
        self.strict = strict
        self.require_marker = require_marker
        self.findings: list[Finding] = []

    def audit_all(self) -> int:
        py_files = sorted(self._iter_python_files())
        print(f"🔍 审计 {len(py_files)} 个 Python 文件...\n")

        for pyfile in py_files:
            self.audit_file(pyfile)

        return self._summarize()

    def audit_paths(self, paths: Iterable[str | Path]) -> int:
        for path in paths:
            self.audit_file(Path(path))
        return self._summarize()

    def audit_file(self, filepath: str | Path) -> None:
        path = Path(filepath)
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()

        rel_path = self._rel(path)
        if path.suffix != ".py" or self._is_excluded(path):
            return

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            self._add(rel_path, 1, "通用", f"无法读取文件: {exc}", "ERROR")
            return

        lines = content.splitlines()
        try:
            tree = ast.parse(content, filename=str(rel_path))
        except SyntaxError as exc:
            self._add(rel_path, exc.lineno or 1, "通用", f"Python 语法错误: {exc.msg}", "ERROR")
            return

        self._check_compliance_marker(rel_path, content)
        self._check_negative_term(rel_path, tree)
        self._check_bare_9bit(rel_path, content)
        self._check_rnafm_config(rel_path, content)
        self._check_run_encoding_scope(rel_path, content, lines, tree)
        self._check_model_guardrails(rel_path, content, tree)
        self._check_split_defaults(rel_path, tree)
        self._check_pos_weight(rel_path, tree)
        self._check_eval_checkpoint(rel_path, content, lines)
        self._check_metric_pair(rel_path, content)
        self._check_cclmoff_metadata_claims(rel_path, content, lines)

    def _iter_python_files(self) -> Iterable[Path]:
        for path in self.root.rglob("*.py"):
            if not self._is_excluded(path):
                yield path

    def _is_excluded(self, path: Path) -> bool:
        try:
            rel = path.resolve().relative_to(self.root)
        except ValueError:
            return True
        parts = set(rel.parts)
        if parts & EXCLUDED_DIR_NAMES:
            return True
        if rel.parts and rel.parts[0] in EXCLUDED_TOP_LEVEL:
            return True
        return False

    def _rel(self, path: Path) -> Path:
        try:
            return path.resolve().relative_to(self.root)
        except ValueError:
            return path

    def _add(
        self,
        rel_path: Path,
        line: int,
        constraint: str | int,
        message: str,
        severity: str = "ERROR",
    ) -> None:
        self.findings.append(
            Finding(
                file=rel_path,
                line=max(int(line), 1),
                constraint=str(constraint),
                message=message,
                severity=severity,
            )
        )

    def _line_for_regex(self, lines: list[str], pattern: str) -> int:
        regex = re.compile(pattern)
        for i, line in enumerate(lines, 1):
            if regex.search(line):
                return i
        return 1

    def _is_active_code(self, rel_path: Path) -> bool:
        return bool(rel_path.parts and rel_path.parts[0] in ACTIVE_CODE_DIRS)

    def _check_compliance_marker(self, rel_path: Path, content: str) -> None:
        if rel_path.name == "__init__.py" or not self._is_active_code(rel_path):
            return
        if rel_path in SELF_FILES:
            return
        if COMPLIANCE_MARKER not in content[:800]:
            severity = "ERROR" if self.require_marker else "WARNING"
            self._add(
                rel_path,
                1,
                "标记",
                "文件开头缺少 AGENTS.md compliance 声明",
                severity,
            )

    def _check_negative_term(self, rel_path: Path, tree: ast.AST) -> None:
        if rel_path in SELF_FILES:
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "negative":
                self._add(
                    rel_path,
                    getattr(node, "lineno", 1),
                    "术语",
                    '禁止把 "negative" 作为核心数据语义，请使用 "unobserved_candidate"',
                    "WARNING",
                )

    def _check_bare_9bit(self, rel_path: Path, content: str) -> None:
        if rel_path in SELF_FILES:
            return
        if "9bit" in content:
            self._add(
                rel_path,
                self._line_for_regex(content.splitlines(), r"9bit"),
                "命名",
                '新增代码禁止裸 "9bit"，请使用 R9/C9/region/run 等明确命名',
                "WARNING",
            )

    def _check_rnafm_config(self, rel_path: Path, content: str) -> None:
        if rel_path in SELF_FILES:
            return
        lowered = content.lower()
        if "use_rnafm" in lowered and "freeze_rnafm" not in lowered:
            self._add(
                rel_path,
                self._line_for_regex(content.splitlines(), r"use_rnafm"),
                2,
                "发现 use_rnafm 但未发现 freeze_rnafm；use_rnafm=true 时必须显式声明 freeze_rnafm",
                "WARNING",
            )
        if ("rna-fm" in lowered or "rnafm" in lowered or "rna_fm" in lowered) and "use_rnafm" not in lowered:
            self._add(
                rel_path,
                self._line_for_regex(content.splitlines(), r"rna-?fm|rna_fm"),
                1,
                "文件涉及 RNA-FM，但未发现 use_rnafm 显式配置检查",
                "WARNING",
            )

    def _check_run_encoding_scope(
        self,
        rel_path: Path,
        content: str,
        lines: list[str],
        tree: ast.AST,
    ) -> None:
        lowered = content.lower()
        run_related = any(token in lowered for token in ("compute_run", "run_state", "run encoding", "连续错配"))
        if not run_related:
            return
        risky_patterns = [
            r"range\(\s*23\s*\)",
            r"range\(\s*1\s*,\s*24\s*\)",
            r"range\(\s*0\s*,\s*23\s*\)",
            r"\[:\s*23\s*\]",
        ]
        for pattern in risky_patterns:
            if re.search(pattern, content):
                line = self._line_for_regex(lines, pattern)
                if self._is_run_scope_line(rel_path, lines, tree, line):
                    self._add(
                        rel_path,
                        line,
                        3,
                        "Run 编码疑似覆盖 23 位，PAM positions 21-23 不得参与连续错配状态计算",
                        "ERROR",
                    )

    def _check_model_guardrails(self, rel_path: Path, content: str, tree: ast.AST) -> None:
        if not rel_path.parts or rel_path.parts[0] != "models" or rel_path.name == "__init__.py":
            return
        has_model_class = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [self._name_of(base) for base in node.bases]
                if "Module" in bases or "nn.Module" in bases or "Model" in node.name or "model" in node.name.lower():
                    has_model_class = True
                    break
        if has_model_class and "check_model_config" not in content:
            self._add(
                rel_path,
                1,
                "通用",
                "模型类中未发现 check_model_config(config) 调用，建议在 __init__ 中添加 Guardrails 检查",
                "WARNING",
            )

    def _is_run_scope_line(
        self,
        rel_path: Path,
        lines: list[str],
        tree: ast.AST,
        line: int,
    ) -> bool:
        filename = rel_path.name.lower()
        if "run" in filename or "c9_encoder" in filename:
            return True

        window_start = max(line - 4, 0)
        window_end = min(line + 3, len(lines))
        window = "\n".join(lines[window_start:window_end]).lower()
        if any(token in window for token in ("run", "连续错配", "mismatch run")):
            return True

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            end_lineno = getattr(node, "end_lineno", getattr(node, "lineno", line))
            if node.lineno <= line <= end_lineno and "run" in node.name.lower():
                return True
        return False

    def _check_split_defaults(self, rel_path: Path, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if self._is_cfg_get_with_default(node, "split_mode"):
                    self._add(
                        rel_path,
                        getattr(node, "lineno", 1),
                        4,
                        '禁止 cfg.get("split_mode", "...") 这类默认 split，必须由 config 显式声明',
                        "ERROR",
                    )
                if self._is_split_argparse_default(node):
                    self._add(
                        rel_path,
                        getattr(node, "lineno", 1),
                        4,
                        "argparse 中 split_mode 不应提供默认 split；必须由 config 或命令行显式声明",
                        "ERROR",
                    )

    def _check_pos_weight(self, rel_path: Path, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "pos_weight":
                value = self._literal_number(node.value)
                if value is not None and value > 50:
                    self._add(rel_path, node.lineno, 5, f"pos_weight={value:g} 超过上限 50", "ERROR")
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(self._name_of(target) == "pos_weight" for target in targets):
                    value = self._literal_number(node.value)
                    if value is not None and value > 50:
                        self._add(rel_path, node.lineno, 5, f"pos_weight={value:g} 超过上限 50", "ERROR")

    def _check_eval_checkpoint(self, rel_path: Path, content: str, lines: list[str]) -> None:
        if rel_path.parts and rel_path.parts[0] == "tests":
            return
        name = str(rel_path).lower()
        is_eval = any(token in name for token in ("eval", "evaluate", "test"))
        if not is_eval:
            return
        if re.search(r"last\.pt|checkpoint_type\s*=\s*['\"]last['\"]", content):
            self._add(
                rel_path,
                self._line_for_regex(lines, r"last\.pt|checkpoint_type\s*=\s*['\"]last['\"]"),
                6,
                "评估/测试代码疑似使用 last checkpoint；test 必须使用 best.pt",
                "ERROR",
            )
        if ("torch.load" in content or "checkpoint" in content.lower()) and "check_eval_procedure" not in content:
            self._add(
                rel_path,
                1,
                6,
                "评估/测试脚本未发现 check_eval_procedure(path, 'best') 调用",
                "WARNING",
            )

    def _check_metric_pair(self, rel_path: Path, content: str) -> None:
        if rel_path in SELF_FILES:
            return
        name = str(rel_path).lower()
        if not any(token in name for token in ("eval", "evaluate", "test", "train")):
            return
        lowered = content.lower()
        has_auroc = "auroc" in lowered or "roc_auc" in lowered
        has_auprc = "auprc" in lowered or "average_precision" in lowered
        if has_auroc != has_auprc:
            missing = "AUPRC" if has_auroc else "AUROC"
            self._add(
                rel_path,
                1,
                7,
                f"指标输出疑似只包含一个核心指标，缺少 {missing}",
                "WARNING",
            )

    def _check_cclmoff_metadata_claims(self, rel_path: Path, content: str, lines: list[str]) -> None:
        outdated_patterns = [
            r"CSV\s*只有\s*5\s*字段",
            r"only\s+5\s+fields",
            r"5\s+columns",
        ]
        for pattern in outdated_patterns:
            if re.search(pattern, content, flags=re.IGNORECASE):
                self._add(
                    rel_path,
                    self._line_for_regex(lines, pattern),
                    8,
                    "CCLMoff 最新审计为 11 字段且 Method/Length 大量为空，禁止沿用“只有 5 字段”的过期结论",
                    "ERROR",
                )

    def _summarize(self) -> int:
        errors = [f for f in self.findings if f.severity == "ERROR"]
        warnings = [f for f in self.findings if f.severity == "WARNING"]

        print("\n" + "=" * 60)
        if not self.findings:
            print("✅ 所有文件通过 AGENTS.md 审计！")
            return 0

        print(f"🔴 ERROR: {len(errors)}")
        print(f"⚠️  WARNING: {len(warnings)}")
        for finding in sorted(self.findings, key=lambda f: (f.severity != "ERROR", str(f.file), f.line)):
            print(f"\n  [{finding.severity}] {finding.file}:{finding.line}")
            print(f"    约束#{finding.constraint}: {finding.message}")

        if errors or (self.strict and warnings):
            return 1
        return 0

    def _name_of(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._name_of(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    def _literal_number(self, node: ast.AST) -> float | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        ):
            return -float(node.operand.value)
        return None

    def _literal_string(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _is_cfg_get_with_default(self, node: ast.Call, key: str) -> bool:
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "get":
            return False
        if len(node.args) < 2:
            return False
        first_arg = self._literal_string(node.args[0])
        second_arg = self._literal_string(node.args[1])
        return first_arg == key and second_arg in VALID_SPLIT_MODES

    def _is_split_argparse_default(self, node: ast.Call) -> bool:
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            return False
        option_names = {self._literal_string(arg) for arg in node.args}
        option_names.discard(None)
        if not ({"--split_mode", "--split-mode"} & option_names):
            return False
        for keyword in node.keywords:
            if keyword.arg == "default" and self._literal_string(keyword.value) in VALID_SPLIT_MODES:
                return True
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="AGENTS.md 合规性审计")
    parser.add_argument("files", nargs="*", help="指定审计的文件；默认审计项目内所有 .py 文件")
    parser.add_argument("--strict", action="store_true", help="将 WARNING 也视为失败")
    parser.add_argument("--require-marker", action="store_true", help="缺少 AGENTS.md compliance 标记时直接失败")
    parser.add_argument("--root", default=None, help="项目根目录，默认自动识别")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="预留参数：当前版本不自动修改文件，避免覆盖其他 AI 工作",
    )
    args = parser.parse_args()

    if args.fix:
        print("⚠️  --fix 当前为只读预留参数；本脚本不会自动修改文件。")

    auditor = ComplianceAuditor(
        project_root=args.root,
        strict=args.strict,
        require_marker=args.require_marker,
    )

    if args.files:
        return auditor.audit_paths(args.files)
    return auditor.audit_all()


if __name__ == "__main__":
    sys.exit(main())
