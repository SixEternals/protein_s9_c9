#!/usr/bin/env python3
"""
AGENTS.md compliance: [pre_commit=True, audit=True]
确认本文件遵守 AGENTS.md 约束

scripts/check_before_commit.py — 提交前检查

用法（手动运行或设为 git pre-commit 钩子）：
    python scripts/check_before_commit.py

检查项：
    1. AGENTS.md 是否存在
    2. 运行 audit_compliance.py
    3. Python 语法检查 (py_compile)
"""

from __future__ import annotations

import argparse
import py_compile
import subprocess
import sys
from pathlib import Path


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


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_excluded(path: Path, root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        return True
    parts = set(rel.parts)
    if parts & EXCLUDED_DIR_NAMES:
        return True
    if rel.parts and rel.parts[0] in EXCLUDED_TOP_LEVEL:
        return True
    return False


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if not _is_excluded(path, root):
            yield path


def check(*, strict: bool = False) -> bool:
    root = _project_root()
    print("=" * 60)
    print("提交前检查")
    print("=" * 60)

    # 1. 检查 AGENTS.md 存在
    agents_md = root / "AGENTS.md"
    if not agents_md.exists():
        print("🔴 AGENTS.md 不存在！")
        print("    这是强制约束文档，每个 AI 都必须读取。")
        return False
    print("✅ AGENTS.md 存在")

    # 2. 运行审计
    print("\n🔍 运行合规性审计...")
    cmd = [sys.executable, str(root / "scripts" / "audit_compliance.py")]
    if strict:
        cmd.append("--strict")
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print("🔴 审计未通过，请修复后再提交。")
        return False
    print("✅ 审计通过")

    # 3. Python 语法检查
    print("\n🔍 Python 语法检查...")
    errors: list[str] = []
    py_files = sorted(_iter_python_files(root))
    for pyfile in py_files:
        try:
            py_compile.compile(str(pyfile), doraise=True)
        except py_compile.PyCompileError as exc:
            rel = pyfile.relative_to(root)
            errors.append(f"{rel}: {exc}")

    if errors:
        print("🔴 Python 语法错误:")
        for error in errors:
            print(f"  {error}")
        return False
    print(f"✅ Python 语法检查通过（{len(py_files)} 个文件）")

    print("\n" + "=" * 60)
    print("✅ 全部检查通过，可以提交")
    print("=" * 60)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="AGENTS.md 提交前检查")
    parser.add_argument("--strict", action="store_true", help="审计 WARNING 也视为失败")
    args = parser.parse_args()
    return 0 if check(strict=args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
