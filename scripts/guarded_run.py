#!/usr/bin/env python3
"""
AGENTS.md compliance: [external_guardrails=True, guarded_run=True]
确认本文件遵守 AGENTS.md 约束

Run a manifest command only after external preflight guardrails pass.
This runner does not use a shell.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from preflight_guardrails import (
    PreflightError,
    _resolve_path,
    command_tokens,
    load_mapping_file,
    print_result,
    run_preflight,
    validate_command_tokens,
)


ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a job manifest after preflight guardrails pass")
    parser.add_argument("manifest", help="Path to job manifest YAML/JSON")
    parser.add_argument("--root", default=None, help="Project root; defaults to this repository")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print command without executing it")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else ROOT
    try:
        result = run_preflight(args.manifest, root=root)
    except PreflightError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print_result(result)
    if not result.ok:
        return 1

    manifest = load_mapping_file(result.manifest_path)
    command = command_tokens(manifest.get("command"))
    validate_command_tokens(command)

    working_dir_value = manifest.get("working_dir", ".")
    try:
        working_dir = _resolve_path(working_dir_value, root)
    except PreflightError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    env = build_env(manifest.get("env"))
    print("=" * 60)
    print(f"Working dir: {working_dir}")
    print(f"Command: {' '.join(command)}")
    if args.dry_run:
        print("Dry run: command not executed")
        return 0

    completed = subprocess.run(command, cwd=working_dir, env=env)
    return int(completed.returncode)


def build_env(extra_env: Any) -> dict[str, str]:
    env = os.environ.copy()
    if extra_env is None:
        return env
    if not isinstance(extra_env, dict):
        raise SystemExit("manifest env must be a mapping")
    for key, value in extra_env.items():
        key = str(key)
        if not ENV_KEY_PATTERN.match(key):
            raise SystemExit(f"invalid env key: {key!r}")
        env[key] = str(value)
    return env


if __name__ == "__main__":
    sys.exit(main())
