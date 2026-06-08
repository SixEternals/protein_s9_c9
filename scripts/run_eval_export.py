#!/usr/bin/env python3
"""
AGENTS.md compliance: [use_rnafm=eval_only, freeze_rnafm=eval_only,
                       split_mode=irrelevant, pos_weight=None]
确认本文件遵守 AGENTS.md 约束

说明：本脚本本身不定义模型训练，仅调用已有 config 和 best.pt 执行 eval-only export。

Run eval-only for BL5 holdout models and export test predictions.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_eval(config_path: Path, output_dir: Path) -> int:
    # Load config and inject export_test_predictions
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config.setdefault("outputs", {})
    config["outputs"]["export_test_predictions"] = True

    # Save temp config
    temp_config_path = output_dir / "_eval_config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)

    # Run eval-only via train_bl5 (single GPU, no DDP, to ensure main process exports predictions)
    import subprocess
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # Clear DDP env vars so setup_distributed() sees world_size=1
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    cmd = [
        sys.executable,
        "scripts/train_bl5.py",
        "--config", str(temp_config_path),
        "--output-dir", str(output_dir),
        "--eval-only",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pam-config", default="configs/bl5_v4_pam_holdout_agg.yaml")
    parser.add_argument("--nopam-config", default="configs/bl5_v4_nopam_holdout_agg.yaml")
    parser.add_argument("--pam-dir", default="results/bl5_v4_pam_holdout_agg")
    parser.add_argument("--nopam-dir", default="results/bl5_v4_nopam_holdout_agg")
    args = parser.parse_args()

    pam_dir = Path(args.pam_dir)
    nopam_dir = Path(args.nopam_dir)

    print("=== Eval PAM ===")
    rc = run_eval(Path(args.pam_config), pam_dir)
    if rc != 0:
        return rc

    print("\n=== Eval NoPAM ===")
    rc = run_eval(Path(args.nopam_config), nopam_dir)
    return rc


if __name__ == "__main__":
    sys.exit(main())
