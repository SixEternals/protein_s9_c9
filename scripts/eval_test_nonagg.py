#!/usr/bin/env python3
"""
AGENTS.md compliance: [use_rnafm=False, freeze_rnafm=False,
                       split_mode=irrelevant, pos_weight=None]
确认本文件遵守 AGENTS.md 约束

说明：本文件是说明性 stub，当前不执行实际评估。

背景：test_nonAGG sanity eval 需要模型在 test_seenPAM（formal_test 中 PAM != AGG）子集上做 eval-only
并导出 predictions。当前 results/bl5_v4_*_holdout_agg/test_predictions.csv 只包含 test_H（PAM=AGG），
因此直接过滤 PAM_original != AGG 会得到 0 行。为避免大范围修改 train_bl5.py 的 split 选择逻辑，
test_nonAGG 评估列为 next check，待后续用 test_seenPAM split 重新跑 eval-only 时实现。
"""
from __future__ import annotations

import sys


def main() -> int:
    print("[INFO] scripts/eval_test_nonagg.py is currently a stub.")
    print("[INFO] Current test_predictions.csv only contains test_H (PAM=AGG) samples.")
    print("[INFO] To obtain test_nonAGG metrics, re-run eval-only with split key test_seenPAM / test_nonAGG first.")
    print("[INFO] This sanity check is documented as 'next check' in the holdout AGG report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
