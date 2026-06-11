# 56. BL6-1 Multi-Seed Repeat 执行记录

> Date: 2026-06-11  
> Phase: Multi-seed repeat — Stage A (prep) + Stage B (training) + Stage C (summary)  
> Executor: Claude  

---

## 1. 任务范围

为 BL6-1 执行 seed43/44 multi-seed repeat，加上已有 seed42，形成 3-seed 稳定性评估。所有训练使用同一 `formal_split_bl5_seed42.json`，仅改变 training seed。

## 2. 新增/修改文件

| 文件 | 说明 |
|:---|:---|
| `configs/bl6_1_pam_gated_fusion_seed43.yaml` | seed43 config |
| `configs/bl6_1_pam_gated_fusion_seed44.yaml` | seed44 config |
| `run/run_bl6_1_seed43_2gpu.sh` | seed43 launcher |
| `run/run_bl6_1_seed44_2gpu.sh` | seed44 launcher |
| `scripts/summarize_bl6_1_multiseed.py` | Multi-seed summary script |
| `results/bl6_1_validation/multiseed_summary.csv` | Summary CSV |
| `results/bl6_1_validation/multiseed_summary.json` | Summary JSON |
| `reborn_doc/56_BL6_1_MultiSeed_Repeat_执行记录.md` | 本文件 |

## 3. 三 seed 训练结果

| Seed | AUROC | AUPRC | Δ vs BL5 | Above BL5? | best_epoch | train_time |
|:---|---:|---:|---:|:---:|:---:|---:|
| 42 | 0.984993 | 0.539917 | +0.008636 | ✅ | 8 | 172.5m |
| 43 | 0.986256 | 0.570872 | +0.039591 | ✅ | 8 | 171.6m |
| 44 | 0.983685 | 0.462466 | −0.068815 | ❌ | 9 | 172.8m |

## 4. AUPRC 统计

| Statistic | Value |
|:---|---:|
| AUPRC mean | 0.524418 |
| AUPRC sample std | 0.055840 |
| AUPRC min | 0.462466 |
| AUPRC max | 0.570872 |
| ΔAUPRC vs BL5 mean | −0.006863 |
| ΔAUPRC vs BL5 sample std | 0.055840 |
| n_above_BL5 | 2/3 |
| all_above_BL5 | ❌ False |

BL5-v4-PAM baseline AUPRC = 0.531281.

## 5. 结论

**BL6-1 multi-seed repeat shows mixed stability.** Seeds 42 and 43 are above the BL5-v4-PAM baseline, but seed44 is below it. The three-seed mean AUPRC (0.5244) is below the BL5 baseline (0.5313), and variance is large (sample std=0.0558). **Current evidence does not support promoting BL6-1 to the main model or claiming stable advantage.**

## 6. Gate caveat

Seed42 gate audit (Part 3) found near-collapse to LearnableRun. Seed43/44 gate behavior has **not** yet been audited. The gate collapse may or may not replicate across seeds.

## 7. 下一步建议

- **不升主模型** — BL6-1 不能替换 BL5-v4-PAM
- **不推进 BL6-2** — 在 seed44 failure 和 gate collapse 未解决前，不应推进更深层架构
- **可选**：seed43/44 gate export + audit，确认 gate collapse 是否跨 seed 复现
- **可选**：gate/head ablation，确认 seed42/43 gain 来源（gate MLP params vs head dimension vs 训练动力学）
- **可选**：回到 BL5-v4-PAM 作为当前主模型候选

## 8. 合规声明

- 所有训练均使用 formal_split_bl5_seed42.json ✅
- 未覆盖 seed42/seed43/seed44 任一目录 ✅
- 未启动 BL6-2 ✅
- 未改 data/reference ✅
- 未 commit/push ✅
