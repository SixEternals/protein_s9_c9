# 53. BL6-1 Gate Export 执行记录

> Date: 2026-06-11  
> Phase: Part 2 — Eval-only Gate Weight Export  
> Executor: Claude  
> Plan: `reborn_doc/51_BL6_1_Gate_Audit_Report_Correction_Plan.md` §6  

---

## 1. 任务范围

本阶段仅做 eval-only gate weight export：用已有 BL6-1 `best.pt` 做单 GPU forward pass（不训练），导出每个 test 样本的 RNA-FM / Run / PAM 三路 gate weights。**未做 gate audit 解读**，那是 Part 3 的任务。

---

## 2. 脚本路径

`scripts/export_bl6_1_gate_predictions.py`

---

## 3. 输入

| 项目 | 路径 |
|:---|:---|
| Config | `configs/bl6_1_pam_gated_fusion.yaml` |
| Checkpoint | `results/bl6_1_pam_gated_fusion/checkpoints/best.pt` (epoch 8) |
| Test split | `formal_split_bl5_seed42.json` → test (954,326 rows) |
| CCLMoff CSV | `data/cclmoff/09212024_CCLMoff_dataset.csv` |
| CCLMoff NPZ | `data/cclmoff/cclmoff_9bit.npz` |
| RNA-FM | `data/rnafm/checkpoints/RNA-FM_pretrained.pth` |

## 4. 运行命令

```bash
CUDA_VISIBLE_DEVICES=0 /data/zwf/conda/envs/reborn_seed/bin/python \
  scripts/export_bl6_1_gate_predictions.py \
  --config configs/bl6_1_pam_gated_fusion.yaml \
  --checkpoint results/bl6_1_pam_gated_fusion/checkpoints/best.pt \
  --output results/bl6_1_pam_gated_fusion/gate_predictions.csv
```

## 5. 输出

| 文件 | 路径 |
|:---|:---|
| Gate predictions CSV | `results/bl6_1_pam_gated_fusion/gate_predictions.csv` |
| Validation JSON | `results/bl6_1_pam_gated_fusion/gate_export_validation.json` |

## 6. Gate 列含义

| 列 | 含义 |
|:---|:---|
| `gate_rnafm` | RNA-FM view weight（gate 第 0 列） |
| `gate_run` | LearnableRun view weight（gate 第 1 列） |
| `gate_pam` | PAM view weight（gate 第 2 列） |
| `gate_sum` | `gate_rnafm + gate_run + gate_pam`，应接近 1 |
| `gate_entropy` | `-sum(g * log(g + 1e-12))` |
| `gate_max` | `max(gate_rnafm, gate_run, gate_pam)` |
| `gate_argmax` | `rnafm` / `run` / `pam` |
| `pam_family` | `NGG` if `PAM_original[1:]=="GG"` else `non-NGG` |

## 7. Validation 结果（2026-06-11 返修后，AMP disabled）

> **返修说明**：V1 使用了 `torch.cuda.amp.autocast`，导致 probability 与原始 `test_predictions.csv` 的 max_abs_diff = 0.0051。根因是 `train_bl5.py::predict_probabilities()` 在原始 export 路径中**未使用 autocast**。返修将 `use_amp` 固定为 `False`，移除 autocast context，使 inference 路径严格匹配原始 prediction export。

| 检查项 | 结果 | 预期 |
|:---|:---:|:---|
| 输出行数 | 954,326 ✅ | 954,326 |
| observed_positive | 3,057 ✅ | 3,057 |
| unobserved_candidate | 951,269 ✅ | 951,269 |
| probability ∈ [0,1] | [3.56e-06, 1.0] ✅ | [0, 1] |
| gate_rnafm ∈ [0,1] | [0, 3.31e-04] ✅ | [0, 1] |
| gate_run ∈ [0,1] | [0, 1] ✅ | [0, 1] |
| gate_pam ∈ [0,1] | [1.39e-43, 1] ✅ | [0, 1] |
| max\|gate_sum - 1\| | 1.0e-07 ✅ | ≤1e-5 |
| PAM_original == off_seq[20:23] | 全部通过 ✅ | all |
| sample_index alignment | 完全一致 ✅ | match |
| sgRNA_type alignment | 完全一致 ✅ | match |
| off_seq alignment | 完全一致 ✅ | match |
| label alignment | 完全一致 ✅ | match |
| **prob max_abs_diff vs test_predictions.csv** | **≈ 0.0 (2.98e-08)** ✅ | ≤1e-5 |
| prob diff mean | ≈ 0.0 ✅ | — |
| prob diff median | ≈ 0.0 ✅ | — |
| rows with diff > 1e-5 | 0 ✅ | 0 |
| rows with diff > 1e-4 | 0 ✅ | 0 |

## 8. 未做事项

- 未做 gate audit 解读（gate 是否 collapse / 是否有意义 / 是否样本相关）
- 未写 `results/experiments.csv`
- 未覆盖 `test_predictions.csv`
- 未改 `summary.json` / `report.md`
- 未改 checkpoint
- 以上均为 Part 3（gate audit）的任务

## 9. 合规声明

- 未训练 ✅
- 仅 eval-only inference ✅
- 使用 best.pt ✅
- 未改 checkpoint ✅
- 未覆盖 test_predictions.csv ✅
- 未写 experiments.csv ✅
- 未改 data/ / reference/ ✅
- 未 commit / push ✅

## 10. 下一步

Part 3: 用 `gate_predictions.csv` 做 gate audit — 检查 gate 是否 collapse、按 label / PAM family / Top-K 分层的 gate behavior，写 `gate_audit.md`。
