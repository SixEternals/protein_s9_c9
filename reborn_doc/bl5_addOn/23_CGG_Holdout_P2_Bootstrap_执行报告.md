# BL5-v4 CGG Holdout P2 — Paired Bootstrap 执行报告

> 执行人：Kimi  
> 日期：2026-06-10  
> 对应任务：CGG PAM Holdout Phase 2（Paired Bootstrap）  
> 前置依赖：P1 训练已完成（PAM/NoPAM best.pt 就绪），prediction CSV 已由 P1 eval-only export 生成

---

## 1. 执行摘要

本任务为纯统计分析，**未训练任何模型**，仅对 P1 已导出的 prediction CSV 执行 paired bootstrap。

- **输入**：`results/bl5_v4_pam_holdout_cgg/test_predictions.csv` + `results/bl5_v4_nopam_holdout_cgg/test_predictions.csv`
- **方法**：paired row resampling，n = 10,000，seed = 42
- **Δ 定义**：NoPAM − PAM
- **输出**：`paired_bootstrap.json` + `bootstrap_report.md`

---

## 2. 合规声明

| 项目 | 状态 |
|:---|:---:|
| 未训练 | ✅ |
| 未 commit / push | ✅ |
| 未修改 data/ / reference/ | ✅ |
| 未修改 BL6 文件 | ✅ |
| 未新增 experiments.csv 行 | ✅ |
| 使用 best.pt 已导出 predictions | ✅ |
| Δ = NoPAM − PAM | ✅ |

---

## 3. Prediction 验证

| 检查项 | 结果 |
|:---|:---|
| PAM rows | 46,592 |
| NoPAM rows | 46,592 |
| Row alignment (sample_index, sgRNA_type, on_seq, off_seq, PAM_original, label) | ✅ Passed |
| PAM_original == off_seq[20:23] | ✅ Passed |
| test_H CGG count | 100% |
| Probability ∈ [0, 1] | ✅ Passed |
| observed_positive | 186 |
| unobserved_candidate | 46,406 |

**Point estimate 复算**：

| Model | AUROC | AUPRC |
|:---|---:|---:|
| PAM | 0.977410 | 0.432872 |
| NoPAM | 0.961204 | 0.270522 |
| Δ = NoPAM − PAM | −0.016207 | −0.162350 |

复算结果与 P1 summary 一致， proceed to bootstrap。

---

## 4. Bootstrap 结果

| Metric | Δ Point Estimate | 95% CI | CI Crosses Zero? |
|:---|---:|:---|:---:|
| ΔAUROC | −0.016207 | [−0.022725, −0.010631] | ❌ No |
| ΔAUPRC | −0.162350 | [−0.202704, −0.122637] | ❌ No |

- Valid bootstrap samples: 10,000 / 10,000
- `delta_AUROC_ci_crosses_zero`: false
- `delta_AUPRC_ci_crosses_zero`: false

---

## 5. 科学解释（谨慎措辞）

CGG test_H 上 paired bootstrap 支持 PAM 模型 AUPRC 高于 NoPAM：

- ΔAUPRC = −0.162，95% CI = [−0.203, −0.123]，CI 不跨 0
- ΔAUROC = −0.016，95% CI = [−0.023, −0.011]，CI 不跨 0

这意味着在 CGG strict holdout 设定下（两模型训练时均未见 CGG），启用 PAM encoder 的模型在 test_H 上 observed AUPRC 和 AUROC 均高于关闭 PAM encoder 的 paired control，且 bootstrap 差异在统计上稳定（CI 不跨 0）。

**限制**：
1. 本结论仅针对 CGG 这一个 PAM motif，不能外推到所有 NGG PAM。
2. CGG 是 AGG/TGG/GAG 之后的又一个 strict PAM-holdout data point。若与其他 holdout 结果方向不一致，则进一步支持 PAM encoder 的 cross-PAM generalization 是 motif-specific / subset-dependent。
3. 严格结论应等待 P3 横向总表（AGG/TGG/GAG/CGG 四者对比）后才能得出。

---

## 6. 产物清单

| 文件 | 路径 |
|:---|:---|
| Bootstrap JSON | `results/bl5_generalization/pam_strict_holdout/CGG/paired_bootstrap.json` |
| Bootstrap Report | `results/bl5_generalization/pam_strict_holdout/CGG/bootstrap_report.md` |
| PAM predictions | `results/bl5_v4_pam_holdout_cgg/test_predictions.csv` |
| NoPAM predictions | `results/bl5_v4_nopam_holdout_cgg/test_predictions.csv` |

---

## 7. Audit & Git 状态

- `audit_compliance.py`: ERROR=0 / WARNING=93
- `git status`: 无意外新增修改，未 commit / push
- `experiments.csv`: 仅保留 P1 两条 CGG completed 正式记录，无 P2 eval-only 行

---

## 8. 下一步

P2 已完成。建议推进 **P3** — AGG/TGG/GAG/CGG 四者横向总表与泛化总报告。
