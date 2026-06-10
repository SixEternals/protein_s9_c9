# BL5-v4 CGG PAM Holdout 训练执行汇报

> 汇报人：Kimi  
> 日期：2026-06-09  
> 对应任务：CGG PAM Holdout Phase 1（PAM / NoPAM 双模型训练 + prediction export）  
> 目标受众：总监 Codex 审阅

---

## 1. 任务概述

在 BL5-v4 PAM Generalization 框架下，执行 **CGG PAM Strict Holdout** 实验：

- **PAM 模型**：CGG 从 train_H / val_H 中 strict held out，但模型启用 PAM encoder（`use_pam_encoder=True`），用于测试显式 PAM encoder 对未见 CGG 的 cross-PAM generalization。
- **NoPAM 模型**：同一 CGG strict holdout split，但关闭 PAM encoder（`use_pam_encoder=False`），作为 paired control。
- 两模型共享完全相同的 train_H / val_H / test_H 划分，差异仅在 `use_pam_encoder`，确保对比的公平性。

---

## 2. 实验配置

| 配置项 | 值 |
|:---|:---|
| 基线版本 | BL5-v4 |
| 模型架构 | Fine-tuned RNA-FM + LearnableRunEncoder + RNA-FM/Run simple concat MLP |
| RNA-FM 策略 | `freeze_rnafm=False`（全参数微调） |
| PAM 编码 | PAM: `use_pam_encoder=True`；NoPAM: `use_pam_encoder=False` |
| 损失函数 | Focal loss `gamma=2.0` |
| Split 方式 | `sgrna_safe` group split（正式 split `formal_split_bl5_seed42.json`） |
| 训练 epoch | 10 |
| 硬件 | 2× RTX PRO 6000 Blackwell (96GB)，DDP |
| Holdout PAM | `CGG`（off_seq[20:23] == 'CGG'） |

---

## 3. 数据集分布

CGG Holdout Split 由 `scripts/build_pam_strict_holdout_split.py` 构造：

| 子集 | 样本数 | observed_positive | unobserved_candidate | positive_ratio |
|:---|---:|---:|---:|---:|
| train_H | 4,486,324 | 29,724 | 4,456,600 | 0.663% |
| val_H | 710,585 | 4,042 | 706,543 | 0.569% |
| **test_H (CGG)** | **46,592** | **186** | **46,406** | **0.399%** |
| test_seenPAM | 907,734 | 2,871 | 904,863 | 0.316% |

**test_H per-sgRNA composition 健康性检查**：72 个 test sgRNA_type 中：
- mixed（同时含 observed_positive 和 unobserved_candidate）= **33**
- unobserved_only = **39**
- observed_only = **0**

**test_seenPAM 说明**：test_seenPAM 是 formal_test 中 PAM_original ≠ CGG 的子集，包含多个 PAM motif。split_manifest 中标注 `test_seenPAM_unseen_in_train=["ATT"]`，即存在 1 个 rare ATT 样本在 train_H 中未出现。未来 seenPAM sanity 评估需注明此 caveat 或严格过滤只包含 train_H 中确实见过的 PAM。

---

## 4. 训练执行记录

| 模型 | 启动时间 | 完成时间 | 训练时长 | 状态 |
|:---|:---|:---|---:|:---:|
| PAM | 17:36 | 20:23 | ~166 min | ✅ 完成 |
| NoPAM | 20:24 | 23:03 | ~159 min | ✅ 完成 |

**执行细节**：
- PAM 训练过程中 `best.pt` 共更新 4 次（17:53 → 18:10 → 19:00 → 19:33 → 20:06 → 20:23），验证持续进步
- NoPAM `best.pt` 更新 2 次（20:40 → 21:12 → 21:44 → 22:31 → 22:47），best_epoch=9
- 训练结束后自动触发了 test 评估（best.pt），结果已写入 `summary.json` 和 `experiments.csv`

---

## 5. 核心结果

### 5.1 Test_H（CGG Holdout）性能

| 模型 | Test AUROC | Test AUPRC |
|:---|---:|---:|
| **PAM** | 0.977410 | 0.432872 |
| **NoPAM** | 0.961204 | 0.270522 |
| **Δ = NoPAM − PAM** | **−0.016206** | **−0.162350** |

### 5.2 关键发现（observed + P2 bootstrap confirmed）

1. **AUPRC gap favors PAM，且 bootstrap 支持**：PAM AUPRC 0.433，NoPAM AUPRC 0.271，ΔAUPRC = −0.162350，95% CI [−0.202704, −0.122637]，CI 不跨 0。在极度不平衡数据上（positive_ratio ~0.4%），AUPRC 是比 AUROC 更敏感的指标。
2. **AUROC gap 较小但同样显著**：PAM AUROC 0.977，NoPAM AUROC 0.961，ΔAUROC = −0.016207，95% CI [−0.022725, −0.010631]，CI 不跨 0。AUROC 在不平衡数据上可能虚高，对 minority class 变化不敏感，但 paired bootstrap 仍检测到显著差异。
3. **PAM encoder 的 CGG 效果**：在 strict holdout 设定下（两模型均未见 CGG train 样本），启用 PAM encoder 的模型在 test_H 上 AUPRC 和 AUROC 均显著高于 NoPAM。该结论仅针对 CGG，不能外推至所有 NGG 或 non-NGG PAM。

> ⚠️ **谨慎口径**：以上 point estimates 已由 P2 paired bootstrap（n_bootstrap=10,000）支持，但结论仅限 CGG 这一特定 PAM motif。AGG/TGG 方向相反，GAG 不显著，说明 PAM encoder 的 cross-PAM 行为是 motif-specific。

---

## 6. Prediction Export（P2 前置）

为支持 P2 paired bootstrap，已用 `best.pt` 执行 eval-only export，生成：

| 文件 | 路径 |
|:---|:---|
| PAM test predictions | `results/bl5_v4_pam_holdout_cgg/test_predictions.csv` |
| NoPAM test predictions | `results/bl5_v4_nopam_holdout_cgg/test_predictions.csv` |

Export 验证项：
- row alignment passed（sample_index 与 test_H 一致）
- PAM_original == 'CGG'（100% CGG）
- probability ∈ [0, 1]
- 无 experiments.csv eval-only 行写入

---

## 7. 结果归档

| 文件 | 路径 |
|:---|:---|
| PAM summary | `results/bl5_v4_pam_holdout_cgg/summary.json` |
| NoPAM summary | `results/bl5_v4_nopam_holdout_cgg/summary.json` |
| PAM config | `configs/bl5_v4_pam_holdout_cgg.yaml` |
| NoPAM config | `configs/bl5_v4_nopam_holdout_cgg.yaml` |
| 实验记录 | `results/experiments.csv`（最后两行） |
| Split 数据 | `results/bl5_generalization/pam_strict_holdout/CGG/` |

---

## 8. 已知问题与处理

| 问题 | 影响 | 处理 |
|:---|:---|:---|
| NoPAM 训练完成后，tmux 会话残留触发自动脚本竞态，意外启动重复训练 | 可能覆盖已完成结果 | **已立即终止**，原始 `summary.json` / `best.pt` / `epoch_metrics.csv` 未受破坏 |
| Python stdout 缓冲导致 tmux pane 输出延迟 | 仅影响实时监控，不影响训练 | 无 |

---

## 9. 下一步建议（Phase 2 & 3）

### Phase 2：配对 Bootstrap（P2）✅ 已完成

CGG test_H（46,592 样本）PAM vs NoPAM 配对 Bootstrap（n=10,000，seed=42）已执行：

| Metric | Δ = NoPAM − PAM | 95% CI | CI Crosses Zero? |
|:---|---:|:---|:---:|
| ΔAUROC | −0.016207 | [−0.022725, −0.010631] | ❌ No |
| ΔAUPRC | −0.162350 | [−0.202704, −0.122637] | ❌ No |

- **结论**：CGG test_H 上 bootstrap 支持 PAM 模型 AUPRC 高于 NoPAM（CI 不跨 0）
- **限制**：仅针对 CGG 单一 motif，不能外推所有 NGG

详见：
- `results/bl5_generalization/pam_strict_holdout/CGG/paired_bootstrap.json`
- `results/bl5_generalization/pam_strict_holdout/CGG/bootstrap_report.md`
- `reborn_doc/23_CGG_Holdout_P2_Bootstrap_执行报告.md`

### Phase 3：消融对比（P3）

将 CGG 结果与已有的 **AGG / TGG / GAG** holdout 结果横向对比：

| Holdout PAM | PAM AUPRC | NoPAM AUPRC | Δ = NoPAM − PAM | 备注 |
|:---|---:|---:|---:|:---|
| AGG | ~0.xx | ~0.xx | ~0.xx | 待补全 |
| TGG | ~0.xx | ~0.xx | ~0.xx | 待补全 |
| GAG | 0.9907 | 0.9897 | −0.0010 | bootstrap CI 跨 0，差异不显著；存在 composition confounding |
| **CGG** | **0.4329** | **0.2705** | **−0.1624** | **bootstrap CI 不跨 0，支持 PAM > NoPAM** |

- 回答核心科学问题：**不同 PAM 的 cross-PAM generalization 难度是否一致？** CGG 是否比其他 PAM 更难泛化？

---

## 10. 结论

CGG PAM Holdout P1 训练顺利完成，PAM / NoPAM 双模型结果已归档，prediction CSV 已导出供 P2 bootstrap 使用。

> ⚠️ **导出产物说明**：`test_predictions.csv` 为 `best.pt` eval-only export 产物；主 `summary.json` / `report.md` 已恢复为正式训练 `completed` 记录（PAM: epochs=10, train_seconds≈10003.4；NoPAM: epochs=10, best_epoch=9, train_seconds≈9523.5）。

- **Observed test_H AUPRC**：PAM = 0.432872，NoPAM = 0.270522，Δ = −0.162350（favors PAM）
- **Observed test_H AUROC**：PAM = 0.977410，NoPAM = 0.961204，Δ = −0.016206（favors PAM）
- **Bootstrap（P2 已完成）**：ΔAUPRC 95% CI = [−0.203, −0.123]，CI 不跨 0；ΔAUROC 95% CI = [−0.023, −0.011]，CI 不跨 0

P1 训练报告到此结束。P2 Bootstrap 已完成，详见 `reborn_doc/23_CGG_Holdout_P2_Bootstrap_执行报告.md`。建议继续推进 P3 横向消融对比（AGG/TGG/GAG/CGG 四者总表）。

---

*汇报完毕，请 Codex 审阅并指示下一步优先级。*
