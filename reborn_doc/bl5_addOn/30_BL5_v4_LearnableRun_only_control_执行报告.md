# BL5-v4-LearnableRun-only-control 执行报告

> **版本**: BL5-v4-LearnableRun-only-control  
> **执行时间**: 2026-06-06  
> **训练时长**: 331.4s (~5.5 min)  
> **GPU**: 2× (CUDA 0,1), DDP  
> **Commit**: 327075a

---

## 1. 任务目标

执行 `reborn_doc/29_BL5_v4_LearnableRun_only_control_plan.md` 中的 Component-Level Ablation：

> **只保留 LearnableRunEncoder，关闭 RNA-FM，关闭 PAM Encoder**，测量单视图在 Formal Split 上的性能上限。

**核心科学问题**：LearnableRun 单视图能否超过 BL4-Run-only（AUPRC=0.206）？

---

## 2. 模型配置

| 配置项 | 值 | 说明 |
|:---|:---|:---|
| `use_rnafm` | `false` | **关闭 RNA-FM** |
| `freeze_rnafm` | `false` | 声明但不生效 |
| `fusion_type` | `run_only` | 仅使用 Run 特征 |
| `use_learnable_run` | `true` | 使用 LearnableRunEncoder |
| `use_pam_encoder` | `false` | **关闭 PAM** |
| `d_model` | 128 | LearnableRun 输出维度 |
| `mlp_hidden` | 256 | 分类器第一层 |
| `mlp_hidden2` | 64 | 分类器第二层 |
| `dropout` | 0.3 | Run encoder dropout |
| `dropout2` | 0.2 | 分类器第二层 dropout |
| `focal_gamma` | 2.0 | Focal loss |
| `batch_size` | 1024 | 2 GPU 各 512 |
| `epochs` | 10 | 固定 epoch |
| `lr_run_encoder` | 1e-3 | Run 编码器学习率 |
| `lr_mlp` | 1e-3 | MLP 学习率 |

**参数量**: ~41K（LearnableRunEncoder + MLP），无 RNA-FM（99.5M 未加载）。

---

## 3. 数据集与划分

- **数据来源**: CCLMoff 完整数据集（6,393,373 条）
- **划分方式**: `sgrna_safe` group split via `formal_split_bl5_seed42.json`
- **划分规模**:

| 集合 | 样本数 | 正样本 | 正样本率 | sgRNA 类型数 |
|:---|---:|---:|---:|---:|
| Train | 4,697,495 | 31,994 | 0.681% | 150 |
| Val | 741,552 | 4,268 | 0.576% | 60 |
| Test | 954,326 | 3,057 | 0.320% | 72 |

---

## 4. 训练过程

| Epoch | Train Loss | Val AUROC | Val AUPRC | Val Precision | Val Recall | Val F1 |
|:---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.007087 | 0.9304 | 0.3464 | 0.7387 | 0.2067 | 0.3230 |
| 2 | 0.005082 | 0.9339 | **0.3929** | 0.6213 | 0.3233 | 0.4253 |
| 3 | 0.004715 | 0.9305 | 0.3839 | 0.5530 | 0.3383 | 0.4198 |
| 4 | 0.004560 | 0.9284 | 0.4050 | 0.4540 | 0.4222 | 0.4375 |
| 5 | 0.004419 | 0.9226 | 0.3935 | 0.5368 | 0.3739 | 0.4408 |
| 6 | 0.004331 | 0.9253 | 0.3764 | 0.5100 | 0.3636 | 0.4246 |
| 7 | 0.004242 | 0.9278 | 0.3906 | 0.4813 | 0.3887 | 0.4301 |
| 8 | 0.004201 | 0.9374 | 0.3932 | 0.6626 | 0.3018 | 0.4147 |
| **9** | 0.003983 | **0.9403** | **0.4156** | 0.5507 | 0.3805 | **0.4500** |
| 10 | 0.003911 | 0.9336 | 0.4123 | 0.5820 | 0.3676 | 0.4506 |

- **Best epoch**: 9（Val AUPRC=0.4156）
- **学习率变化**: epoch 1-7 保持 1e-3，epoch 8-10 ReduceLROnPlateau 降至 5e-4

---

## 5. Test 评估结果（best.pt，epoch 9）

| 指标 | 值 | 说明 |
|:---|---:|:---|
| **AUROC** | **0.9609** | 排序能力优秀 |
| **AUPRC** | **0.2949** | 正样本检索能力 |
| Accuracy | 0.9966 | 阈值 0.5 |
| Precision | 0.4378 | 阈值 0.5 |
| Recall | 0.2669 | 阈值 0.5 |
| F1 | 0.3316 | 阈值 0.5 |
| Test Loss | 0.0040 | BCE + Focal |

---

## 6. 与历史 Baseline 对照

> **口径说明**：以下 formal BL5 split 结果均基于 `formal_split_bl5_seed42.json`，test set 完全对齐（954,326 条，72 sgRNA 类型）。旧 split 结果（如 BL0b-finetune-v1=0.522）因划分不同，**不参与严格消融对比**，仅作参考。

| 版本 | Test AUPRC | Test AUROC | 参数量 | RNA-FM | PAM | Run 编码器 | Split 口径 |
|:---|---:|---:|---:|:---:|:---:|:---|:---|
| **BL0a** | 0.073 | — | 41K | ❌ Frozen | ❌ | — | 旧 split |
| **BL0b-on-BL5split** | **0.2957** | — | ~99.5M | ✅ Fine-tune | ❌ | — | **formal** |
| **BL4-Run-only** | 0.206 | — | ~99.5M+ | ✅ Fine-tune | ❌ | Run CNN | 旧 split |
| **BL5-v4-NoPAM-control** | 0.5024 | — | ~99.5M+ | ✅ Fine-tune | ❌ | LearnableRun | **formal** |
| **BL5-v4-PAM** | 0.5313 | 0.9855 | ~99.5M+ | ✅ Fine-tune | ✅ | LearnableRun | **formal** |
| **BL6-1** | 0.5399 | 0.9850 | ~99.5M+ | ✅ Fine-tune | ✅ (Gated) | LearnableRun | **formal** |
| **BL5-v4-LearnableRun-only** | **0.2949** | **0.9609** | **~41K** | ❌ **关闭** | ❌ **关闭** | **LearnableRun** | **formal** |

### 关键发现

1. **vs BL0b-on-BL5split (0.2957)**：
   - 本版本 AUPRC **0.2949 ≈ 0.2957**，差距仅 **-0.0008**
   - **结论**：在 formal BL5 split 上，LearnableRun 单视角几乎达到 CCLMoff-style RNA-FM fine-tune baseline 的水平。这说明 LearnableRun 编码的连续错配先验具有极强的独立预测能力。

2. **vs BL5-v4-NoPAM-control (0.5024)**：
   - RNA-FM + LearnableRun 组合后 AUPRC 从 ~0.295 跃升至 **0.5024**，提升 **+0.207**
   - **结论**：RNA-FM 与 LearnableRun 之间存在**显著互补性**。二者单独都约为 0.295，但组合后远超各自单独使用，说明两种视角捕捉了不同的决策边界信息。

3. **vs BL4-Run-only (0.206)**（legacy，旧 split，仅作参考）：
   - 本版本 AUPRC **0.2949 > 0.206**
   - **参考结论**：提示 LearnableRun 是更强的 Run prior 实现，但该对比不如同一框架内的严格消融，解释时应作为参考证据。

4. **vs BL0a (0.073)**（旧 split，仅作参考）：
   - 本版本 AUPRC **0.2949 >> 0.073**
   - **结论**：Run 特征（连续错配状态）单独使用就远超 frozen RNA-FM 的基线

---

## 7. Val-Test 泛化 gap 分析

- **Val AUPRC (best)**: 0.4156
- **Test AUPRC**: 0.2949
- **Gap**: 0.1207 (-29%)

**可能原因**：
1. Val 集正样本率（0.576%）高于 Test（0.320%），Val 集更容易预测
2. 模型在 150 个 train sgRNA 上学习，对 72 个 unseen test sgRNA 泛化存在挑战
3. ~41K 参数的轻量模型记忆能力有限，难以覆盖全部 sgRNA 特异性模式

---

## 8. 运行时审计

| 项目 | 值 |
|:---|:---|
| 训练总时间 | 331.4s (~5.5 min) |
| GPU 峰值显存 | 0.10GB |
| 单 epoch 时间 | ~30s |
| RNA-FM 加载 | **未加载** ✅ |
| DDP | 2 GPU，正常 |
| Guardrails | 全部通过 ✅ |

**重要确认**：训练仅耗时 ~5.5 分钟，远低于 RNA-FM fine-tune 的 ~3 小时，确认 RNA-FM 确实被关闭。

---

## 9. 科学结论

1. **LearnableRunEncoder 单视图 AUPRC=0.295**，远超 random baseline（~0.0032）和 BL0a（0.073），证明 Run 特征（连续错配状态）具有强大的独立预测能力。

2. **LearnableRun 单视角 ≈ BL0b-on-BL5split**：在 formal BL5 split 上，LearnableRun-only（0.2949）与 RNA-FM fine-tune baseline（0.2957）几乎持平。这是本实验最漂亮的发现——**一个仅 41K 参数的 hand-crafted prior 编码器，达到了 99.5M 参数 RNA-FM fine-tune 的同等水平**。

3. **RNA-FM 与 LearnableRun 强互补**：二者单独都约为 0.295 AUPRC，但组合的 BL5-v4-NoPAM-control 达到 0.5024，提升超过 70%。这说明两种视角捕捉了不同的决策边界信息，融合后产生了显著的协同效应。**不能将 NoPAM 与任一单视角 baseline 的差值简单解释为另一个组件的“纯贡献”**，因为 classifier/head 和特征交互也参与其中。

4. **LearnableRun > Run CNN（参考）**：相比 legacy BL4-Run-only（0.206），本版本明显更高，提示可学习位置嵌入优于固定 CNN 编码；但该对比为跨框架参考，不作为严格消融结论。

---

## 10. 后续建议

1. **学习率调优**：尝试更大的 `lr_run_encoder`（如 3e-3）或更长的 warmup，可能提升收敛
2. **早停**：当前 10 epoch 固定，val AUPRC 在 epoch 9 达到峰值后 test 性能未继续提升，建议设置 `early_stopping_patience=3`
3. **Dropout 调优**：当前 dropout=0.3，val-test gap 较大，尝试更大的 dropout（0.4-0.5）可能改善泛化
4. **d_model 扩展**：尝试 d_model=256 或 512，增加模型容量以捕捉更复杂的 Run 模式

---

## 11. 文件与产物

| 文件 | 路径 |
|:---|:---|
| Config | `configs/bl5_v4_learnablerun_only_control.yaml` |
| Smoke Config | `configs/bl5_v4_learnablerun_only_control_smoke.yaml` |
| Run Script | `run/run_bl5_v4_learnablerun_only_control_2gpu.sh` |
| Best Checkpoint | `results/bl5_v4_learnablerun_only_control/checkpoints/best.pt` |
| Epoch Metrics | `results/bl5_v4_learnablerun_only_control/epoch_metrics.csv` |
| Test Predictions | `results/bl5_v4_learnablerun_only_control/test_predictions.csv` |
| Summary JSON | `results/bl5_v4_learnablerun_only_control/summary.json` |
| 模型代码 | `models/bl5_dynamic_fusion.py`（新增 `run_only` fusion_type） |
| 训练脚本 | `scripts/train_bl5.py`（新增 `use_rnafm=false` 支持） |
