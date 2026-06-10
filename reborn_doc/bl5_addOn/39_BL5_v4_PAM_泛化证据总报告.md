# 39. BL5-v4-PAM 泛化证据总报告

> 📅 2026-06-08 ｜ 🎯 整理 BL5-v4-PAM 所有泛化证据，明确哪些已做、哪些未做、哪些不可做
>
> **性质**：跨实验泛化证据总结，不混淆 feasibility audit 和 training/evaluation result。未训练模型、未运行推理、未调用 GPU。

---

## 1. 泛化维度全景

| 泛化维度 | 当前状态 | 是否训练/评估 | 关键证据 | 当前结论 | 下一步 |
|:---|:---:|:---:|:---|:---|:---|
| **Same-dataset held-out test** | ✅ 完成 | 是（训练+评估） | BL5-v4-PAM AUPRC=0.5313 on unseen sgRNA (72 types), test=954,326 | 模型在 unseen sgRNA 上有强排序能力 | 维持为 formal test baseline |
| **Cross-PAM feasibility audit** | ✅ 完成 | 否（仅 audit） | feasible=5 (AGG/GGG/TGG/CGG/GAG), marginal=6, infeasible=51 | CCLMoff formal split 上有 5 个 feasible PAM holdout 候选 | AGG 已做，TGG 进行中，GAG 等后续 |
| **AGG strict PAM-holdout** | ✅ 完成 | 是（训练+评估） | PAM AUPRC=0.0276, NoPAM AUPRC=0.2038, Δ=+0.1762 (NoPAM better) | **显式 PAM 编码器在 unseen AGG 上严重负迁移** | 需 TGG/GAG 复验确认是否普遍 |
| **TGG strict PAM-holdout** | 🔄 进行中 | Kimi 正在跑 | — | — | **不要碰，等待 Kimi 完成** |
| **External dataset feasibility** | ✅ 完成 | 否（仅 audit） | ready_for_strict_external_eval=0 | CCLMoff 仓库内无独立外部数据集可做严格 cross-dataset eval | future work |
| **Cross-cell-line** | ❌ 不可做 | 否 | CCLMoff Method/Length 大量为空（~328 万行），无可靠细胞系元数据 | 不可做有意义的细胞系分层 | 等待合适数据 |
| **Cross-species** | ❌ 不可做 | 否 | 无可靠跨物种数据 | future work | — |

---

## 2. Same-Dataset Formal Test

### 2.1 测试集定义

| 参数 | 值 |
|:---|:---|
| Split 方式 | `sgrna_safe`（formal_group_json） |
| Split JSON | `formal_split_bl5_seed42.json` |
| Test 样本 | 954,326 |
| Test observed_positive | 3,057 |
| Test unobserved_candidate | 951,269 |
| Test positive_rate | 0.00320 |
| Test sgRNA_type | 72（全部在训练集和验证集中未出现） |

> 这是当前所有 BL5/BL6 实验的统一评估集。所有 AUROC/AUPRC 均在此 test set 上计算，best.pt 加载。

### 2.2 主模型 Test 性能

| 模型 | AUROC | AUPRC | 说明 |
|:---|---:|---:|:---|
| BL0b-on-BL5split | 0.8578 | 0.2957 | RNA-FM-only baseline |
| BL5-v4-NoPAM-control | 0.9841 | 0.5024 | 无 PAM |
| **BL5-v4-PAM** | **0.9842** | **0.5313** | BL5 anchor（历史最佳） |
| BL5-v4-PAM-shuffle-control | 0.6697 | 0.1389 | PAM 对应关系破坏 |
| BL6-1-PAM-Gated-Fusion | 0.9850 | 0.5399 | Gate variant (single-run) |

### 2.3 泛化含义

- Unseen sgRNA_type = 72，意味着 test sgRNA 在训练时完全不可见。
- 这不是 random split（会导致信息泄漏），是 group-safe split。
- 当前结果证明模型对**未见过的 sgRNA** 有一定泛化能力。
- 但这仍然是 **same-dataset** 泛化——train/val/test 都来自 CCLMoff，细胞系、实验条件、数据分布相同。

---

## 3. Cross-PAM Generalization

### 3.1 Feasibility Audit（已完成，不训练模型）

详见：`results/bl5_generalization/pam_holdout_feasibility/pam_holdout_feasibility_report.md`

| 结果 | 详情 |
|:---|:---|
| Feasible holdout candidates | **5**: AGG (744 pos), GGG (716 pos), TGG (703 pos), CGG (186 pos), **GAG (111 pos)** |
| Marginal | **6**: CAG, AAG, ATG, TAG, GTG, TTG |
| Infeasible | **51**（含 GCG/ACG/CCG/TCG 等 positive<20 的 motif，以及 GG 等 noncanonical motif） |
| 推荐 | AGG 优先（positive 最多，统计功效最好），TGG 次之（样本最多），GAG 为 non-NGG exploratory |

### 3.2 AGG Strict Holdout Result（已完成，训练+评估）

详见：`reborn_doc/36_BL5_v4_PAM_Holdout_AGG_执行报告.md`

| 指标 | PAM holdout AGG | NoPAM holdout AGG | Δ (NoPAM − PAM) |
|:---|---:|---:|---:|
| Test AUROC | 0.4797 | 0.9456 | **+0.4659** |
| Test AUPRC | 0.0276 | 0.2038 | **+0.1762** |
| Test 样本 | 277,247 | 277,247 | same test_H |
| Test pos | 744 | 744 | — |

> 🎯 **核心发现**：在 AGG 被完全排除出 train/val 后，显式 PAM 编码器在 test_H（全为 AGG）上**严重负迁移**：AUROC 从 0.946（NoPAM）降至 0.480（PAM），AUPRC 从 0.204 降至 0.028（仅为 NoPAM 的 1/7.4）。
>
> **解释**：PAM 编码器在训练时从未见过 AGG 的 PAM motif 向量，当遇到 AGG 时无法正确编码，反而输出错误信号干扰了 RNA-FM + LearnableRun 的联合判断。NoPAM 模型不依赖显式 PAM 向量，通过 RNA-FM 的隐式序列上下文反而能保持更好的泛化能力。

**解释边界**：
- ⚠️ 这只是一个 PAM motif（AGG）的单点结果，**不能断言所有 unseen PAM 都会负迁移**。
- ⚠️ TGG（当前 Kimi 在进行中）和 GAG 的复验是必需的：如果 TGG 也显示同样趋势，则「显式 PAM 编码器对 unseen PAM 泛化能力弱」的结论会大大加强。
- ⚠️ seen-PAM test set（test_nonAGG）的 sanity check 未完成——需要验证 PAM 和 NoPAM 在 seen-PAM 上性能相近，以排除「PAM 模型在 AGG 上崩溃是因为整体训练失败」的可能性。

**下一步**：
1. 🔄 TGG strict holdout（Kimi 进行中）— 作为第二个 NGG motif 复验
2. ⏳ GAG strict holdout — 唯一 feasible non-NGG，但 positive_ratio 较高（0.012），需谨慎解读
3. ⏳ test_nonAGG sanity check for AGG holdout
4. ⏳ GGG / CGG holdout（优先级低于 AGG/TGG/GAG）

---

## 4. Cross-Dataset Feasibility

### 4.1 Audit 结论（已完成，不训练模型）

详见：`reborn_doc/35_External_Dataset_Feasibility_Audit_执行报告.md`

| 结论 | 详情 |
|:---|:---|
| **Ready for strict external eval** | **0** 个候选 |
| Provenance-required limited candidates | 2（SITE-seq/K562，但需要确认数据来源和检测方法映射） |
| Smoke test only | 1（`test_20_samples.csv`——仅 20 条测试数据，只能验证 pipeline 连通性，不能评估模型性能） |
| Not recommended | 35 |

> 🎯 **核心结论**：CCLMoff 仓库内没有独立的外部数据集可做严格 cross-dataset generalization。所有 test_predictions.csv 都来自 CCLMoff 同一数据源，sgRNA/off_seq 与 CCLMoff train set 有不同程度重叠。现有外部数据（如 SITE-seq/K562）元数据不完整，不能作为严格独立外部 benchmark。

**解释边界**：
- ❌ 不能写「完成了 cross-dataset 泛化评估」
- ❌ 不能将 `test_20_samples.csv` 的 smoke test 当作正式 external eval
- ✅ 可以写「external dataset feasibility audit 显示当前无可用的严格外部独立数据集」

### 4.2 与 GUIDE-seq 历史结果的关系

GUIDE-seq P0/BL3 结果（AUPRC ~0.55-0.85）**不能**与 CCLMoff formal BL5 split 直接混比：
- 数据集不同（GUIDE-seq vs CCLMoff）
- Split 方式不同（旧 random split 有信息泄漏风险 vs 新 sgrna_safe split）
- 评估集不同（GUIDE-seq 测试集 vs CCLMoff formal test）
- 正负样本比例不同

如果需要跨数据集比较，必须在同一个严格外部 holdout 集上对所有模型重新评估。

---

## 5. Cross-Cell-Line / Cross-Species

| 维度 | 状态 | 原因 |
|:---|:---:|:---|
| Cross-cell-line | ❌ | CCLMoff `Method` 和 `Length` 字段有 ~328 万行空值，无法做可靠的细胞系分层 |
| Cross-species | ❌ | 无可用的跨物种数据 |

这两个维度列为 future work，等待合适数据的出现。

---

## 6. 泛化证据强度评级

| 泛化类型 | 证据强度 | 说明 |
|:---|:---:|:---|
| Same-dataset unseen sgRNA | ⭐⭐⭐⭐ | 强证据。72 个 unseen sgRNA_type，954k 样本，已多模型验证 |
| Same-dataset unseen PAM (AGG) | ⭐⭐⭐ | 中等证据。仅 1 个 motif 复验，TGG/GAG 正在/即将进行 |
| Cross-PAM feasibility | ⭐⭐⭐⭐⭐ | 强证据（audit 方法论成熟）。但不等于 cross-PAM 泛化评估 |
| Cross-dataset | ⭐ | 极弱/无证据。无可用独立外部数据集 |
| Cross-cell-line | — | 无证据。元数据缺失不可做 |
| Cross-species | — | 无证据。无可用数据 |

---

## 7. 解释边界总结

| ❌ 不能写 | ✅ 正确表述 |
|:---|:---|
| "BL5-v4-PAM 对 unseen PAM 泛化能力差" | "AGG strict holdout 显示 PAM 编码器在 unseen AGG 上严重负迁移。该结论仅针对 AGG，需要 TGG/GAG 复验后才能评估跨 PAM 的一般性" |
| "已完成跨数据集泛化评估" | "External dataset feasibility audit 显示当前仓库内无可用的严格独立外部数据集。跨数据集泛化列为 future work" |
| "模型可泛化到新细胞系" | "CCLMoff 元数据缺失无法支持细胞系分层，cross-cell-line 泛化未评估" |
| "Feasibility audit 证明了模型的泛化能力" | "Feasibility audit 仅判断数据是否支持实验，不评估模型性能" |
| "AGGG holdout 失败说明 PAM 编码器没有用" | "在 seen-PAM 上 PAM Encoder 贡献 +0.0289 AUPRC，在 AGG unseen PAM 上反而负迁移。PAM 编码器在 seen/unseen PAM 上表现不对称" |

---

## 8. 源码与结果索引

| 证据 | 来源 |
|:---|:---|
| Formal test 指标 | `results/experiments.csv`, 各 `results/*/summary.json` |
| PAM feasibility audit | `results/bl5_generalization/pam_holdout_feasibility/` |
| AGG holdout 训练结果 | `results/bl5_v4_pam_holdout_agg/`, `results/bl5_v4_nopam_holdout_agg/` |
| AGG holdout 执行报告 | `reborn_doc/36_BL5_v4_PAM_Holdout_AGG_执行报告.md` |
| External feasibility audit | `results/bl5_generalization/external_dataset_feasibility/` |
| External feasibility 报告 | `reborn_doc/35_External_Dataset_Feasibility_Audit_执行报告.md` |
| TGG holdout | `results/bl5_generalization/pam_strict_holdout/TGG/`（**Kimi 进行中，勿碰**） |

---

## 9. 合规说明

- 本文档仅整理已有结果和报告，未训练模型，未运行推理，未调用 GPU。
- 未删除或覆盖 data/ / reference/ 下任何文件。
- 未 commit / push。
- 未触碰 Kimi 正在进行的 TGG holdout 目录。
- PAM 坐标统一为 `off_seq[20:23]`（positions 21-23）。
- label=0 解释为 unobserved_candidate，不是 safe site。
- Feasibility audit ≠ 泛化训练结果；AGG 单点 ≠ 所有 unseen PAM 结论。
