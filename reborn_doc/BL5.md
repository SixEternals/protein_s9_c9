# BL5 历史归档

> ⚠️ **过期文档**：本文档整合自 `21_公平基线重置与PAM贡献拆解报告.md`、`22_bl5_shuffle.md`、`23_BL5_shuffle_control_交接报告.md`、`24_BL5_v4_PAM_封口路线与后续规划.md`、`25_BL6_based_on_BL5_v4_PAM_plan.md`、`29_BL5_v4_LearnableRun_only_control_plan.md`、`5_28p0到bl5的现状.md` 等历史文件。信息已部分过时，**仅供重整文档时参照**。最新项目状态请以 `reborn_doc/todo.md`、`38_BL5_v4_PAM_组件消融总报告.md`、`39_BL5_v4_PAM_泛化证据总报告.md` 为准。
> ❌ **禁止直接引用**：`reborn_doc/过期/` 路径下的所有文件均不可作为当前事实依据。

---

## 1. BL5 的定位

BL5 是 **Three-View Late Gated Fusion**（实际上是 simple concat + 可选 gate），回答的核心问题：
> RNA-FM + LearnableRun + PAM 三视角融合是否优于单视角？PAM 的贡献是真实的吗？

根据 AGENTS.md 第 18 章，BL5 系列包含：
- BL5-0~3：原始定义（Cross-Attn + Gated Fusion）
- **BL5-v4-PAM**：实际最强模型（simple concat + PAM Encoder）

**当前项目主结果 = BL5-v4-PAM。**

---

## 2. 关键结果（已修正为最新信息）

### 2.1 Formal BL5 split 统一 test cohort

| 指标 | 数值 |
|:---|---:|
| split_file | `formal_split_bl5_seed42.json` |
| split_mode | `sgrna_safe` |
| test_samples | **954,326** |
| test_observed_positive | **3,057** |
| test_unobserved_candidate | **951,269** |
| test_sgRNA_type_count | **72（全部 unseen）** |
| positive_rate | 0.3203% |

> 所有 BL5/BL6 实验均在此同一 test set 上评估，best.pt 加载。

### 2.2 完整组件消融矩阵（formal split, seed=42）

| 实验 | RNA-FM | LearnableRun | PAM | Gate | AUROC | AUPRC |
|:---|:---:|:---:|:---:|:---:|---:|---:|
| BL0b-on-BL5split | ✅ | ❌ | ❌ | ❌ | 0.8578 | 0.2957 |
| LearnableRun-only | ❌ | ✅ | ❌ | ❌ | 0.9609 | 0.2949 |
| PAM-only | ❌ | ❌ | ✅ | ❌ | 0.4994 | 0.0592 |
| NoPAM-control | ✅ | ✅ | ❌ | ❌ | 0.9841 | 0.5024 |
| RNAFM-PAM-noRun | ✅ | ❌ | ✅ | ❌ | 0.8380 | 0.2765 |
| LearnableRun-PAM-noRNAFM | ❌ | ✅ | ✅ | ❌ | 0.9527 | 0.1772 |
| **BL5-v4-PAM（历史最佳）** | ✅ | ✅ | ✅ | ❌ | **0.9842** | **0.5313** |
| PAM-shuffle-control | ✅ | ✅ | ⚠ shuffled | ❌ | 0.6697 | 0.1389 |
| BL6-1-PAM-Gated-Fusion | ✅ | ✅ | ✅ | ✅ | 0.9850 | 0.5399 |

> 来源：`38_BL5_v4_PAM_组件消融总报告.md`。这是当前最权威的消融总表。

### 2.3 核心发现（已整合并更正）

1. **RNA-FM + LearnableRun 是主性能来源**：单独均约 0.29，融合跃升至 0.50（+70%）
2. **PAM 单独几乎无价值**（0.0592），但在 RNA-FM+Run 强联合上下文中贡献 **+0.0289**
3. **PAM 增益依赖正确对应关系**：shuffle 后崩溃至 0.1389
4. **BL6-1 gate 是微调增益**（+0.0086），single-run，需 multi-seed 确认

### 2.4 历史 BL5 子版本（非当前主线）

| 版本 | 核心特点 | test AUPRC | 状态 |
|:---|:---|---:|:---|
| BL5-3 | Cross-Attn + Gated, hand-crafted Run | 0.4452 | 旧路线，非主线 |
| BL5-3-LearnableRun | Cross-Attn + Gated + LearnableRun | 0.5180 | 旧路线，非主线 |
| BL5-v3-CLS | CLS + LearnableRun, no PAM | 0.4836 | 旧路线，非主线 |
| BL5-3-v2 Simple Backend | mean pool concat, no Cross-Attn | 0.4842 | 旧路线，非主线 |
| BL5-3-LearnableRun-Reg | Cross-Attn + Gated + Reg | 0.4500 | epoch 4 NaN, recovered |

> 这些版本证明：复杂的 Cross-Attn/Gated 路线不如 simple concat + PAM（BL5-v4-PAM）。因此 BL5-v4-PAM 成为实际主模型。

### 2.5 封口路线文档（24_...）的历史状态

`24_BL5_v4_PAM_封口路线与后续规划.md` 当时的决策：
- ❌ 不进 BL6
- ❌ 不继续堆 BL5 新架构
- ✅ 优先封口 BL5-v4-PAM（per-sgRNA、bootstrap、top-k、kNN、in-silico）

> **该决策已被后续工作覆盖**：BL6-1 已跑、AGG holdout 已完成、TGG holdout 正在进行中。文档中的"当前不要开 BL6"不再反映最新状态。

---

## 3. 工程记录（23_交接报告）

BL5 训练过程中修复的关键 bug：
1. **DDP 死锁**：`predict_probabilities` 被 `is_main_process` 保护，但内部 `all_gather_object` 需要所有 rank 参与
2. **RNA-FM unused parameters**：`contact_head` / `lm_head` 在 `return_contacts=False` 时不被使用，导致 DDP `find_unused_parameters=True` 偶发 NCCL 超时
3. **PAM shuffle 实现**：从 intra-batch shuffle 升级为 within_split shuffle（train/val/test 各自独立 seed 置换）

---

## 4. 重要修正

原 `5_28p0到bl5的现状.md` 和 `24_...` 中多处使用 `off_seq[-3:]` 提取 PAM，**已修正为**：

```python
PAM_original = off_seq[20:23]  # positions 21-23
```

> 禁止在新增代码中使用 `off_seq[-3:]`。

---

## 5. 原始文档索引

| 原始文件 | 时间 | 核心内容 |
|:---|:---|:---|
| `21_公平基线重置与PAM贡献拆解报告.md` | 2026-05-29 | 公平基线重置动机、PAM shortcut 信号发现、NoPAM 进行中状态 |
| `22_bl5_shuffle.md` | 2026-06-03 | shuffle control 执行指令 |
| `23_BL5_shuffle_control_交接报告.md` | 2026-06-03 | shuffle 结果、代码修改记录（DDP 死锁、unused params、shuffle 实现） |
| `24_BL5_v4_PAM_封口路线与后续规划.md` | 2026-06-05 | 封口决策（不进 BL6、不堆架构）、证据链、AUROC/AUPRC 科普 |
| `25_BL6_based_on_BL5_v4_PAM_plan.md` | 2026-06-05 | BL6 plan（以 BL5-v4-PAM 为 backbone） |
| `29_BL5_v4_LearnableRun_only_control_plan.md` | 2026-06-06 | LearnableRun-only control 计划 |
| `5_28p0到bl5的现状.md` | 2026-06-04 | BL5 各子版本技术关键词、结果汇总 |
