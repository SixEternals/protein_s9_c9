# BL4 历史归档

> ⚠️ **过期文档**：本文档整合自 `5_28p0到bl5的现状.md` 等历史文件中关于 BL4 的部分。信息已部分过时，**仅供重整文档时参照**。最新项目状态请以 `reborn_doc/todo.md` 为准。
> ❌ **禁止直接引用**：`reborn_doc/过期/` 路径下的所有文件均不可作为当前事实依据。

---

## 1. BL4 的定位

BL4 是 **Frozen RNA-FM + Prior Concat**，回答的核心问题：
> 显式生物先验能否增强 frozen RNA-FM？

根据 AGENTS.md 第 18 章：
- BL4-Run-only：RNA-FM + Run（已跑）
- BL4-full：RNA-FM + Region + Run（**未实现**）

**BL4 包含 RNA-FM，因此不归入 BL3。**

---

## 2. 关键结果（已修正为最新信息）

### 2.1 BL4-Run-only / legacy BL3-RNAFM-Fusion

| 项目 | 数值 |
|:---|:---|
| freeze_rnafm | true |
| precomputed RNA-FM | true |
| Run 编码 | C9 Run features [B, 20, 9] |
| Seed 加权 | soft, tau=4.0 |
| concat | RNA-FM CLS [640] + Run CNN pooled [128] = 768 |
| **test AUROC** | **0.9585** |
| **test AUPRC** | **0.2056** |

> 注意：原 `summary.json` 使用 last.pt 指标，已修正为 best.pt 指标（0.2056）。

### 2.2 BL4-frozen

| 项目 | 数值 |
|:---|:---|
| freeze_rnafm | true |
| use_precomputed_rnafm | true |
| **test AUROC** | **0.9323** |
| **test AUPRC** | **0.0822** |
| best_epoch | 1 |

### 2.3 BL4-finetune

| 项目 | 数值 |
|:---|:---|
| freeze_rnafm | false |
| use_precomputed_rnafm | false（live RNA-FM forward） |
| **test AUROC** | **0.9827** |
| **test AUPRC** | **0.4899** |
| best_epoch | 9 |

> BL4-finetune（0.490）与后续 BL5-v4-PAM（0.531）之间仍有显著差距，说明 PAM + LearnableRun 的改进是有效的。

### 2.4 BL4-full

| 项目 | 状态 |
|:---|:---|
| 定义 | RNA-FM + Region + Run 三视角拼接 |
| 本地 config | ❌ 不存在 |
| 本地结果 | ❌ 不存在 |
| 状态 | **planned but not completed** |

> 根据 AGENTS.md 路线图，BL4-full 应在 BL5 之前完成。但当前项目主线已跳过至 BL5-v4-PAM，BL4-full 仍为待补项（低优先级）。

---

## 3. 重要修正

原 `5_28p0到bl5的现状.md` 中多处使用 `off_seq[-3:]` 提取 PAM，这是**错误的**。正确约定：

```python
PAM_original = off_seq[20:23]  # positions 21-23
```

> 该修正必须贯彻到所有新代码和报告中。`off_seq[-3:]` 在 23nt 序列中结果相同，但在 22/24nt 混合长度 CSV 中存在风险。

---

## 4. 原始文档索引

| 原始文件 | 来源位置 | 核心内容 |
|:---|:---|:---|
| `5_28p0到bl5的现状.md` | BL4 部分 | BL4-Run-only、BL4-frozen、BL4-finetune、BL4-full 状态 |
