# BL3 历史归档

> ⚠️ **过期文档**：本文档整合自 `5_28p0到bl5的现状.md` 等历史文件中关于 BL3 的部分。信息已部分过时，**仅供重整文档时参照**。最新项目状态请以 `reborn_doc/todo.md` 为准。
> ❌ **禁止直接引用**：`reborn_doc/过期/` 路径下的所有文件均不可作为当前事实依据。

---

## 1. BL3 的定位

BL3 是**先验特征验证**（无 RNA-FM），回答的核心问题：
> 显式生物先验（Region + Run + Seed 梯度）是否有效？hard/soft/learnable seed 哪个更好？

根据 AGENTS.md 第 18 章：**BL3 系列不包含 RNA-FM**。任何包含 RNA-FM 的模型必须从 BL4 起编号。

---

## 2. 关键结果

### 2.1 BL3-hard 系列（GUIDE-seq 数据集）

| 版本 | Seed 策略 | test AUROC | test AUPRC | best_epoch |
|:---|:---|---:|---:|---:|
| BL3-hard-A | Hard（1-15 weight=1, 16-20 weight=2） | 0.9894 | 0.5554 | 32 |
| BL3-hard-B | Soft tau=4 | 0.9922 | 0.5545 | 50 |
| BL3-hard-C | Learnable（20 个可学习权重） | 0.9908 | **0.5657** | 35 |

> 数据集：GUIDE-seq（520,281 样本），**不是 CCLMoff**。因此这些结果**不能**与 CCLMoff formal split 直接混比。

### 2.2 BL3 消融（GUIDE-seq）

| 消融项 | test AUPRC | 结论 |
|:---|---:|:---|
| Region-only | 0.5827 | — |
| Run-only | **0.6094** | Run > Region |
| Region+Run (BL3-hard-B) | 0.5545 | **Combined < Run-only** |

> 关键发现：**Run-only > Region-only >> Combined**。这是后续 BL5 放弃 Region、只保留 Run 的重要动机。

### 2.3 BL3 on CCLMoff

| 版本 | 数据集 | test AUROC | test AUPRC |
|:---|:---|---:|---:|
| BL3-cclmoff-runonly | CCLMoff | 0.9317 | 0.2257 |
| BL3.5-Full (Region+Run+Cross-Attn) | CCLMoff | 0.9731 | 0.2847 |
| BL3-Region-only-CCLMoff | CCLMoff | 0.9255 | 0.2071 |

> BL3.5-Full 是 CCLMoff 上的 R9+C9+Cross-Attn 尝试，AUPRC=0.285，远低于后续 BL5-v4-PAM（0.531）。

### 2.4 BL3b（Seed Regression）— 未跑

| 版本 | 状态 |
|:---|:---|
| BL3b-baseline | configured, no completed local result |
| BL3b-A (position embedding) | configured, no completed local result |
| BL3b-B (seed gate) | configured, no completed local result |
| BL3b-AB (both) | configured, no completed local result |

> BL3b 至今未正式跑完。根据 AGENTS.md 路线图，BL3-gradient 应在 BL4-full 之前完成，但目前优先级被 BL5/BL6 覆盖。

---

## 3. 原始文档索引

| 原始文件 | 来源位置 | 核心内容 |
|:---|:---|:---|
| `5_28p0到bl5的现状.md` | BL3 部分 | BL3-hard-A/B/C、消融、CCLMoff 版本、BL3b 配置 |
