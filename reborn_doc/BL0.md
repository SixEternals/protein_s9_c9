# BL0 历史归档

> ⚠️ **过期文档**：本文档整合自 `03_P1_RNAFM接入执行汇报.md`、`04_BL0a正式训练运行说明.md`、`05_BL0b微调运行说明.md`、`5_28p0到bl5的现状.md` 等历史文件。信息已部分过时，**仅供重整文档时参照**。最新项目状态请以 `reborn_doc/todo.md` 为准。
> ❌ **禁止直接引用**：`reborn_doc/过期/` 路径下的所有文件均不可作为当前事实依据。

---

## 1. BL0 的定位

BL0 是 **CCLMoff 复现基线**，回答的核心问题：
> 在同一份 CCLMoff CSV、同一个 `sgRNA_type_group` 严格切分下，RNA-FM frozen vs fine-tune 差距有多大？

BL0 **不包含** Region encoder、Run encoder、Cross-Attention 或 Gated Fusion。它只是 RNA-FM + CCLMoff-style MLP head。

---

## 2. 关键结果（已修正为最新信息）

### 2.1 BL0a / frozen RNA-FM + MLP head

| 项目 | 数值 |
|:---|:---|
| 数据集 | CCLMoff 6,393,373 行 |
| split | `sgRNA_type_group` / `sgrna_safe` |
| freeze_rnafm | true |
| trainable params | 41,089 |
| epochs | 5 |
| batch_size | 256 |
| **test AUROC** | **0.8406** |
| **test AUPRC** | **0.0727** |

> frozen RNA-FM 单独性能极弱（AUPRC≈0.07），但此结果在后续 formal split 上未被直接复用。历史 BL0a 使用的是另一套 split。

### 2.2 BL0b / fine-tune RNA-FM + MLP head

| 项目 | 数值 |
|:---|:---|
| freeze_rnafm | false |
| trainable params | ~99.56M |
| epochs | 10 |
| batch_size | 128 |
| **test AUROC（旧 split）** | **0.9745** |
| **test AUPRC（旧 split）** | **0.5223** |
| **test AUROC（formal BL5 split）** | **0.8578** |
| **test AUPRC（formal BL5 split）** | **0.2957** |

> 关键发现：fine-tune 相对 frozen 提升 **+615%**（0.073 → 0.522）。但 formal BL5 split（72 unseen sgRNA_type）上 AUPRC 降至 0.296，说明严格 split 大幅增加了难度。

### 2.3 RNA-FM Tokenizer 关键发现

CCLMoff 公开代码使用 `sgRNA + "<sep>" + off_seq`，但当前 `rna-fm==0.2.2` 的 alphabet **没有原生 `<sep>` token**。项目实际策略：
- `alphabet.get_idx("<sep>") == alphabet.unk_idx`
- 将 `"<sep>"` 作为**单个 `<unk>` delimiter** 处理
- 这比默认 batch converter 更安全（后者会把 `"<sep>"` 拆成多个字符）

> 该发现至今仍然有效，未找到带 `<sep>` 的 matching alphabet。

---

## 3. 关键技术约定（沿用至今）

- `use_rnafm: true/false` 必须显式声明
- `freeze_rnafm: true/false` 必须显式声明（当 `use_rnafm=true` 时）
- `split_mode: random/sgrna_safe/loo` 必须显式声明
- test 评估必须加载 `best.pt`（val AUPRC 最佳）
- 必须同时报告 AUROC 和 AUPRC

---

## 4. 原始文档索引

| 原始文件 | 时间 | 核心内容 |
|:---|:---|:---|
| `03_P1_RNAFM接入执行汇报.md` | 2026-05-25 | RNA-FM 接入、`<sep>` 发现、smoke test |
| `04_BL0a正式训练运行说明.md` | 2026-05-26 | BL0a frozen 训练运行说明 |
| `05_BL0b微调运行说明.md` | 2026-05-26 | BL0b fine-tune 训练运行说明 |
