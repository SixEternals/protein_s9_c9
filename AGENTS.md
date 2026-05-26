# AGENTS.md — AI 协作三层约束体系（Guardrails + 审计）

> 本文件约束所有在本项目上工作的 AI 助手：Codex / Kimi / Claude。
> 开始任何编码、重构、训练、评估或实验记录任务前，必须先阅读 `reborn_doc/1. 大纲拟定.md`，并确认理解本文的红色约束。

---

## 0. 三层约束体系

| 层级 | 文件 | 目的 |
|:---:|:---|:---|
| L1 文档层 | `AGENTS.md` | 统一 AI 交接规则、硬约束、术语、Git 纪律 |
| L2 代码层 | `utils/guardrails.py` | 在模型初始化、Run 编码、评估、指标报告处做运行时检查 |
| L3 审计层 | `scripts/audit_compliance.py`、`scripts/check_before_commit.py` | 扫描 `.py` 文件并在提交前阻断高风险违规 |

任何新模型、训练脚本、评估脚本都应在文件开头写明合规标记：

```python
"""
AGENTS.md compliance: [use_rnafm=True, freeze_rnafm=False,
                       split_mode=sgrna_safe, pos_weight=12]
确认本文件遵守 AGENTS.md 约束
"""
```

如果某个字段不适用，必须显式写出原因，例如 `use_rnafm=False` 表示该文件不使用 RNA-FM 视角，但仍需通过 config 显式声明。

---

## 1. <span style="color:red">🔴 绝对禁止（做任何任务前先看这个）</span>

1. 🔴 **禁止直接将原始字符串序列输入神经网络**：`sgRNA_seq` / `off_seq` 不能绕过 RNA-FM tokenization 或项目既定 R9/C9 编码后直接喂给 NN。
2. 🔴 **禁止隐式 RNA-FM 配置**：`use_rnafm` 必须在 config 中显式声明为 `true` 或 `false`；当 `use_rnafm=true` 时，`freeze_rnafm` 必须显式声明为 `true` 或 `false`，禁止默认值、禁止 `None`。
3. 🔴 **禁止 Run 编码跨入 PAM 位**：连续错配状态只在 positions `1-20` 计算，positions `21-23` PAM 必须单独编码。
4. 🔴 **禁止默认 split**：`split_mode` 必须在 config 中显式声明为 `random` / `sgrna_safe` / `loo` 之一。
5. 🔴 **禁止 class weighting 中 `pos_weight > 50`**：需要更强正样本关注时改用 focal loss（推荐 `gamma=2.0`）或重新审视采样策略。
6. 🔴 **禁止用 last checkpoint 做 test 评估**：test 必须显式加载验证集 AUPRC 最佳的 `best.pt`。
7. 🔴 **禁止只报告 AUROC 或只报告 AUPRC**：必须同时报告 AUROC、AUPRC，并注明 split 方式。
8. 🔴 **禁止假设 CCLMoff CSV 有完整元数据**：当前审计显示 `Method` / `Length` 存在大量空值，`sgRNA_type` 不是检测方法字段。
9. 🔴 **禁止复制目录做版本管理**：不得创建 `baseline_v2/`、`model_new/`、`experiments_backup/` 等目录保存版本差异。
10. 🔴 **禁止危险 Git 操作**：除非用户明确逐字确认，禁止 `git push --force` / `git push -f` / `git reset --hard` / `git rebase` / `git clean -f`。
11. 🔴 **禁止未经用户确认执行 `git commit` 或 `git push`**。
12. 🔴 **禁止删除或覆盖用户数据**：不得删除 `data/`、`results/`、`reference/` 下的数据，即使它们被 `.gitignore` 忽略。
13. 🔴 **多 AI 并行时禁止改动他人正在运行的 BL 文件**：例如用户说明 Kimi 正在跑 BL3A / BL3B 时，不得修改相关代码、配置、结果或运行脚本。

---

## 2. 任务开始前必须确认（必读）

开始任何涉及模型、训练、评估、数据预处理或实验记录的任务前，AI 必须确认：

1. 已读取 `reborn_doc/1. 大纲拟定.md`。
2. 已检查 `git status --short`，知道哪些文件是用户或其他 AI 的未提交工作。
3. 本次改动范围不会覆盖其他 AI 正在运行的 BL 代码或结果。
4. 所有序列特征模型都有明确的 tokenization / encoding 路径。
5. 本次 config 中显式声明了：
   - `use_rnafm: true/false`
   - 如果 `use_rnafm: true`，则 `freeze_rnafm: true/false`
   - `split_mode: random/sgrna_safe/loo`
   - 如使用 class weighting，`pos_weight <= 50`

---

## 3. 🔴 约束 #1：RNA-FM 必须接入或显式关闭

任何使用序列特征的模型都必须先经过 RNA-FM tokenization 或项目既定 R9/C9 编码流程，禁止直接将原始字符串 `sgRNA_seq` / `off_seq` 输入神经网络。

Config 中必须有：

```yaml
use_rnafm: true   # 使用 RNA-FM 视角
# 或
use_rnafm: false  # 不使用 RNA-FM，但必须显式说明
```

违规后果：模型无法学习序列上下文，AUPRC 可能接近随机（约 `0.07`）。

代码级检查：`utils.guardrails.check_model_config(config)`。

---

## 4. 🔴 约束 #2：`freeze_rnafm` 必须显式声明

如果 `use_rnafm=true`，则 `freeze_rnafm` 必须设为 `true` 或 `false`：

| 值 | 含义 |
|:---:|:---|
| `true` | frozen feature extractor，只训练 MLP head，约 41K 参数 |
| `false` | full fine-tune，RNA-FM 全部约 99.5M 参数参与训练 |

禁止默认值，禁止 `None`，禁止在代码里偷偷决定 frozen / fine-tune。

违规后果：frozen vs fine-tune 性能差异可达 615%（示例：`0.073` vs `0.522` AUPRC）。

代码级检查：`utils.guardrails.check_model_config(config)`。

---

## 5. 🔴 约束 #3：Run 编码限制 positions `1-20`

连续错配状态（`match` / `isolated` / `run2` / `run3+`）只在 protospacer positions `1-20` 计算。

PAM positions `21-23` 不参与 Run 状态计算，必须单独编码，例如 Region 编码中的 PAM 标记。

坐标约定：

| 区间 | 含义 |
|:---|:---|
| positions `1-20` | protospacer / guide-target aligned region |
| position `1` | PAM-distal end（最远端） |
| position `20` | PAM-proximal end（紧挨 PAM） |
| positions `21-23` | PAM region（NGG） |
| positions `16-20` | hard seed |

违规后果：C9 旧代码若跨 23 位编码，会让 PAM 位错误参与连续错配判断，导致 Region / Run 特征语义错乱。

代码级检查：`utils.guardrails.check_run_encoding_positions(run_states, max_pos=20)`。

---

## 6. 🔴 约束 #4：Split 方式必须显式声明

Config 中必须有 `split_mode` 字段，值只能是：

| `split_mode` | 含义 | 使用场景 |
|:---|:---|:---|
| `random` | 随机行划分，存在信息泄漏风险 | 仅 debug 或复现实验 |
| `sgrna_safe` | sgRNA-safe group split | 默认推荐，真实性能 |
| `loo` | Leave-One-sgRNA-Out | 最严格泛化测试 |

禁止无声明使用默认 split。禁止训练脚本通过 `cfg.get("split_mode", "random")` 之类代码默默回退。

违规后果：random split AUPRC 可虚高，例如 `0.769` vs group-safe `0.226`，差距约 3.4 倍。

代码级检查：`utils.guardrails.check_model_config(config)`。

---

## 7. 🔴 约束 #5：`pos_weight` 上限为 50

如果使用 class weighting，`pos_weight` 不得超过 `50`。

如果需要更大正样本权重，优先：

1. 改用 focal loss（推荐 `gamma=2.0`）。
2. 使用 `sqrt(n_neg / n_pos)` 级别的保守权重。
3. 重新检查 split、采样、标签分布和指标解释。

违规后果：`pos_weight=145` 可能导致 precision 崩溃至约 4.3%，AUPRC 解释失真。

代码级检查：`utils.guardrails.check_model_config(config)`。

---

## 8. 🔴 约束 #6：Test 评估必须用 best checkpoint

禁止用 last checkpoint 做 test 评估。

必须在训练过程中保存 validation AUPRC 最佳时的 checkpoint，例如：

```text
checkpoints/best.pt
```

Test 评估时必须显式加载 `best.pt`，并在日志中说明 checkpoint 类型。

违规后果：last checkpoint 可能已经过拟合，导致 test AUPRC 虚低或不可解释。

代码级检查：`utils.guardrails.check_eval_procedure(checkpoint_path, checkpoint_type="best")`。

---

## 9. 🔴 约束 #7：AUPRC 和 AUROC 必须同时报告

禁止只看 AUROC：在极度不平衡数据上 AUROC 可能虚高。

禁止只看 AUPRC：AUPRC 必须结合 AUROC、split 方式、positive rate、评估集定义解释。

每次 test / final eval 必须同时报告：

```text
split_mode
AUROC
AUPRC
```

违规后果：性能判断可能被误导，例如 `AUROC=0.994` 但 `AUPRC=0.226`。

代码级检查：`utils.guardrails.report_metrics(auroc, auprc, split_mode)`。

---

## 10. 🔴 约束 #8：CCLMoff CSV 字段和 Tier 元数据不完整

以 `reborn_doc/1. 大纲拟定.md` 的最新审计为准：

| 项 | 当前结论 |
|:---|:---|
| 文件 | `data/cclmoff/09212024_CCLMoff_dataset.csv` |
| 总行数 | `6,393,373` |
| 字段数 | 11 个 |
| 字段 | `sgRNA_seq`, `off_seq`, `read`, `sgRNA_type`, `label`, `chr`, `Location`, `Direction`, `Length`, `Method`, `id` |
| 标签 | `label=1` 是 `observed_positive`；`label=0` 是 `unobserved_candidate` |
| 元数据缺口 | `Method` / `Length` 大量为空；`sgRNA_type` 不是检测方法 |

做 Tier-aware 训练前必须先建立可靠的检测方法映射：

| Tier | 方法 | 权重 |
|:---:|:---|:---:|
| Tier 1 | DISCOVER-seq, DISCOVER-seq+, GUIDE-seq, HTGTS | 1.0 |
| Tier 2 | IDLV, BLESS, BLISS, PEM-seq, VIVO | 0.7 |
| Tier 3 | CIRCLE-seq, CHANGE-seq, Digenome-seq, SITE-seq | 0.4 |

`Method` 为空、`SURRO-seq`、`Extru-seq` 等未确认方法先标为 `unknown`，不得强行映射到 Tier。

如果某个派生 CSV / NPZ 只保留 `sgRNA_seq`, `off_seq`, `sgRNA_type`, `label`, `id` 五个字段，也不能反推完整元数据存在。

---

## 11. 术语规范

1. 不用 `"negative"` 作为核心数据语义，使用 `"unobserved_candidate"`。
2. `label=0` 不代表安全位点，只代表当前实验未检测到切割。
3. `"positive"` 可用，但更推荐在数据表语义中写作 `"observed_positive"`。
4. 命名使用 R9/C9 规范；禁止在新增代码中出现裸 `"9bit"`。文件名、变量名、注释应使用 `R9` / `C9` / `region` / `run` 等明确命名。

---

## 12. Git 管理铁律

### 绝对禁止

1. 禁止复制文件夹做版本管理。所有 BL 差异必须通过 Git 分支和 YAML config 管理。
2. 禁止执行危险 Git 命令，除非用户明确逐字确认：
   - `git push --force` / `git push -f`
   - `git reset --hard`
   - `git rebase`
   - `git clean -f`
3. 禁止在未经用户确认的情况下执行 `git commit` 或 `git push`。

### 强制要求

1. 版本切换通过 Git 分支 + YAML config 文件完成。
2. 每个 BL 版本完成后必须打附注 tag：

   ```bash
   git tag -a BL{n}-{sub}-v{x.y} -m "描述 | AUROC=x.xx | AUPRC=x.xx | GPU=xxGB"
   ```

3. 实验结果必须记录到 `results/experiments.csv`。
4. 删除或移动文件前，先确认该文件是否被 Git 追踪。

---

## 13. 代码和文件操作规范

1. 超参数、模型选型、数据路径必须通过 YAML config 传入，禁止硬编码在 `.py` 文件中。
2. 新增 Baseline 时，优先新增独立模型文件和对应 config，不在旧模型中堆叠大段 `if-else`。
3. 每次训练必须输出可复现实验记录：config 文件、commit hash、随机种子、AUROC、AUPRC、split_mode、best_epoch。
4. 禁止修改 `.env`、环境变量文件或系统级配置文件。
5. 禁止删除 `data/`、`results/`、`reference/` 目录下的用户数据。
6. 删除任何现有代码文件前，必须先询问用户确认。
7. 新建目录用于版本备份是被禁止的。

---

## 14. AI 交接流程

每次新 AI 接入时：

1. 读取 `AGENTS.md`。
2. 读取 `reborn_doc/1. 大纲拟定.md`。
3. 运行 `git status --short`，识别他人未提交工作。
4. 运行 `python scripts/audit_compliance.py`。
5. 确认理解所有 🔴 约束。
6. 在模型代码中调用 `check_model_config(config)`。
7. 在 Run 编码代码中调用或测试 `check_run_encoding_positions(...)`。
8. 在 test 评估代码中调用 `check_eval_procedure(path, "best")`。
9. 在指标输出处使用 `report_metrics(auroc, auprc, split_mode)` 或等价逻辑。

---

## 15. 项目关键文件速查

| 文件 | 用途 |
|:---|:---|
| `reborn_doc/1. 大纲拟定.md` | 项目总纲、执行计划、检查清单 |
| `configs/*.yaml` | 各 BL 版本配置 |
| `results/experiments.csv` | 实验记录，必须随 tag 同步更新 |
| `utils/guardrails.py` | 代码级强制约束检查 |
| `scripts/audit_compliance.py` | 自动合规审计 |
| `scripts/check_before_commit.py` | 提交前检查入口 |
| `scripts/audit_schema.py` | 数据集 schema 审计 |
| `scripts/audit_coordinates.py` | 坐标约定检查 |
| `scripts/verify_rnafm.py` | RNA-FM 权重验证 |

---

## 16. 约束覆盖检查

| 坑编号 | 是什么 | AGENTS.md 约束 | guardrails 检查 |
|:---:|:---|:---|:---|
| 坑1 | CCLMoff ckpt 不匹配 / RNA-FM 绕过 | #1 RNA-FM 必须接入或显式关闭 | `check_model_config` |
| 坑2 | Region + Run 拼接失败 | #3 Run 限制 1-20 | `check_run_encoding_positions` |
| 坑3 | 预编码 CPU 瓶颈 | 流程问题 | 暂无 |
| 坑4 | CSV / Tier 标注缺失 | #8 CCLMoff 元数据不完整 | `check_cclmoff_columns` |
| 坑5 | 坐标方向 | #3 坐标约定 + Run 限制 1-20 | `check_run_encoding_positions` |
| 坑6 | C9 旧代码 Run 跨 23 位 | #3 Run 限制 1-20 | `check_run_encoding_positions` |
| 坑7 | frozen RNA-FM 不够 | #2 `freeze_rnafm` 必须声明 | `check_model_config` |
| 坑8 | 只用单数据 / split 不清 | #4 split 必须声明 | `check_model_config` |
| 坑9 | random split 信息泄漏 | #4 split 必须声明 | `check_model_config` |
| 坑10 | test 用 last checkpoint | #6 test 用 best | `check_eval_procedure` |
| 坑11 | `pos_weight=145` | #5 `pos_weight` 上限 | `check_model_config` |

---

## 17. 变更日志

| 日期 | AI | 修改内容 | 约束影响 |
|:---|:---|:---|:---|
| 2026-05-26 | Kimi 提案 / Codex 落地 | 建立 8 条核心约束 + Guardrails + 自动审计 + 提交前检查 | 新建三层约束体系 |


---

## 18. BL 系列路线图（所有 AI 必须遵守，禁止跳步）

> 本章节定义 Baseline 的精确递进关系、版本归属、以及每个版本回答的科学问题。
> **任何新模型在编码前必须先确认自己属于哪个 BL，不得跳过步骤、不得归属混乱。**

### 18.1 当前状态（截至 2026-05-26）

| 版本 | 实际做了什么 | Test AUPRC | 状态 |
|:---|:---|---:|:---:|
| P0 | R9(0.16) + C9(0.84) 加权平均 | 0.851 (GUIDE-seq) | ✅ |
| BL0a | frozen RNA-FM + MLP | 0.073 (CCLMoff) | ✅ |
| BL0b | fine-tune RNA-FM + MLP | 0.522 (CCLMoff) | ✅ |
| BL3-hard-A | Hard seed (1×/2×) + Region + Run + CNN | 0.555 (GUIDE-seq) | ✅ |
| BL3-hard-B | Soft seed (exp) + Region + Run + CNN | 0.555 (GUIDE-seq) | ✅ |
| BL3-hard-C | Learnable seed + Region + Run + CNN | 0.566 (GUIDE-seq) | ✅ |
| BL3-Run-only | Run-only (无 Region, 无 RNA-FM) | 0.609 (GUIDE-seq) | ✅ |
| **BL4-Run-only** | RNA-FM frozen + Run CNN | **0.206 (CCLMoff, group-safe)** | ✅ |
| BL3-gradient (BL3b) | Run-only + position embedding / seed gate | 待跑 | 🔄 |
| BL4-full | RNA-FM frozen + Region + Run (三者拼接) | 未实现 | ❌ |
| BL5-0~3 | Three-view late gated fusion | 未实现 | ❌ |
| BL6 | Cross-view attention + gated fusion | 未实现 | ❌ |

**⚠️ 归属混乱已修正：**
- 原命名 `"BL3-RNA-FM-Fusion"` → 修正为 **`BL4-Run-only`**（包含 RNA-FM，不属于 BL3）
- BL3 系列**不包含 RNA-FM**，只包含 hand-crafted prior（Region + Run）
- 任何包含 RNA-FM 的模型必须从 **BL4** 开始编号

### 18.2 版本精确定义（禁止修改核心问题）

**P0 — 旧模型晚期融合**
- 是什么：R9/DeepFocus(0.16) + C9/ConMismatch9(0.84) 概率加权
- 回答的问题：两个已有模型的互补性？
- 结论：Late Fusion 太浅，提升微小(+0.0024)

**BL0 — RNA-FM 基线**
- BL0a：frozen RNA-FM + MLP → AUPRC=0.073
- BL0b：fine-tuned RNA-FM + MLP → AUPRC=0.522
- 回答的问题：RNA-FM 单独有多强？fine-tune 能提升多少？
- 结论：fine-tune 是必须的（+615%）

**BL3-hard — 先验特征验证（无 RNA-FM）**
- 共同定义：canonical prior encoder（Region + Run）+ CNN
- BL3-hard-A：Hard seed（pos 1-15 weight=1, pos 16-20 weight=2）
- BL3-hard-B：Soft seed（exp(-d/τ), τ=4）
- BL3-hard-C：Learnable seed（20个可学习位置权重）
- 回答的问题：区域先验 + 连续错配先验是否有效？soft/hard/learnable 哪个更好？
- 关键发现：**Run-only > Region-only >> Combined**（0.609 > 0.583 > 0.555）

**BL3-gradient（BL3b）— Seed 区回归（无 RNA-FM）**
- 定义：在 Run-only 基础上加 position embedding 或 seed gate
- 回答的问题：seed 渐变是否优于硬 seed？能否超过纯 Run-only？
- 候选：BL3b-pos / BL3b-gate / BL3b-ab（两者叠加）

**BL4 — Frozen RNA-FM + Prior Concat**
- 定义：`z = concat(Pool(H_prior), Pool(H_rnafm))`
- BL4-Run-only：RNA-FM + Run（Region 缺失）→ **已跑**
- BL4-full：RNA-FM + Region + Run（三者拼接）→ **待补**
- 回答的问题：显式生物先验能否增强 frozen RNA-FM？

**BL5 — Three-View Late Gated Fusion**
- BL5-0：Concat + MLP（三个视角简单拼接）
- BL5-1：Cross-Attn（prior ↔ RNA-FM）
- BL5-2：Gated Fusion（动态权重，无 Cross-Attn）
- BL5-3：Cross-Attn + Gated Fusion（完整版 ⭐）
- 回答的问题：动态融合是否优于简单拼接？

**BL6 — Cross-View Attention + Gated Fusion**
- 定义：在 BL5-3 基础上加双向 Cross-Attn、多层 Cross-Attn、不确定性感知加权
- 回答的问题：token-level 视角交互能否进一步提升？

### 18.3 递进关系（禁止跳步）

```
P0 ──→ BL0a ──→ BL3-hard ──→ BL3-gradient ──→ BL4 ──→ BL5 ──→ BL6
  │       │           │              │            │        │        │
  ▼       ▼           ▼              ▼            ▼        ▼        ▼
加权融合  frozen     先验验证       seed回归     先验+FM   动态融合  token交互
```

**禁止行为：**
- 跳过 BL4 直接做 BL5（必须先验证拼接 baseline）
- 跳过 BL3-gradient 直接做 BL4-full（必须先确认 seed 回归是否有效）
- 把含 RNA-FM 的模型命名为 BL3（BL3 无 RNA-FM）

### 18.4 版本归属检查表（新增模型时必须核对）

| 检查项 | 如果为是 → 归属 | 如果为否 → 归属 |
|:---|:---|:---|
| 包含 RNA-FM？ | 必须从 BL4 起编号 | 可能是 BL3 系列 |
| 包含 Region + Run 先验？ | 可能是 BL3/BL4/BL5/BL6 | 可能是纯 BL0 |
| 只做 seed 梯度/回归？ | BL3-gradient | — |
| 三者拼接（Region+Run+RNA-FM）？ | BL4-full | — |
| 动态权重/Cross-Attn？ | BL5 系列 | — |
| 多层/token-level 交互？ | BL6 | — |

---

## 19. 变更日志

| 日期 | AI | 修改内容 | 约束影响 |
|:---|:---|:---|:---|
| 2026-05-26 | Kimi 提案 / Codex 落地 | 建立 8 条核心约束 + Guardrails + 自动审计 + 提交前检查 | 新建三层约束体系 |
| 2026-05-26 | Kimi 提案 | 追加 BL 系列路线图 v2，修正版本归属混乱，禁止跳步 | 新增第 18 章 |
