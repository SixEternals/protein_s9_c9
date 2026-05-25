# AGENTS.md — CRISPR/Cas9 Off-Target Fusion 项目

> 本文件约束所有在此项目上工作的 AI 助手（Kimi / Codex / Claude）。
> 开始任何编码、重构或实验任务前，必须先阅读 `reborn_doc/1. 大纲拟定.md` 了解项目总纲。

---

## 一、Git 管理铁律（最高优先级）

### 绝对禁止
1. **禁止复制文件夹做版本管理**。不得创建 `baseline_v2/`、`model_new/`、`experiments_backup/` 等子文件夹来保存不同版本的代码或实验结果。所有版本必须通过 Git 分支和 config 文件管理。
2. **禁止执行以下危险 Git 命令**（除非用户明确要求并逐字确认）：
   - `git push --force` / `git push -f`
   - `git reset --hard`
   - `git rebase`
   - `git clean -f`
3. **禁止在未经用户确认的情况下执行 `git commit` 或 `git push`**。

### 强制要求
1. **版本切换通过 Git 分支 + YAML config 文件完成**。不同 Baseline（BL0/BL3/BL4/BL5/BL6）共享同一套代码库，差异通过 `configs/*.yaml` 和模型入口文件区分。
2. **每个 BL 版本完成后必须打附注 tag**：
   ```bash
   git tag -a BL{n}-{sub}-v{x.y} -m "描述 | AUROC=x.xx | AUPRC=x.xx | GPU=xxGB"
   ```
3. **实验结果必须记录到 `results/experiments.csv`**，不可只在 commit message 里记录。
4. **删除或移动文件前，先确认该文件是否被 Git 追踪**，避免误删历史版本的关键代码。

### Git 工作流规范
```bash
# 新功能开发
git checkout -b feat/bl{x}-{description}

# 开发完成，合并到 main 并打 tag
git checkout main
git merge feat/bl{x} --no-ff -m "feat: 描述"
git tag -a BL{x}-v1.0 -m "..."

# 查看历史
git log --oneline --decorate --tags --graph
```

---

## 二、坐标与术语规范（硬性约束）

### 坐标约定
- position 1–20: protospacer / guide-target aligned region
- position 1: PAM-distal end（最远端）
- position 20: PAM-proximal end（紧挨 PAM）
- positions 21–23: PAM region（NGG）
- hard seed: positions 16–20

> ⚠️ **任何涉及位置索引的代码修改前，必须对照此约定验证**。如果坐标反了，Region 编码的 seed/ordinary 标记全错。

### 术语规范
- 不用 `"negative"`，用 `"unobserved_candidate"`
- `label=0` ≠ 安全位点，只是没被实验检测到
- `"positive"` 可用（表示实验检测到切割）

### 检测方法 Tier 分层
| Tier | 方法 | 权重 |
|:---:|:---|:---:|
| Tier 1 | DISCOVER-seq, DISCOVER-seq+, GUIDE-seq, HTGTS | 1.0 |
| Tier 2 | IDLV, BLESS, BLISS, PEM-seq, VIVO | 0.7 |
| Tier 3 | CIRCLE-seq, CHANGE-seq, Digenome-seq, SITE-seq | 0.4 |

---

## 三、代码风格约束

1. **命名规范**：使用 R9/C9 命名规范，禁止在任何代码中出现裸 `"9bit"`。
2. **配置驱动**：超参数、模型选型、数据路径必须通过 YAML config 文件传入，禁止硬编码在 `.py` 文件中。
3. **模块化**：新增 Baseline 时，优先新增独立模型文件（如 `models/bl5_series.py`）和对应 config，而非在旧模型中堆叠 `if-else` 分支。
4. **实验记录**：每次运行必须输出可复现的实验记录（config 文件、commit hash、随机种子、指标）。

---

## 四、文件操作规范

1. **禁止修改 `.env`、环境变量文件或系统级配置文件**。
2. **禁止删除 `data/`、`results/`、`reference/` 目录下的用户数据**，即使它们在 `.gitignore` 中。
3. **删除任何现有代码文件前，必须先询问用户确认**。
4. **新建目录用于版本备份是被禁止的**（参见 Git 管理铁律）。

---

## 五、项目关键文件速查

| 文件 | 用途 |
|:---|:---|
| `reborn_doc/1. 大纲拟定.md` | 项目总纲、执行计划、检查清单 |
| `configs/*.yaml` | 各 BL 版本的配置文件 |
| `results/experiments.csv` | 实验记录（必须随 tag 同步更新） |
| `scripts/audit_schema.py` | 数据集 schema 审计 |
| `scripts/audit_coordinates.py` | 坐标约定检查 |
| `scripts/verify_rnafm.py` | RNA-FM 权重验证 |
