# BL5-v4 PAM Strict Holdout P3 — AGG/TGG/GAG/CGG 横向总结

> 执行人：Kimi  
> 日期：2026-06-10  
> 对应任务：P3 四 motif 横向对比与泛化总结  
> 前置依赖：AGG/TGG/GAG/CGG strict holdout training 与 paired bootstrap 均已可用；其中 CGG paired bootstrap 由 P2 新增完成。
>
> 历史命名差异（bootstrap JSON 文件名不一致，以下为实际路径，后续脚本请勿硬编码单一命名）：
> - AGG: `results/bl5_generalization/pam_strict_holdout/AGG/paired_bootstrap_results.json`
> - TGG: `results/bl5_generalization/pam_strict_holdout/TGG/paired_bootstrap.json`
> - GAG: `results/bl5_generalization/pam_strict_holdout/GAG/paired_bootstrap.json`
> - CGG: `results/bl5_generalization/pam_strict_holdout/CGG/paired_bootstrap.json`

---

## 1. 执行摘要

本任务为纯汇总分析，**未训练任何模型**，仅读取已有产物并生成统一证据链。

- **输入**：AGG/TGG/GAG/CGG 的 summary.json、paired_bootstrap.json、seenPAM sanity 数据
- **输出**：统一 CSV 总表 + 技术报告 + 总监简报 + 执行报告
- **CGG seenPAM**：此前缺失，本轮补做 eval-only export（非训练）

---

## 2. 合规声明

| 项目 | 状态 |
|:---|:---:|
| 未训练 | ✅ |
| 未 commit / push | ✅ |
| 未修改 data/ / reference/ | ✅ |
| 未修改 BL6 文件 | ✅ |
| 未新增 experiments.csv 行 | ✅ |
| Δ = NoPAM − PAM | ✅ |

---

## 3. CGG seenPAM Sanity（本轮补做）

由于 CGG 此前缺少 seenPAM sanity，本轮执行 eval-only export：

```bash
# PAM
python scripts/train_bl5.py \
  --config /tmp/bl5_v4_pam_holdout_cgg_seenpam_eval.yaml \
  --checkpoint results/bl5_v4_pam_holdout_cgg/checkpoints/best.pt \
  --eval-only --eval-split-key test_seenPAM --no-experiment-log

# NoPAM
python scripts/train_bl5.py \
  --config /tmp/bl5_v4_nopam_holdout_cgg_seenpam_eval.yaml \
  --checkpoint results/bl5_v4_nopam_holdout_cgg/checkpoints/best.pt \
  --eval-only --eval-split-key test_seenPAM --no-experiment-log
```

**验证结果**：

| 检查项 | 结果 |
|:---|:---|
| PAM seenPAM rows | 907,734 ✅ |
| NoPAM seenPAM rows | 907,734 ✅ |
| Row alignment (6 key cols) | ✅ Passed |
| PAM_original == off_seq[20:23] | ✅ Passed |
| CGG count in test_seenPAM | 0 ✅ |
| Probability ∈ [0, 1] | ✅ Passed |
| observed_positive | 2,871 |
| unobserved_candidate | 904,863 |

**Point estimates**：

| Model | AUROC | AUPRC |
|:---|---:|---:|
| PAM | 0.989886 | 0.598008 |
| NoPAM | 0.988353 | 0.526608 |
| Δ = NoPAM − PAM | −0.001534 | −0.071400 |

**ATT caveat**：CGG split_manifest 标注 `test_seenPAM_unseen_in_train=["ATT"]`，即存在 1 个 rare ATT 样本在 train_H 中未出现，seenPAM sanity 解释需带此 caveat。

**主 summary.json 未受破坏**：PAM status=completed, epochs=10; NoPAM status=completed, epochs=10, best_epoch=9。

---

## 4. 四 Motif Test_H 总表

| Holdout PAM | family | test_H n | observed_positive | PAM AUPRC | NoPAM AUPRC | ΔAUPRC = NoPAM−PAM | 95% CI | CI crosses 0? |
|:---|:---|---:|---:|---:|---:|---:|:---|:---:|
| AGG | NGG | 277,247 | 744 | 0.0276 | 0.2038 | **+0.1762** | [0.151, 0.204] | ❌ No |
| TGG | NGG | 292,861 | 703 | 0.1040 | 0.0510 | **−0.0529** | [−0.069, −0.039] | ❌ No |
| GAG | non-NGG | 9,061 | 111 | 0.9907 | 0.9897 | **−0.0010** | [−0.006, 0.004] | ✅ Yes |
| CGG | NGG | 46,592 | 186 | 0.4329 | 0.2705 | **−0.1624** | [−0.203, −0.123] | ❌ No |

### AUROC 辅表

| Holdout PAM | PAM AUROC | NoPAM AUROC | ΔAUROC = NoPAM−PAM | 95% CI | CI crosses 0? |
|:---|---:|---:|---:|:---|:---:|
| AGG | 0.4797 | 0.9456 | **+0.4659** | [0.451, 0.479] | ❌ No |
| TGG | 0.9285 | 0.7194 | **−0.2090** | [−0.223, −0.194] | ❌ No |
| GAG | 0.9998 | 0.9998 | **−0.0000** | [−0.0002, 0.0000] | ✅ Yes |
| CGG | 0.9774 | 0.9612 | **−0.0162** | [−0.023, −0.011] | ❌ No |

---

## 5. seenPAM Sanity 总表

| Holdout | PAM seenPAM AUPRC | NoPAM seenPAM AUPRC | Δ = NoPAM−PAM | Interpretation |
|:---|---:|---:|---:|:---|
| AGG | 0.5900 | 0.5700 | −0.0201 | No catastrophic collapse; AGG failure is clean unseen-PAM failure |
| TGG | 0.5028 | 0.5744 | +0.0716 | seenPAM shows NoPAM > PAM; TGG test_H PAM advantage needs cautious interpretation |
| GAG | 0.4665 | 0.4498 | −0.0167 | No catastrophic collapse; GAG remains exploratory |
| CGG | 0.5980 | 0.5266 | −0.0714 | No catastrophic collapse; PAM > NoPAM on seen-PAM AUPRC; ATT caveat applies |

---

## 6. Evidence Strength 分级

| Holdout | Strength | Direction | Key Caveat |
|:---|:---:|:---|:---|
| **AGG** | Strong | NoPAM > PAM | Clean unseen-PAM failure; seenPAM sanity confirms no collapse |
| **TGG** | Strong test_H | PAM > NoPAM | seenPAM shows opposite direction (NoPAM > PAM); interpretation cautious |
| **CGG** | Strong | PAM > NoPAM | Third NGG evidence point; conclusion limited to CGG |
| **GAG** | Exploratory | No stable difference | non-NGG; CI crosses 0; composition confounding (21 observed_positive_only sgRNA_type) |

### GAG Composition Confounding

GAG test_H 的 per-sgRNA composition 严重偏斜：

- 21 个 sgRNA_type 为 **observed_positive_only**（105 observed_positive，0 unobserved_candidate）
- 2 个 sgRNA_type 为 **mixed**（6 observed_positive，8,950 unobserved_candidate）
- 仅看 mixed sgRNA_type 时，AUPRC 跌至约 **PAM 0.494 / NoPAM 0.439**
- 因此 full-test AUPRC ~0.99 被 observed_positive_only sgRNA 严重 inflate
- paired bootstrap CI crosses 0

**GAG 不允许作为"第三种稳定模式"使用。**

---

## 7. 科学结论（谨慎措辞）

AGG/TGG/CGG 三个 NGG strict holdout 给出 **motif-dependent behavior**：

- **AGG**: NoPAM outperforms PAM（ΔAUPRC = +0.176，bootstrap CI 不跨 0，paired bootstrap supports the direction）
- **TGG**: PAM outperforms NoPAM（ΔAUPRC = −0.053，bootstrap CI 不跨 0，paired bootstrap supports the direction）
- **CGG**: PAM outperforms NoPAM（ΔAUPRC = −0.162，bootstrap CI 不跨 0，paired bootstrap supports the direction）

**GAG** non-NGG subset does not show stable PAM/NoPAM difference（CI crosses 0），and is composition-confounded；it should remain exploratory.

### 总口径

当前证据**不支持**一刀切删除 PAM encoder，也**不支持** PAM encoder 在所有 unseen PAM 上稳定有益。更稳妥的结论是：

> BL5-v4 的 explicit PAM encoder 在 strict cross-PAM generalization 中呈现 **motif-specific / subset-dependent behavior**，而不是单调稳定收益。

### 禁止口径

以下表述在本报告中未出现、也不应被引用：

- ❌ "PAM encoder 总是有益"
- ❌ "PAM encoder 总是有害"
- ❌ "AGG 被 CGG/TGG 推翻"
- ❌ "GAG 证明 non-NGG 没问题"
- ❌ "四种模式已确认"
- ❌ "最终规律已证明"

---

## 8. 产物清单

| 文件 | 路径 |
|:---|:---|
| Holdout summary CSV | `results/bl5_generalization/pam_strict_holdout/four_motif_holdout_summary.csv` |
| seenPAM sanity CSV | `results/bl5_generalization/pam_strict_holdout/four_motif_seenpam_sanity_summary.csv` |
| 技术报告 | `results/bl5_generalization/pam_strict_holdout/four_motif_generalization_report.md` |
| 总监简报 | `results/bl5_generalization/pam_strict_holdout/four_motif_executive_report.md` |
| 执行报告（本文档） | `reborn_doc/bl5_addOn/24_BL5_v4_PAM_Strict_Holdout_AGG_TGG_GAG_CGG_横向总结.md` |

---

## 9. Audit & Git 状态

- `audit_compliance.py`: ERROR=0 / WARNING=95（WARNING=95 来自当前工作区扫描到的既有 Python 文件及未跟踪脚本，非 P3 文档返修新增实验风险）
- `git status`: 无 data/reference 改动；本任务未新增/修改 BL6 文件；当前工作区存在此前遗留的未跟踪 BL6 文件（`configs/bl6_1_*.yaml`、`run/run_bl6_1_*.sh`、`reborn_doc/26_BL6_1_*.md`、`reborn_doc/40_BL6_1_*.md`），BL5 P3 提交时必须排除。禁止 `git add .`；提交时必须精确列出 BL5 P3 文件。未 commit / push。
- `experiments.csv`: 无新增 eval-only 行

---

## 10. 下一步建议

P3 横向总结已完成。建议：

1. **归档 P1–P3 结果**并打 tag
2. **探索 motif-aware PAM encoder 变体**（如 learned PAM embedding with motif clustering）
3. **GGG strict holdout 可作为 optional confirmatory appendix**；当前不作为 BL5 封口前必做项。更高优先级是 P1–P3 归档/提交范围清理，以及 BL6-1 gate audit + report correction
