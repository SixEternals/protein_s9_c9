# P1 RNA-FM 接入执行汇报

> 时间：2026-05-25  
> 范围：P1 修正版的 Phase 1 smoke gate，以及 Phase 2 的最小训练链路验证。  
> 结论：RNA-FM 已接入当前代码，BL0a 的真实前向、反向、CCLMoff CSV 单样本推理、tiny smoke training、非沙盒 GPU smoke test、1000 行 quick train 均已通过；全量 BL0a 正式训练尚未启动。

---

## 1. 当前状态

已完成：

- RNA-FM 权重加载：`data/rnafm/checkpoints/RNA-FM_pretrained.pth`
- RNA-FM 规格确认：12 层、640 维、99,521,546 参数
- BL0a 模型路径接入：RNA-FM + CCLMoff-style MLP head
- CCLMoff CSV 单样本前向验证：已通过
- MLP head 反向传播验证：已通过
- tiny smoke training：已通过，并写入 `results/experiments.csv`
- 非沙盒 GPU 环境确认：已通过，2 张 `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- 非沙盒 GPU smoke test：已通过
- 1000 行 CCLMoff CSV quick train：已通过，并写入 `results/experiments.csv`

未完成：

- 全量 BL0a 训练未启动
- DDP 训练入口未实现
- BL0a 正式 benchmark 指标未产生

原因：

- 沙盒环境中 `torch.cuda.is_available() == False`，只能做 CPU smoke；非沙盒环境中 CUDA 可用
- 目前 `scripts/train_bl0a.py` 是 quick training/smoke training 入口，不是 DDP 正式训练器
- 正式训练还需要明确 split 策略，避免把 quick train 的同集评估误当成泛化指标

---

## 2. 关键修正

### 2.1 不再使用过期 checkpoint 路径

旧配置里曾指向：

```text
/data/zwf/cache/torch/hub/checkpoints/RNA-FM_pretrained.pth
```

该路径当前不存在。

现在统一使用：

```text
data/rnafm/checkpoints/RNA-FM_pretrained.pth
```

相关文件：

- `configs/bl0.yaml`
- `configs/bl0a_smoke.yaml`
- `utils/rnafm.py`

### 2.2 PyTorch 2.6+ checkpoint 安全加载问题已处理

`rna-fm==0.2.2` 加载官方 RNA-FM checkpoint 时需要反序列化 `argparse.Namespace`。当前项目通过 `utils/rnafm.py` 中的窄 allow-list 处理：

```python
torch.serialization.safe_globals([argparse.Namespace])
```

这个处理只用于已经审计过的本地 RNA-FM checkpoint。

### 2.3 `<sep>` 不是当前 RNA-FM alphabet 的原生 token

这是本次最重要的实现发现。

CCLMoff 公开代码中写法是：

```python
seq = sgRNA_seq + "<sep>" + off_seq
```

但当前 `rna-fm==0.2.2` + `RNA-FM_pretrained.pth` 加载出来的 alphabet 是 `roberta_large` 风格，实际 token 列表中没有 `<sep>`：

```text
<cls>, <pad>, <eos>, <unk>, A, C, G, U, R, Y, K, M, S, W, B, D, H, V, N, -, <null_1>, <null_2>, <null_3>, <null_4>, <mask>
```

因此：

- `alphabet.get_idx("<sep>") == alphabet.unk_idx`
- 当前策略不是“真正的 `<sep>` special embedding”
- 当前策略是把 `<sep>` 作为单个 `<unk>` delimiter

这比 RNA-FM 默认 batch converter 更安全，因为默认 converter 会把字符串 `"<sep>"` 拆成多个字符，从而产生多个未知 token。我们自己的 tokenizer 会把它压成一个 delimiter token，并审计 unknown token 数量。

现阶段策略：

```text
sgRNA + single <unk> delimiter + off_seq
```

后续如果找到 CCLMoff 作者实际使用的带 `<sep>` alphabet 或匹配 checkpoint，需要重新审计这一点。

---

## 3. 新增与修改文件

新增：

- `utils/rnafm.py`
  - RNA-FM 安全加载
  - 参数统计
  - pair sequence 规范化
  - CCLMoff-style tokenizer
- `scripts/check_rnafm_import.py`
  - 验证 `fm` 导入、RNA-FM 权重加载、alphabet、pair tokenizer
- `scripts/smoke_test_bl0a.py`
  - 验证 BL0a 真实前向、反向、CCLMoff CSV 单样本前向
- `datasets/__init__.py`
- `datasets/cclmoff_dataset.py`
  - CCLMoff CSV 数据集加载器
  - 明确 `label=0` 为 `unobserved_candidate`
- `configs/bl0a_smoke.yaml`
  - tiny smoke training 配置
- `configs/bl0a_quick_1000.yaml`
  - 1000 行 quick train 配置
- `scripts/train_bl0a.py`
  - BL0a tiny smoke training 入口

修改：

- `models/bl0_cclmoff.py`
  - 使用统一 RNA-FM loader
  - `freeze_rnafm=True` 时保持 RNA-FM 子模型为 `eval()`
  - 只训练 MLP head
- `scripts/train_bl0.py`
  - 训练 collate 改用项目 tokenizer，避免默认 converter 错误处理 `"<sep>"`
- `configs/bl0.yaml`
  - 修正 RNA-FM checkpoint 路径
  - 更新过期的 data source 状态

---

## 4. 已产出结果

### 4.1 RNA-FM import check

命令：

```bash
/data/zwf/conda/envs/reborn_seed/bin/python scripts/check_rnafm_import.py
```

产物：

- `results/audits/rnafm_import_check.txt`
- `results/audits/rnafm_import_check.json`

结果：

```text
ALL CHECKS PASSED
```

关键记录：

- `rna-fm` 版本：0.2.2
- RNA-FM：12L / 640d / 99.5M
- native `<sep>`：不存在
- separator policy：`single_<unk>_delimiter`
- canonical pair unknown token count：1，符合预期

### 4.2 BL0a smoke test

命令：

```bash
/data/zwf/conda/envs/reborn_seed/bin/python scripts/smoke_test_bl0a.py
```

产物：

- `results/smoke_tests/bl0a_smoke_test.md`
- `results/smoke_tests/bl0a_smoke_test.json`

结果：

```text
ALL SMOKE TESTS PASSED
```

关键记录：

- 沙盒 CPU smoke：通过
- 非沙盒 GPU smoke：通过
- GPU device：`cuda`
- GPU 显存峰值：约 0.392 GB
- Total params：99,562,635
- Trainable params：41,089
- Forward logits shape：`[2]`
- MLP head backward：通过
- CCLMoff CSV 第 1 条样本前向：通过
- 单真实样本 probability：CPU 0.634661；GPU 0.627839

### 4.3 BL0a tiny smoke training

命令：

```bash
/data/zwf/conda/envs/reborn_seed/bin/python scripts/train_bl0a.py --config configs/bl0a_smoke.yaml
```

产物：

- `results/bl0a_smoke_train/report.md`
- `results/bl0a_smoke_train/summary.json`
- `results/bl0a_smoke_train/bl0a_head_smoke.pt`

结果：

- Status：`smoke_train_passed`
- Dataset rows used：16
- observed_positive：8
- unobserved_candidate：8
- Epochs：1
- Batch size：2
- Final loss：0.690288
- AUROC on same tiny smoke set：0.578125
- AUPRC on same tiny smoke set：0.667361

注意：

- 这是训练链路 smoke test，不是正式 BL0a benchmark
- 指标不能写作论文结果
- checkpoint 只保存 MLP head，不重复保存 RNA-FM 本体

### 4.4 BL0a 1000 行 quick train

命令：

```bash
/data/zwf/conda/envs/reborn_seed/bin/python scripts/train_bl0a.py --config configs/bl0a_quick_1000.yaml
```

执行环境：

- 非沙盒 GPU 环境
- `torch.cuda.is_available() == True`
- GPU 数量：2
- GPU 型号：`NVIDIA RTX PRO 6000 Blackwell Server Edition`

产物：

- `results/bl0a_quick_1000/report.md`
- `results/bl0a_quick_1000/summary.json`
- `results/bl0a_quick_1000/bl0a_head_smoke.pt`

结果：

- Status：`train_chain_passed`
- Dataset rows used：1000
- observed_positive：407
- unobserved_candidate：593
- Epochs：2
- Batch size：32
- Device：CUDA
- Final loss：0.575725
- AUROC on same quick set：0.928581
- AUPRC on same quick set：0.913107
- Last train batch loss：0.527771

注意：

- 这仍然是 quick training-chain check，不是正式 BL0a benchmark。
- 当前评估是在同一个 1000 行 quick subset 上做的，只能说明训练链路和 GPU 路径可用，不能说明泛化能力。

---

## 5. 对原 P1 prompt 的修正说明

原 prompt 中提到 CCLMoff CSV 是 950 万样本。根据当前已下载并审计的 Figshare v2 文件，真实行数是：

```text
6,393,373
```

因此后续文档和训练计划应以 6,393,373 行为准。

原 prompt 示例代码里假设 `alphabet.encode(test_seq)` 可直接处理 `"<sep>"`。当前环境中 alphabet 没有 `encode` 方法，也没有原生 `<sep>` token。因此本项目不能照抄该写法，必须使用 `utils/rnafm.py` 中的项目 tokenizer。

---

## 6. 下一步建议

### 6.1 已确认 GPU 环境

已在非沙盒环境确认：

```text
cuda_available=True
device_count=2
0 NVIDIA RTX PRO 6000 Blackwell Server Edition
1 NVIDIA RTX PRO 6000 Blackwell Server Edition
```

GPU smoke test 和 1000 行 quick train 均已通过。

### 6.2 BL0a 正式训练前还缺三件事

1. DDP 或单 GPU 正式训练脚本  
   当前 `scripts/train_bl0a.py` 已能做 tiny/quick training-chain check，但不是 DDP 正式训练器。

2. 明确 split 策略  
   不能随机打散后直接报正式泛化指标。至少需要：
   - guide/sgRNA-safe split
   - leave-one-sgRNA-out 或按 `sgRNA_type` 留出
   - 后续再做 leave-one-dataset/method split

3. 重新确认 `<sep>` 兼容策略是否接受  
   当前策略可跑，但它不是作者可能宣称的“真实 `<sep>` special token”。如果后续找到 Docker 镜像里的实际 CCLMoff inference 代码或 alphabet，应优先复核。

### 6.3 不建议马上做全量训练

理由：

- GPU 已确认可用，但当前只完成 GPU smoke 和 1000 行 quick train
- 正式 split、DDP、日志、checkpoint、评估表还未完善
- CCLMoff 官方 checkpoint 仍与公开 MLP head 不兼容，不能作为官方推理权重使用

推荐顺序：

```text
1. 写正式训练 config 和 split
2. 实现单 GPU/双 GPU DDP 正式训练入口
3. 先跑 1 个 sgRNA-safe 小规模验证
4. 再跑 BL0a retrained benchmark
5. 再进入 BL3-hard 或 BL4
```
