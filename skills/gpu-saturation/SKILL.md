# GPU 训练饱和度优化

> 本 skill 用于诊断和解决 PyTorch DDP 训练在高端 GPU 上利用率不足的问题。
> 适用场景：模型较小（<10M 参数）、序列较短（<100 tokens）、batch 不够大，导致 GPU SM/显存带宽利用率低。

---

## 1. 诊断 checklist

训练启动后，运行以下命令收集数据：

```bash
# 1. 查看 GPU 实时利用率（每秒采样）
nvidia-smi dmon -s u -c 10

# 2. 查看显存占用详情
nvidia-smi -q -d MEMORY | grep -E "Used|Total|Free"

# 3. 计算单 epoch 耗时（从日志中提取 epoch 输出时间戳）
# 若 6.4M 数据 / batch=2048 ≈ 3125 steps/epoch，epoch 时间 > 30 秒即存在优化空间
```

### 判断标准

| 指标 | 健康 | 需优化 |
|:---|:---|:---|
| SM 利用率 | > 80% | < 70% |
| 显存带宽 | > 30% | < 15% |
| 显存占用 | 50–80% | < 10% |
| epoch 时间 | 与计算量匹配 | 明显慢于预期 |

---

## 2. 常见根因与解决方法

### 根因 A：模型太小，计算密度不足

**现象**：SM 40–60%，显存 < 5%，模型参数 < 1M。

**解决**：
- **增大 batch_size**：这是最有效的手段。小模型的显存占用主要来自激活（activation），不是参数。以 BL3.5-Full 为例（325K 参数）：
  - batch=1024 → 显存 ~500MB
  - batch=8192 → 显存 ~2GB（仍远 < 96GB 上限）
  - 目标：把显存拉到 **50–80%**（如 48GB / 96GB）
- **增大模型 hidden_dim / 层数**：如果实验设计允许，直接增加模型容量。

### 根因 B：DDP `find_unused_parameters=True`

**现象**：日志出现警告：
```
Warning: find_unused_parameters=True was specified in DDP constructor,
but did not find any unused parameters in the forward pass.
This flag results in an extra traversal of the autograd graph every iteration...
```

**解决**：
```python
# 如果模型没有 flow control（如 if/else 导致某些参数某些 step 不参与计算）
model = DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False,  # 改为 False
)
```

**注意**：只有模型存在动态结构（如某些 layer 条件性跳过）时才需要 `True`。纯 CNN/Transformer 固定结构一律用 `False`。

### 根因 C：DataLoader 每 epoch 重新 fork worker

**现象**：epoch 之间有明显停顿（> 2 秒），CPU 利用率在 epoch 边界出现尖峰。

**解决**：
```python
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,       # 保持 4–8
    persistent_workers=True,       # 关键：不杀 worker 进程
    prefetch_factor=4,             # 每个 worker 预取 4 个 batch
    pin_memory=True,               # 已做则保持
)
```

### 根因 D：CPU→GPU 数据传输阻塞

**现象**：GPU 利用率周期性波动（高→低→高），nvidia-smi 中 `GPU-Util` 不稳定。

**解决**：
```python
# 1. DataLoader 中 pin_memory=True（已做则跳过）

# 2. training loop 中使用 non_blocking
for batch in loader:
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    # 必须配合 pin_memory=True 才有效
```

### 根因 E：数据预处理在 CPU 上成为瓶颈

**现象**：nvidia-smi 中 GPU 利用率低，但 `top`/`htop` 中 Python 进程 CPU 利用率 100%。

**解决**：
- 预编码所有特征并存入 NPZ / LMDB / Tensor，训练时只做索引和 `to(device)`
- 避免在 `__getitem__` 中做字符串处理、numpy 动态计算
- 如果必须用 on-the-fly 编码，把编码逻辑用 `torch` 实现并在 GPU 上做

---

## 3. 快速优化模板（按优先级排序）

对正在运行但利用率低的训练，按以下顺序修改：

```
Step 1: find_unused_parameters=False（如果模型无动态结构）
Step 2: batch_size × 2 → 观察显存 → 重复直到显存 50–80%
Step 3: persistent_workers=True
Step 4: prefetch_factor=4（或更高）
Step 5: non_blocking=True（配合 pin_memory）
Step 6: 若仍不够，考虑 torch.cuda.amp 混合精度（进一步降低显存占用，允许更大 batch）
```

---

## 4. 本项目案例

### 案例：BL3.5-Full 在 CCLMoff 上 GPU 利用率低

**硬件**：2× RTX PRO 6000 (96GB)
**模型**：BL3.5-Full，325K 参数，seq_len=20
**初始配置**：batch=2048，num_workers=8，find_unused_parameters=True

**诊断结果**：
- SM 40–62%，显存带宽 8–9%，显存 2GB/96GB
- epoch 时间 ~35 秒（6.4M / 2048 = 3125 steps）

**根因**：
1. 模型太小（325K），计算密度不足
2. find_unused_parameters=True 浪费 DDP 开销
3. batch=1024/卡 喂不饱 GPU

**优化后（下次实验应用）**：
```yaml
# config 修改
training:
  batch_size: 8192        # 从 2048 提升 4 倍
  num_workers: 8
  persistent_workers: true

# train script 修改
model = DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False,  # 模型无动态结构
)

# dataloader 修改
train_loader = DataLoader(
    ..., persistent_workers=True, prefetch_factor=4, pin_memory=True
)

# training loop
tokens = tokens.to(device, non_blocking=True)
```

**预期效果**：SM 80–95%，epoch 时间从 35 秒降至 15–20 秒。

---

## 5. 验证优化效果

修改后重跑，对比以下指标：

```bash
# 1. GPU 利用率
nvidia-smi dmon -s u -c 10

# 2. 训练速度（epoch 时间）
tail -f output.log | grep "epoch="

# 3. 显存占用
nvidia-smi -l 1
```

**合格标准**：
- SM > 80% 持续稳定
- 显存占用 50–80%
- 相同收敛质量下 epoch 时间减少 ≥ 30%

---

## 6. 注意事项

- **增大 batch_size 后需要重新调学习率**：通常线性缩放，如 batch × 4 → lr × 4。但使用 AdamW + weight decay 时需谨慎验证。
- **混合精度（amp）可能引入数值问题**：小模型或极小梯度场景下慎用。
- **`persistent_workers=True` 在 num_workers=0 时会报错**：仅在 num_workers > 0 时使用。
- **不要无脑把 batch 拉到 100% 显存**：留 10–20% 余量给 CUDA runtime 和临时 buffer。
