---
title: RMSNorm 归一化
tags: [概念, 归一化, Transformer, Qwen2.5-VL]
created: 2026-05-03
updated: 2026-05-03
sources: 3
status: active
---

# RMSNorm 归一化

## 模块整体说明

RMSNorm (Root Mean Square Layer Normalization) 是 Qwen2.5-VL 中全面采用的归一化方案，用于替代传统的 LayerNorm。它出现在 ViT 的每一个 VisionBlock（`norm1`, `norm2`）和 [[patchmerger_空间降维]] 的归一化层中。

**它解决的核心问题**：Transformer 在训练过程中，隐藏层的激活值分布会不断漂移，导致梯度消失或爆炸。归一化通过强制将分布拉回标准范围来解决这个问题。

**直观比喻**：想象一群学生的考试成绩（激活值）。LayerNorm 是先算平均分和标准差，然后把每个人的成绩转化为"偏离平均多少个标准差"。RMSNorm 则简化了——不算平均分，只算"均方根"（一个衡量整体成绩大小的指标），然后把每个人的成绩除以这个值。省去了求平均的步骤，速度更快。

---

## 逻辑链输入与输出

- **逻辑链（输入）**：`[seq_len, hidden_size]` 的特征张量。
- **逻辑链（输出）**：同维度 `[seq_len, hidden_size]`，分布被标准化。

---

## 核心算法原理详解

### 1. 传统 LayerNorm vs RMSNorm

**LayerNorm**（Qwen2-VL 中使用）：
$$LayerNorm(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$
需要计算均值 $\mu$ 和方差 $\sigma^2$，以及两个可学习参数 $\gamma$（缩放）和 $\beta$（偏移）。

**RMSNorm**（Qwen2.5-VL 中使用）：
$$RMSNorm(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot w$$

- 省去了均值的计算（$\mu$），直接用均方根。
- 省去了偏移参数（$\beta$），只保留缩放参数 $w$。
- $w$ 初始化为全 **1**（恒等映射）。

**为什么可以省去均值？** 论文（Zhang & Sennrich, 2019）发现，LayerNorm 的性能主要来自缩放不变性（除以方差），而不是平移不变性（减去均值）。去掉均值后，计算量减少约 10%，且性能几乎无损。

### 2. 数值计算示例

假设 `hidden_size = 4`，输入向量 $x = [2.0, -1.0, 0.5, 1.5]$：

```
Step 1: 计算均方 (mean of squares)
  x² = [4.0, 1.0, 0.25, 2.25]
  mean(x²) = (4.0 + 1.0 + 0.25 + 2.25) / 4 = 1.875

Step 2: 计算均方根 (RMS)
  RMS = √(1.875 + 1e-6) ≈ 1.3693

Step 3: 除以 RMS
  x_norm = [2.0/1.3693, -1.0/1.3693, 0.5/1.3693, 1.5/1.3693]
         = [1.4606, -0.7303, 0.3651, 1.0954]

Step 4: 乘以可学习权重 w（初始全为 1）
  output = w * x_norm = [1.4606, -0.7303, 0.3651, 1.0954]
```

### 可训练参数与训练方式

- **网络结构**：一个 `nn.Parameter`，形状为 `(hidden_size,)`。
- **初始化**：全部为 **1**（`torch.ones(hidden_size)`）。
- **训练状态**：随所在模块一起训练。在 ViT 中，Stage 1-3 可训练，后训练冻结。

---

## 源码逐行解剖

**代码路径**：`transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`

```python
class Qwen2_5_VLRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # 可学习缩放权重，初始化为全 1
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)  # 升精度防止溢出

        # 核心：计算均方，求倒数平方根
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 乘以可学习权重
        return self.weight * hidden_states.to(input_dtype)
```

---

## 后续演进：Zero-Centered RMSNorm

在 Qwen3-Next 和 Qwen3.5 中，RMSNorm 进一步演化为 **Zero-Centered RMSNorm**：

$$ZeroCentered\_RMSNorm(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot (1 + w)$$

- 权重 $w$ 初始化为全 **0**（而非全 1）。
- 计算时乘以 $(1 + w)$，初始时等价于恒等映射。
- **架构意义**：初始时分布严格约束在零中心，极大缓解了混合注意力（特别是 Linear Attention 的 Q/K 归一化）中的数值不稳定。

---

## 版本演化对比

| 版本 | 归一化 | 权重初始值 | 计算公式 |
|------|--------|-----------|---------|
| Qwen2-VL ViT | LayerNorm | γ=1, β=0 | $(x-\mu)/\sigma \cdot \gamma + \beta$ |
| **Qwen2.5-VL ViT** | **RMSNorm** | **w=1** | $x/RMS \cdot w$ |
| Qwen3-VL ViT | LayerNorm (回退) | γ=1, β=0 | 因 DeepStack 需要均值去中心化 |
| Qwen3.5 LLM | **Zero-Centered RMSNorm** | **w=0** | $x/RMS \cdot (1+w)$ |

---

## 关联概念

- ✅ 支持 [[qwen2.5_vl_技术报告解析]]：ViT 归一化从 LayerNorm 升级为 RMSNorm 是架构向 LLM 靠拢的标志。
- 与 [[swiglu_门控激活函数]] 配合使用于 VisionBlock 中（Pre-Norm 架构）。
- 与 [[window_attention_交错注意力]] 组成 VisionBlock 的完整结构。
- 在 [[patchmerger_空间降维]] 中用作归一化层（`ln_q`）。

## 参考来源

- 原始资料：`knowledge_base/Qwen2.5-VL/Qwen2.5-VL.md`
- 前置知识：`qwen_learning_guide_phase0.md` §3 归一化
