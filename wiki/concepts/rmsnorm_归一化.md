---
title: RMSNorm 归一化
tags: [概念, 归一化, Transformer, Qwen2.5-VL, LLM]
created: 2026-05-03
updated: 2026-05-04
sources: 2
status: active
---

# RMSNorm 归一化

## 模块整体说明与架构拆解

RMSNorm（Root Mean Square Layer Normalization）是 Qwen2.5-VL 中**视觉编码器与语言基座共用**的标准化机制。它是一种计算效率更高的 LayerNorm 变体。

### 为什么需要归一化？
在深度神经网络中，每一层的输出都会作为下一层的输入。随着层数加深，梯度的微小波动会被不断放大（梯度爆炸）或缩小（梯度消失）。归一化通过将张量数值映射到一个稳定的分布，确保模型训练的数值稳定性。

### 内部架构流转
在 `Qwen2_5_VLVisionBlock` 中，RMSNorm 采用 **Pre-Norm** 结构：

```mermaid
graph LR
    A["输入 hidden_states"] --> B["RMSNorm (norm1/norm2)"]
    B --> C["注意力层 / MLP层"]
    C --> D["残差连接: + 输入"]
    D --> E["下一层输入"]
```

---

## 逻辑链输入与输出

- **逻辑链（输入）**：`hidden_states` [seq_len, hidden_size] (例如 [N, 1152])。
- **逻辑链（输出）**：归一化后的 `hidden_states` [seq_len, hidden_size]。

---

## 核心算法原理详解

### 1. 从 LayerNorm 到 RMSNorm (第一性原理)

**传统 LayerNorm**：
它通过减去均值（Mean）并除以标准差来对齐分布。它的公式包含两个统计量：
$$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \times \gamma + \beta$$
- $E[x]$：均值对齐（平移）
- $Var[x]$：方差对齐（缩放）

**RMSNorm 的革新**：
研究表明，LayerNorm 的效果主要来自于**缩放（Scaling）**而非平移。RMSNorm 抛弃了均值计算，只计算均方根（Root Mean Square）：
$$RMS(x) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}$$
$$y = \frac{x}{RMS(x) + \epsilon} \times \gamma$$

**优点**：
1. **计算更省**：不需要计算均值，减少了加法和减法操作。
2. **不变性**：具备重缩放不变性（Re-scaling Invariance）。
3. **主流标准**：已成为 LLaMA、Qwen 等顶流多模态模型的标配。

---

## 核心源码解剖

**代码路径**：`transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`

```python
class Qwen2_5_VLRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        # 唯一的可训练参数：缩放权重 gamma
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # 为了数值精度，强制转为 float32 计算
        hidden_states = hidden_states.to(torch.float32)
        # 计算均方根 (x^2 的均值)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 归一化核心逻辑
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 应用可训练的 gamma 缩放并转回原精度
        return self.weight * hidden_states.to(input_dtype)
```

---

## 参数生命周期与渊源追溯

- **网络结构**：纯数学变换 + 单层可训练权重 `self.weight`。
- **参数量**：在 ViT 中为 1152 个参数；在 LLM 中为 4096 个参数。
- **演化追溯**：
  - Qwen2-VL 的视觉侧使用的是传统的 `LayerNorm`。
  - **Qwen2.5-VL 实现了“架构统一”**：将视觉侧的归一化全部升级为 `RMSNorm`，使其在数学性质上更接近 LLM 基座，有助于跨模态特征的顺滑对齐。
- **生命周期追踪**：
  - **训练阶段**：解冻，参与视觉语义的提纯。
  - **SFT阶段**：随 ViT 整体冻结。

---

## 关联概念

- ✅ 支持 [[window_attention_交错注意力]]：在 Attention 计算前进行 Pre-Norm。
- ✅ 支持 [[swiglu_门控激活函数]]：在 MLP 计算前进行 Pre-Norm。
- 🔄 演化自 T5 模型中的 LayerNorm 简化版。

## 参考来源

- `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- `knowledge_base/raw/面试官!从Qwen-VL到Qwen3.5技术改进？(26年2月版)/面试官!从Qwen-VL到Qwen3.5技术改进？(26年2月版) .md`
