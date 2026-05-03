---
title: MRoPE 多模态旋转位置编码
tags: [概念, 位置编码, RoPE, 多模态, Qwen2.5-VL, LLM]
created: 2026-05-03
updated: 2026-05-03
sources: 4
status: active
---

# MRoPE 多模态旋转位置编码

## 模块整体说明

MRoPE (Multimodal Rotary Positional Embedding) 是 Qwen 多模态系列模型的**灵魂组件**，负责在 LLM 骨干网内部为混合了文本和视觉的序列提供三维时空位置信息。

**它解决的核心问题**：在多模态场景中，一个 Token 可能带有三个空间属性——时间 $T$（视频第几秒）、高度 $H$（图像的第几行）、宽度 $W$（图像的第几列）。传统的 1D-RoPE 只有一个位置维度，无法表达这种三维结构。MRoPE 就是把 1D-RoPE 扩展到三维的方案。

**直观比喻**：1D-RoPE 就像一条直线上的门牌号——"第 1 号、第 2 号、第 3 号……"。但图像上的 Token 不是排成一条线的，而是排成一个平面网格（有行有列），视频的 Token 更是排成一个三维矩阵（有帧有行有列）。MRoPE 就像一个三维坐标系的地址——"3 楼 5 排 7 座"（时间=3，高度=5，宽度=7），让模型精确知道每个 Token 来自哪里。

**在全链路中的位置**：MRoPE 的位置 ID 在 Processor 阶段就计算好了（通过 `get_rope_index` 方法），但实际注入是在 LLM 骨干网的 Attention 计算中（作用在 Q 和 K 上）。

---

## 逻辑链输入与输出

- **逻辑链（输入）**：`position_ids` 张量，形状 `[3, Batch, Seq_len]`（3 代表 T, H, W 三个坐标）。
- **逻辑链（输出）**：`cos` 和 `sin` 旋转矩阵，形状 `[Batch, Seq_len, Dim_rot]`，用于旋转 Q 和 K。

---

## 核心算法原理详解

### 1. RoPE 基础回顾

RoPE 的核心思想：**用复数乘法（或等价的二维旋转矩阵）来表示绝对位置，从而在 Q·K 内积中自然地实现相对位置表达**。

对于位置 $m$，特征维度的第 $i$ 对，旋转角度为 $m\theta_i$，其中逆频率 $\theta_i$ 按维度衰减：

$$\theta_i = base^{-2i/d}$$

位置 $m$ 的旋转编码应用到特征向量 $x$ 的第 $2i$ 和 $2i+1$ 维度上：

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

**关键性质**：位置 $m$ 的 Q 和位置 $n$ 的 K 做内积时，结果只依赖于**相对距离** $m-n$。这就实现了"编码绝对位置，但 Attention 自动感知相对位置"。

### 2. 从 1D-RoPE 到 3D-MRoPE

**设计思路**：不把所有的特征维度都拿来做 T, H, W 的旋转。而是**将用于旋转编码的维度切分为 3 段**。

在 Qwen2.5-VL 的配置中，`mrope_section` 参数为 `[16, 24, 24]`，共 64 维（即 `head_dim // 2 = 128 // 2 = 64`）：
- 前 16 维：编码时间 $T$ 的位置
- 中间 24 维：编码高度 $H$ 的位置
- 后 24 维：编码宽度 $W$ 的位置

每段独立计算各自的旋转频率：
```
freqs_T = compute_rope(position_ids[0], theta, dims=16)  # 时间维频率
freqs_H = compute_rope(position_ids[1], theta, dims=24)  # 高度维频率
freqs_W = compute_rope(position_ids[2], theta, dims=24)  # 宽度维频率
freqs = concat(freqs_T, freqs_H, freqs_W)  # 拼接：[Batch, Seq_len, 64]
```

最后 `cat(freqs, freqs)` 镜像拼接到 128 维，分别求 `cos` 和 `sin`，用于旋转 Q 和 K。

### 3. 三种输入类型的位置 ID 分配

**纯文本**：三个维度使用**相同的位置 ID**，等价于 1D-RoPE。
```
text:  T=[101, 102, 103], H=[101, 102, 103], W=[101, 102, 103]
```

**静态图像**：时间 ID 不变（只有一帧），H 和 W 按网格位置分配。
```
image: T=[0, 0, 0, 0], H=[0, 0, 1, 1], W=[0, 1, 0, 1]
```

**视频**：三个维度都有变化。

### 4. Qwen2.5-VL 核心创新：绝对时间对齐 (Time-Absolute MRoPE)

**来龙去脉**：在 Qwen2-VL 的 naive MRoPE 中，视频的时间 ID 只是简单递增（0, 1, 2, 3...），不反映真实的物理时间间隔。一个 2fps 的视频和一个 30fps 的视频，同样的 4 帧，时间 ID 都是 0-3，但前者跨越了 2 秒，后者只跨越了 0.13 秒。模型无法区分这种时间尺度差异。

**Qwen2.5-VL 的突破**：MRoPE 的 Temporal 维度对齐了物理时间（秒）。核心公式：

$$interval = tokens\_per\_second \times \frac{temporal\_patch\_size}{fps}$$

- `tokens_per_second`：每秒的时间 Token 数（超参数，如 25），定义时间粒度
- `temporal_patch_size`：每个时间 Patch 包含多少帧（2）
- `fps`：当前视频的帧率

**数值示例**：
```
fps=1, tokens_per_second=25, temporal_patch_size=2
interval = 25 × 2 / 1 = 50

3 个时间 Patch, 每帧 2×2 = 4 个空间 Token:
vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
vision height position_ids:   [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
vision width position_ids:    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
text temporal position_ids:   [101, 102, 103, 104, 105]
text height position_ids:     [101, 102, 103, 104, 105]
text width position_ids:      [101, 102, 103, 104, 105]
```

文本起始位置 = max(视觉位置) + 1 = 100 + 1 = 101。

### 可训练参数与训练方式

RoPE 的旋转矩阵（cos/sin 值）是基于固定公式计算的，**不包含可训练参数**。但它直接作用在 Attention 的可训练参数 $Q, K$ 上，通过改变 Q 和 K 的方向来注入位置信息。

---

## 源码逐行解剖

**代码路径**：`transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`

```python
def get_rope_index(
    self,
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,  # 每个时间步对应多少秒
    attention_mask: Optional[torch.Tensor] = None,
):
    # ...
    if modality_type == 2:  # 2 代表视频
        # 基于物理真实秒数的绝对时间对齐
        time_interval = tokens_per_second * int(next(second_per_grid_ts))
    # ...

# 在模型前向传播中：
vision_outputs = self.visual(pixel_values, ...)
image_embeds = vision_outputs.pooler_output  # [seq_len/4, 4096]

# special_image_mask 是从 input_ids 算出的布尔掩码
inputs_embeds[special_image_mask] = image_embeds  # 视觉特征替换占位符
```

---

## 版本演化对比

| 版本 | MRoPE 特性 | 频率排布 |
|------|-----------|---------|
| Qwen2-VL | 首次引入 3D-MRoPE（时间步均匀递增） | 分段 (Chunked) |
| **Qwen2.5-VL** | **Time-Absolute MRoPE（对齐物理秒数）** | 分段 (Chunked) |
| Qwen3-VL | 继续绝对时间 + **MRoPE-Interleave** | **交织** `[T1,H1,W1,T2,H2,W2...]` |
| Qwen3.5 | 继续 + Partial RoPE（只有部分维度旋转） | 交织 |

---

## 关联概念

- ✅ 支持 [[qwen2.5_vl_技术报告解析]]：绝对时间对齐是 LLM 端的核心改进。
- 🔄 演化自 Qwen2-VL 的 naive MRoPE（时间步不对齐物理时间）。
- 与 [[2d_rope_视觉位置编码]] 协作：ViT 内部用 2D-RoPE，LLM 内部用 3D-MRoPE。
- 上游：位置 ID 在 Processor 阶段计算，实际注入在 LLM Attention 中。
- 下游：位置信息影响 LLM 的 Q·K Attention 权重分布。
- 替代了早期 [[qwen_vl_projector]] 中在 Projector 内使用 2D 绝对位置编码的方案。

## 参考来源

- 原始资料：`knowledge_base/Qwen2.5-VL/Qwen2.5-VL.md`
- 前置知识：`qwen_learning_guide_phase0.md` §4 MRoPE
- 学习指南：`knowledge_base/Qwen_Architecture_Guides/qwen_learning_guide_phase1.md`
