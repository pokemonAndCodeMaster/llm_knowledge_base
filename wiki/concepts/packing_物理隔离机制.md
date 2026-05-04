---
title: Packing 物理隔离机制 (cu_seqlens)
tags: [概念, Packing, NaViT, Qwen2.5-VL, 物理隔离, cu_seqlens]
created: 2026-05-04
updated: 2026-05-04
sources: 3
status: active
---

# Packing 物理隔离机制 (cu_seqlens)

## 模块整体说明与架构拆解

Packing 物理隔离机制是对 NaViT 核心思想的工程落地。在 Qwen2.5-VL 的视觉骨干网中，为了最大化 GPU 的吞吐量，系统不对不同尺寸、不同分辨率的图像进行补零（Padding），而是将一个 Batch 内多张图的视觉 Patch 拼接打包成一个一维的长序列（即 Packing）。

本机制的核心作用是**利用 `cu_seqlens` 建立一道“数学物理围墙”**，防止在 Attention 计算时，图片 A 的像素（Query）“看到”并去查询图片 B 的像素（Key/Value），从而解决不同异构图像特征互相污染的问题。

### 全局代码调用顺序与流转概览
1.  **元数据摄入**：在 `Qwen2_5_VLVisionTransformer.forward` 的最开始，接收来自外部的 `grid_thw` 网格张量。
2.  **构建围墙**：利用 `grid_thw` 计算出 `cu_seqlens`（每个媒体项的累积序列长度偏移量）。
3.  **下发传递**：这个 `cu_seqlens` 被作为关键参数传入各层 `Qwen2_5_VLVisionBlock`，并最终传递到 `Qwen2_5_VLVisionAttention` 中。
4.  **内核执行**：Flash Attention 根据 `cu_seqlens` 进行硬件级别的掩码（Masking）隔离运算。

---

## 子模块/步骤详解

### 1. 数据打包与边界划分 (Packing & cu_seqlens)

#### 模块说明
计算出每个媒体项（图片或视频的时间步）在长序列中的绝对边界位置。

#### 逻辑链输入与输出
- **逻辑链（输入）**：`grid_thw` [Num_Media, 3]。张量每一行记录了对应媒体的高维信息 `(T, H, W)`。
- **逻辑链（输出）**：`cu_seqlens` [Num_Media + 1]。累积的序列偏移坐标，数据类型为 `torch.int32`。

#### 第一性原理与原理解读
*   **为什么要隔离？** 传统的 Transformer 是针对补齐（Padding）到相同长度的序列设计的，不同序列在 Batch 维度天然隔离。但当所有变长的图片被 Flatten 并 `cat` 到同一个维度后，注意力矩阵 $QK^T$ 会变成一个庞大的全局矩阵。如果不加掩码，图片 A 的特征就会被图片 B 影响，这是物理意义上的“空间错乱”。
*   **`cu_seqlens` 的本质**：它是 Cumulative Sequence Lengths 的缩写。它是一组从 0 开始的累加绝对坐标。通过这组坐标，计算后端（如 Flash Attention）就能知道：第 0 到 1024 个 Token 属于图 A，第 1024 到 2560 个 Token 属于图 B。
*   **不仅隔离空间，还隔离时间**：Qwen2.5-VL 认为视觉骨干网的任务是纯粹的“空间提纯”。因此，视频的 $T$ 个时间步不仅不会互相 Attention，反而在算 `cu_seqlens` 时，会被“拆成” $T$ 个独立的图像帧进行空间隔离。

#### 核心源码解剖与公式推导
**计算公式**：
$$cu\_seqlens = cumsum([S_1, S_1, ..., S_2, S_2, ...])$$
其中 $S_i = H_i \times W_i$ 为第 $i$ 个媒体项每一帧的空间 Token 数。如果有 $T_i$ 帧，就会重复 $T_i$ 次。

**代码实现** (`transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`):
```python
# 计算每张图（每一帧）的空间 Patch 数: H * W
spatial_len = grid_thw[:, 1] * grid_thw[:, 2]

# 重复 T 次，将视频的每一帧也作为独立的实体进行隔离
repeated_len = torch.repeat_interleave(spatial_len, grid_thw[:, 0])

# 计算累加和，得到绝对偏移量
cu_seqlens = repeated_len.cumsum(0, dtype=torch.int32)

# 在开头补 0，形成完整的起始和结束区间，例如 [0, 1024, 2560]
cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
```

### 2. 工业级硬件隔离实现 (Flash Attention / Split)

#### 模块说明
将构建好的数学围墙真正作用于计算引擎，实现底层的物理隔离。

#### 具体操作逻辑拆解
1.  **模式 A（硬件掩码）**：若启用 Flash Attention，直接将 `cu_seqlens` 传给底层 CUDA 内核。内核在扫描矩阵乘法时，发现越界就会将外围元素置为 $-\infty$，实现真正的零开销隔离。
2.  **模式 B（手动拆分）**：若显卡不支持，系统则利用 `cu_seqlens` 前后两项相减得到每个媒体的真实长度，然后通过 `torch.split` 物理切割出若干独立的 Tensor，分别做常规 Attention 后，再 `torch.cat` 回去。

---

## Packing 隔离与 Window 注意力的关系

**这是一个极其重要的认知边界：**
*   **Packing 隔离 (`cu_seqlens`)** 解决的是**“图与图之间”**以及**“帧与帧之间”**不互串的问题，属于**全局级别的物理硬隔离**。它贯穿整个 32 层 Backbone。
*   **Window 注意力 (`cu_window_seqlens`)** 解决的是**“单张图内部”**因为分辨率太大导致算力爆炸的问题。它是在 `cu_seqlens` 已经切分好的“图片专属空间”中，**再进一步**用更小的网格（如 8x8）去划分的小隔离墙。
*   因此，`cu_window_seqlens` 必然是建立在 `cu_seqlens` 基础之上的“二次约束”。

---

## 关联概念

- ✅ 支持 [[window_attention_交错注意力]]：为其提供大前提的图像边界，使得 Window 划分不会跨图片。
- 🔄 演化自 [[navit_动态分辨率]]：这是 NaViT Packing 思想的最终计算落地点。
