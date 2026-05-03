---
title: 2D-RoPE 视觉位置编码
tags: [概念, 位置编码, RoPE, ViT, Qwen2.5-VL]
created: 2026-05-03
updated: 2026-05-03
sources: 3
status: active
---

# 2D-RoPE 视觉位置编码

## 模块整体说明

2D-RoPE 是 Qwen2-VL/Qwen2.5-VL 的 ViT 视觉编码器内部使用的位置编码方式。它将 1D-RoPE 扩展到二维空间，分别编码图像 Patch 的 X 轴和 Y 轴位置。

**它解决的核心问题**：传统 ViT 使用固定长度的可学习绝对位置编码，导致分辨率被锁死。2D-RoPE 基于位置动态计算旋转角度，天然支持任意分辨率，是实现 [[navit_动态分辨率]] 的关键技术。

**与 3D-MRoPE 的分工**：
- **ViT 内部**（图像特征提取阶段）→ 使用 2D-RoPE（只编码 H, W）
- **LLM 内部**（多模态融合阶段）→ 使用 [[mrope_多模态位置编码]]（编码 T, H, W）

---

## 逻辑链输入与输出

- **逻辑链（输入）**：每个 Patch 的 (行号, 列号) 位置坐标。
- **逻辑链（输出）**：`cos` 和 `sin` 旋转矩阵，用于旋转 ViT Attention 的 Q 和 K。

---

## 核心算法原理详解

### 实现原理

将 `head_dim` 均分为两半：前半部分编码 X 轴位置，后半部分编码 Y 轴位置。每半部分内独立应用标准的 1D-RoPE 旋转。

```python
head_dim = config.embed_dim // config.num_heads  # 例如 1152/16 = 72
self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)  # 36 维
```

**数值示例**：`head_dim = 72`，则 `rotary_dim = 36`。
- 前 36 维：用 X 坐标计算旋转角度
- 后 36 维：用 Y 坐标计算旋转角度

对于位置 $(x=3, y=5)$ 的 Patch，前 36 维用 $m=3$ 计算 $3\theta_i$，后 36 维用 $m=5$ 计算 $5\theta_i$。

### 为什么沿用 Qwen2-VL 的 2D-RoPE？

Qwen2.5-VL 在 ViT 内部使用的**和 Qwen2-VL 完全一样**的 2D-RoPE，没有做任何改动。原因是 2D-RoPE 已经足够好：ViT 只需要编码平面空间位置，不涉及时间维度（时间信息由 [[conv3d_时空切块器]] 的 2× 合并和 LLM 端的 [[mrope_多模态位置编码]] 分别处理）。

---

## 关联概念

- ✅ 支持 [[qwen2.5_vl_技术报告解析]]：ViT 内部沿用 2D-RoPE。
- 🔄 演化自 Qwen2-VL 中首次引入的 2D-RoPE（完全相同）。
- [[mrope_多模态位置编码]]：LLM 内部的三维扩展。
- [[navit_动态分辨率]]：2D-RoPE 是实现动态分辨率的关键技术。

## 参考来源

- 原始资料：`knowledge_base/Qwen2.5-VL/Qwen2.5-VL.md` §1.1.3
