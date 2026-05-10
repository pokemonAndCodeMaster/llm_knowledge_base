---
title: Qwen3.5 文本嵌入与特殊 Token
tags: [Qwen3.5, Embedding, 文本嵌入, 特殊Token, ChatML, vocab]
created: 2026-05-10
updated: 2026-05-10
sources: 2
status: active
---

# Qwen3.5 文本嵌入与特殊 Token

## 模块整体说明与架构拆解

文本嵌入层（`embed_tokens`）是 Qwen3.5 前向传播的第一个神经网络操作。它将离散的 integer token ID 通过查表映射为连续的高维向量。此阶段产生的图像占位符 embedding 是**语义无效的**，会在后续阶段④被视觉特征替换。

### 推演样本

沿用 [[qwen3.5_前向传播全链路]] 的统一样本。

### 全局代码调用顺序

```
Qwen3_5Model.forward()
  文件: models/qwen3_5/modular_qwen3_5.py:612
  │
  └─ inputs_embeds = self.get_input_embeddings()(input_ids)
     │  = nn.Embedding(248320, 4096)(input_ids)
     │
     │  输入: input_ids (1, 921) — 整数序列
     │  输出: inputs_embeds (1, 921, 4096) — 浮点向量序列
     │
     │  其中:
     │  ├─ 位置 0-2: <|im_start|>, user, \n → 有效文本 embedding
     │  ├─ 位置 3: <|vision_start|> → 有效特殊 token embedding
     │  ├─ 位置 4-259: <|image_pad|> × 256 → 占位值（将被图像特征替换）
     │  ├─ 位置 260: <|vision_end|> → 有效特殊 token embedding
     │  ├─ 位置 261: \n
     │  ├─ 位置 262: <|vision_start|> → 有效特殊 token embedding
     │  ├─ 位置 263-906: <|video_pad|> × 644 → 占位值（将被视频特征替换）
     │  ├─ 位置 907: <|vision_end|> → 有效特殊 token embedding
     │  └─ 位置 908-920: \n描述这张图片和视频... → 有效文本 embedding
```

---

## 子模块详解

### 1. nn.Embedding 查表原理

#### 模块说明

`nn.Embedding(V, D)` 内部维护一个形状为 `(V, D)` 的权重矩阵 $W \in \mathbb{R}^{V \times D}$。给定 token ID $i$，输出就是 $W[i]$（矩阵的第 $i$ 行）。

这与视觉编码器的 Conv3d **完全不同**：
- **文本嵌入 = 查表**：离散符号 → 直接索引取行，$O(1)$ per token
- **视觉嵌入 = 滤波**：连续像素 → 卷积加权求和，$O(K^3)$ per patch

#### 核心源码

```python
self.embed_tokens = nn.Embedding(
    config.vocab_size,    # 248320
    config.hidden_size,   # 4096
    padding_idx=None      # Qwen3.5 不使用 padding_idx
)

# forward 中:
inputs_embeds = self.embed_tokens(input_ids)
# input_ids: (1, 921) int64
# 输出: (1, 921, 4096) float16/bfloat16
```

#### 参数量

`248320 × 4096 = 1,017,118,720` ≈ **1.02B 参数**（约占模型总参数的 3.8%@27B）

这是模型中**最大的单层权重矩阵**之一（与 `lm_head` 共享或独立）。

### 2. ChatML 模板与 Token 序列结构

#### 模块说明

Qwen3.5 使用 **ChatML** 格式组织对话。chat template 将用户的原始文本展开为结构化序列：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|>×256<|vision_end|>
<|vision_start|><|video_pad|>×644<|vision_end|>
描述这张图片和视频<|im_end|>
<|im_start|>assistant
```

#### 推演（逐 token 拆解）

| 位置 | Token | ID (示例) | mm_type | 说明 |
|------|-------|----------|---------|------|
| 0 | `&lt;&#124;im_start&#124;&gt;` | 151644 | 0 | 对话轮次开始 |
| 1 | `user` | 872 | 0 | 角色标识 |
| 2 | `\n` | 198 | 0 | 换行 |
| 3 | `&lt;&#124;vision_start&#124;&gt;` | 151652 | 0 | 视觉区域开始 |
| 4-259 | `&lt;&#124;image_pad&#124;&gt;` | 248056 | 1 | 图像占位符 ×256 |
| 260 | `&lt;&#124;vision_end&#124;&gt;` | 151653 | 0 | 视觉区域结束 |
| 261 | `\n` | 198 | 0 | 换行 |
| 262 | `&lt;&#124;vision_start&#124;&gt;` | 151652 | 0 | 视觉区域开始 |
| 263-906 | `&lt;&#124;video_pad&#124;&gt;` | 248057 | 2 | 视频占位符 ×644 |
| 907 | `&lt;&#124;vision_end&#124;&gt;` | 151653 | 0 | 视觉区域结束 |
| 908 | `\n` | 198 | 0 | 换行 |
| 909-916 | `描述这张图片和视频` | varies | 0 | 用户文本 |
| 917 | `&lt;&#124;im_end&#124;&gt;` | 151645 | 0 | 对话轮次结束 |
| 918 | `\n` | 198 | 0 | 换行 |
| 919 | `&lt;&#124;im_start&#124;&gt;` | 151644 | 0 | 助手轮次开始 |
| 920 | `assistant` | 14776 | 0 | 角色标识 |

### 3. 特殊 Token 角色定义

#### 模块说明

Qwen3.5 词表中定义了多个特殊 token，各有不同的语义功能：

| Token | ID | 功能 |
|-------|-----|------|
| `&lt;&#124;im_start&#124;&gt;` | 151644 | ChatML 对话轮次边界（开始） |
| `&lt;&#124;im_end&#124;&gt;` | 151645 | ChatML 对话轮次边界（结束），生成时停止信号 |
| `&lt;&#124;vision_start&#124;&gt;` | 151652 | 视觉内容区域开始标记 |
| `&lt;&#124;vision_end&#124;&gt;` | 151653 | 视觉内容区域结束标记 |
| `&lt;&#124;image_pad&#124;&gt;` | 248056 | 图像占位符（被 masked_scatter 替换） |
| `&lt;&#124;video_pad&#124;&gt;` | 248057 | 视频占位符（被 masked_scatter 替换） |

**关键设计**：`<|vision_start|>` 和 `<|vision_end|>` 的 embedding **不会**被替换 — 它们保留原始的可学习 embedding，充当视觉内容的「括号」，让模型知道这里的 token 来自图像/视频而非文本。

### 4. 占位 Embedding 的命运

在 `embed_tokens` 阶段，256 个 `<|image_pad|>` token 也会获得各自的 embedding 向量。但这些向量是**语义无效的占位值**：

```python
# 阶段②: embed_tokens 查表 → 256 个占位 embedding
inputs_embeds[0, 4:260, :] = embed_tokens.weight[248056]  # 相同的行向量重复256次

# 阶段④: masked_scatter 替换 → 256 个视觉 embedding
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
# 此时占位值被完全覆盖，不再参与后续计算
```

**因此 `<|image_pad|>` 的 embedding 权重在训练中几乎不会收到有效梯度** — 它的存在仅仅是为了让 Embedding 层能正常运行（不能传入不存在的 ID）。

---

## 第一性原理：为什么不直接用视觉特征构建 inputs_embeds？

理论上可以跳过 Embedding 查表，直接将 `pixel_values → ViT → 视觉embedding` 和 `文本 token → Embedding` 拼接成 `inputs_embeds`。但 Qwen 的设计选择是**先全部查表，再替换占位符**，原因是：

1. **代码简洁**：不需要手动管理不同模态的拼接顺序和 padding
2. **灵活性**：支持任意位置的视觉 token 插入（不仅仅是序列头部）
3. **兼容性**：纯文本推理时完全不需要视觉分支，Embedding 层就是全部

---

## 关联概念

- [[conv3d_时空切块器#第一性原理深度对比：视觉 vs 文本]] — 查表 vs 滤波对比 ✅ 支持
- [[qwen3.5_processor预处理]] — 上游：生成 input_ids ✅ 支持
- [[qwen3.5_多模态融合机制]] — 下游：masked_scatter 替换占位 embedding ✅ 支持
- [[qwen3.5_前向传播全链路]] — 全链路流转 ✅ 支持

## 参考来源

- `transformers/src/transformers/models/qwen3_5/modular_qwen3_5.py:612`
- `transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1368-1400`
