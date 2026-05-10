---
title: Qwen3.5 Processor 预处理
tags: [Qwen3.5, Processor, 预处理, mm_token_type_ids, 帧时间戳, 占位符]
created: 2026-05-10
updated: 2026-05-10
sources: 2
status: active
---

# Qwen3.5 Processor 预处理 (Qwen3VLProcessor)

## 模块整体说明与架构拆解

Qwen3.5 复用 Qwen3-VL 的 `Qwen3VLProcessor`，它在 CPU 上将异构原始数据（图片、视频、文本）转换为模型可消费的标准化张量。不含任何神经网络权重。

与 Qwen2.5-VL Processor（详见 [[qwen2.5_vl_预处理流水线]]）的核心差异：
1. **新增 `mm_token_type_ids`**：为每个 token 标记模态类型
2. **新增视频帧时间戳**：将帧索引转换为物理秒数插入 prompt
3. **视频占位符结构变化**：每帧独立包裹在 `<|vision_start|>...<|vision_end|>` 中

### 推演样本

沿用 [[qwen3.5_前向传播全链路]] 的统一样本：1 张 448×448 图 + 1 段 4 帧（392×644）视频 + "描述这张图片和视频"。

### 全局代码调用顺序与流转概览

```
Qwen3VLProcessor.__call__(images, text, videos)
  文件: models/qwen3_vl/processing_qwen3_vl.py:78
  │
  ├─ 1. image_inputs = self.image_processor(images)
  │     输出: pixel_values (1024, 3, 2, 14, 14)
  │           image_grid_thw (1, 3) = [1, 32, 32]
  │
  ├─ 2. 图像占位符展开
  │     merge_length = merge_size² = 4
  │     num_image_tokens = grid_thw.prod() // merge_length
  │                      = (1×32×32) // 4 = 256
  │     text中 <|image_pad|> → 替换为 256 个 <|placeholder|>
  │     → 再还原为 256 个 <|image_pad|>
  │
  ├─ 3. (如有视频) 视频占位符展开 + 帧时间戳插入
  │     详见下方 §视频帧时间戳
  │
  ├─ 4. text_inputs = self.tokenizer(text)
  │     输出: input_ids (1, 272)
  │
  ├─ 5. mm_token_type_ids = create_mm_token_type_ids(input_ids)
  │     输出: (1, 272) — 0=文本, 1=图像, 2=视频
  │
  └─ 6. return BatchFeature({...所有张量...})
```

---

## 子模块详解

### 1. mm_token_type_ids — 模态类型标记

#### 模块说明

`mm_token_type_ids` 是 Qwen3-VL/3.5 新引入的张量，为序列中每个 token 标记其所属模态：

| 值 | 含义 | 示例 token |
|----|------|-----------|
| 0 | 文本 | `&lt;&#124;im_start&#124;&gt;`, `user`, `描`, `述`, `&lt;&#124;vision_start&#124;&gt;`, `&lt;&#124;vision_end&#124;&gt;` |
| 1 | 图像 | `&lt;&#124;image_pad&#124;&gt;` |
| 2 | 视频 | `&lt;&#124;video_pad&#124;&gt;` |

**注意**：`<|vision_start|>` 和 `<|vision_end|>` 本身被标记为文本（0），只有中间的 pad token 才是图像/视频。

#### 推演

输入序列（简化）：`[<im_s>, user, \n, <vs>, <img>×256, <ve>, \n, <vs>, <vid>×644, <ve>, \n, 描, 述, ...]`

```
mm_token_type_ids: [0, 0, 0, 0, 1,1,...,1, 0, 0, 0, 2,2,...,2, 0, 0, 0, 0, ...]
                                  ↑ 256个1              ↑ 644个2
```

#### 下游用途

`mm_token_type_ids` 被传递给 `Qwen3_5Model.forward()` → `get_rope_index()`，用于按模态分组计算 3D 位置坐标：
- 模态=0 的连续段：T=H=W 同步递增（退化为 1D）
- 模态=1 的连续段：按 `image_grid_thw` 展开为 2D 空间坐标
- 模态=2 的连续段：按 `video_grid_thw` 展开为 3D 时空坐标

详见 [[qwen3.5_多模态融合机制#4. get_rope_index — 逐 token 位置分配]]。

#### 核心源码

```python
# 文件: models/qwen3_vl/processing_qwen3_vl.py:182-183
if return_mm_token_type_ids:
    text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

# create_mm_token_type_ids 逻辑（继承自 ProcessorMixin）：
# 扫描 input_ids，将 == image_token_id 的位置标为 1，== video_token_id 的位置标为 2
```

### 2. 图像占位符展开

#### 模块说明

Processor 在分词前，根据 `image_grid_thw` 计算每张图的视觉 token 数，将文本中的单个 `<|image_pad|>` 展开为精确数量的占位符。

#### 核心源码

```python
# 文件: models/qwen3_vl/processing_qwen3_vl.py:127-135
if image_grid_thw is not None:
    merge_length = self.image_processor.merge_size ** 2  # 2² = 4
    index = 0
    for i in range(len(text)):
        while self.image_token in text[i]:
            num_image_tokens = image_grid_thw[index].prod() // merge_length
            # = (1×32×32) // 4 = 256
            text[i] = text[i].replace(
                self.image_token,
                "<|placeholder|>" * num_image_tokens,  # 256 个
                1  # 只替换第一个匹配
            )
            index += 1
        text[i] = text[i].replace("<|placeholder|>", self.image_token)
        # 还原: 256个 <|placeholder|> → 256个 <|image_pad|>
```

**为什么要先替换再还原？** 因为原文本中可能有多个 `<|image_pad|>`（多张图），每个需要展开为不同数量。先用 `<|placeholder|>` 占位可以避免后续 `replace` 误替换已展开的 token。

### 3. 视频帧时间戳 (_calculate_timestamps)

#### 模块说明

Qwen3.5 的视频处理引入了**帧级时间戳**：将每帧的物理时间（秒）插入到 prompt 中，让模型感知视频的时间结构。

#### 核心源码

```python
# 文件: models/qwen3_vl/processing_qwen3_vl.py:257-268
def _calculate_timestamps(self, indices, video_fps, merge_size=2):
    # indices: 抽帧后的帧索引列表，如 [0, 4, 8, 12]
    # video_fps: 视频原始帧率，如 24

    # 1. 对齐到 temporal_patch_size 的倍数
    if len(indices) % merge_size != 0:
        indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))

    # 2. 帧索引 → 物理秒数
    timestamps = [idx / video_fps for idx in indices]
    # 如 [0/24, 4/24, 8/24, 12/24] = [0.0, 0.167, 0.333, 0.5]

    # 3. 每 merge_size 帧取均值（因为 temporal_patch_size=2，两帧合一）
    timestamps = [
        (timestamps[i] + timestamps[i + merge_size - 1]) / 2
        for i in range(0, len(timestamps), merge_size)
    ]
    # 如 [(0.0+0.167)/2, (0.333+0.5)/2] = [0.083, 0.417]
    return timestamps
```

#### 推演（4 帧视频，24fps）

输入：`indices=[0, 4, 8, 12]`, `fps=24`, `merge_size=2`
1. 对齐：4 帧已对齐（4 % 2 == 0）
2. 秒数：`[0.0, 0.167, 0.333, 0.5]`
3. 合并：`[(0.0+0.167)/2, (0.333+0.5)/2]` = `[0.083, 0.417]`

生成的视频 prompt：
```
<0.1 seconds><|vision_start|><|video_pad|>×322<|vision_end|>
<0.4 seconds><|vision_start|><|video_pad|>×322<|vision_end|>
```

每帧的 N = `grid_thw[1:].prod() // merge_length`（空间维度的 token 数）。本例中为 `28×46 // 4 = 322`。

### 4. 与 Qwen2.5-VL Processor 的差异对比

| 维度 | Qwen2.5-VL | Qwen3.5 |
|------|-----------|---------|
| mm_token_type_ids | ❌ 不存在 | ✅ 标记每个 token 的模态类型 |
| 视频帧时间戳 | ❌ 不插入时间信息 | ✅ `<X.X seconds>` 物理时间戳 |
| 视频占位符结构 | 单个 `<video>` 展开为整体 | 每帧独立 `<vs>...<ve>` 包裹 |
| 图像占位符 | 相同：按 grid_thw 展开 | 相同 |
| ImageProcessor | `Qwen2_5_VLImageProcessor` | `Qwen3VLImageProcessor`（兼容） |

---

## 第一性原理：为什么需要 mm_token_type_ids？

在 Qwen2.5-VL 中，`get_rope_index` 通过扫描 `input_ids` 中的特殊 token（`vision_start_id`, `image_token_id`）来判断模态。但这种方式有两个问题：
1. **视频和图像的 pad token 不同**（`<|image_pad|>` vs `<|video_pad|>`），需要两套扫描逻辑
2. **嵌套结构复杂**：视频的每帧都有 `<|vision_start|>...<|vision_end|>`，扫描逻辑容易出错

`mm_token_type_ids` 将模态判断**前移到 Processor 阶段**，让 `get_rope_index` 只需要按连续段的类型分组，逻辑大幅简化。

---

## 关联概念

- [[qwen2.5_vl_预处理流水线]] — Qwen2.5-VL Processor 完整流水线 🔄 演化自
- [[qwen2.5_vl_预处理框架集成与显存评估]] — 框架集成与 OOM 评估 ✅ 支持
- [[qwen3.5_多模态融合机制]] — mm_token_type_ids 的下游消费者 ✅ 支持
- [[qwen3.5_interleaved_mrope]] — 时间戳如何影响 MRoPE 的 T 维度 ✅ 支持
- [[qwen3.5_前向传播全链路]] — 全链路流转 ✅ 支持

## 参考来源

- `transformers/src/transformers/models/qwen3_vl/processing_qwen3_vl.py` (272行)
