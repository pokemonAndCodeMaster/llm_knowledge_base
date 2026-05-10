---
title: Qwen3.5 多模态融合机制
tags: [Qwen3.5, 多模态融合, masked_scatter, 占位符替换, MRoPE, position_ids]
created: 2026-05-10
updated: 2026-05-10
sources: 2
status: active
---

# Qwen3.5 多模态融合机制

## 模块整体说明与架构拆解

Qwen3.5 的多模态融合在 `Qwen3_5Model.forward()` 中完成，包含两个核心步骤：
1. **占位符替换**：用视觉 embedding 替换 input_ids 中的 `<image>` 占位 token
2. **3D 位置计算**：为文本/图像/视频 token 分别计算 MRoPE 的 (T,H,W) 位置

### 推演样本

沿用 [[qwen3.5_前向传播全链路]] 的统一样本：1 张 448×448 图 + 1 段 4 帧（392×644）视频 + "描述这张图片和视频"，seq_len=921。

### 全局代码调用顺序

```
Qwen3_5Model.forward(input_ids, pixel_values, image_grid_thw, mm_token_type_ids, ...)
  文件: models/qwen3_5/modular_qwen3_5.py:588
  │
  ├─ 1. inputs_embeds = embed_tokens(input_ids)
  │     (1, 921) → (1, 921, 4096)
  │
  ├─ 2. image_outputs = get_image_features(pixel_values, image_grid_thw)
  │     → visual(pixel_values, grid_thw) → [[qwen3.5_视觉编码器]]
  │     → image_embeds: [tensor(256, 4096), tensor(644, 4096)]  按图/视频拆分
  │     → cat → (900, 4096)
  │
  ├─ 3. image_mask, _ = get_placeholder_mask(input_ids, inputs_embeds, image_embeds)
  │     找到 input_ids 中 == 248056 和 248057 的 900 个位置
  │     image_mask: (1, 921, 4096) bool
  │
  ├─ 4. inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
  │     900 个占位位置被替换为视觉特征
  │
  ├─ 5. position_ids = compute_3d_position_ids(...)
  │     └─ get_rope_index(input_ids, mm_token_type_ids, image_grid_thw)
  │        输出: position_ids (4, 1, 921)
  │
  └─ 6. self.language_model(inputs_embeds, position_ids, ...)
```

---

## 子模块详解

### 1. get_placeholder_mask — 占位符定位

```python
# 文件: models/qwen3_vl/modeling_qwen3_vl.py:1168-1207
def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
    # input_ids: (1, 921), 包含图像和视频占位符
    special_image_mask = (input_ids == 248056) | (input_ids == 248057)
    # shape: (1, 921), dtype: bool, 有 900 个 True

    n_image_tokens = special_image_mask.sum()  # = 900

    # 扩展为 3D: (1, 921) → unsqueeze(-1) → (1, 921, 1) → expand → (1, 921, 4096)
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)

    # 验证数量一致
    assert inputs_embeds[special_image_mask].numel() == image_features.numel()
    # 900 × 4096 == 900 × 4096 ✓


    return special_image_mask, special_video_mask
```

### 2. masked_scatter — 原位替换

```python
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

`masked_scatter` 语义：将 `image_embeds` 的值**按顺序**填入 `inputs_embeds` 中 mask 为 True 的位置。

**推演**：假设 input_ids = `[BOS, <vs>, <img>×256, <ve>, <vs>, <video>×644, <ve>, 描, 述, ...]`
- 位置 4-259（256 个 `<img>`）和 263-906（644 个 `<video>`）的 embedding 原本是随机的占位值
- `masked_scatter` 后这 900 个位置被替换为 ViT 编码后的视觉特征向量
- 其他位置（文本 token 和 `vision_start/end`）的 embedding 不变

### 3. compute_3d_position_ids — MRoPE 位置计算

#### 核心源码解剖

```python
# 文件: models/qwen3_vl/modeling_qwen3_vl.py:1209-1256
def compute_3d_position_ids(self, input_ids, inputs_embeds, image_grid_thw, ...):
    # 首次 forward（有多模态输入时）:
    position_ids, rope_deltas = self.get_rope_index(
        input_ids, image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids, attention_mask=attention_mask
    )
    self.rope_deltas = rope_deltas  # 缓存用于后续自回归

    return position_ids  # (3, 1, 921)
    # → 拼上 text 维后变成 (4, 1, 921)
```

### 4. get_rope_index — 逐 token 位置分配

```python
# 文件: models/qwen3_vl/modeling_qwen3_vl.py:1033-1124
def get_rope_index(self, input_ids, mm_token_type_ids, image_grid_thw, ...):
    spatial_merge_size = 2  # 来自 vision config

    # 对每个 batch:
    for batch_idx, current_input_ids in enumerate(input_ids):
        # mm_token_type_ids 标记每个 token 的模态: 0=文本, 1=图像, 2=视频
        # 按模态分组: [(0, 0, 2), (1, 2, 258), (0, 258, 268)]
        #              文本段      图像段(256个)    文本段

        for modality_type, start_idx, end_idx in groups:
            if modality_type == 0:  # 文本
                text_len = end_idx - start_idx
                # 三个维度都用相同的递增位置
                pos = arange(text_len) + current_pos
                llm_pos_ids = pos.view(1,-1).expand(3,-1)
                # 推演: 前2个文本token → [[0,1],[0,1],[0,1]]
                current_pos += text_len

            else:  # 图像(1) 或 视频(2)
                grid_thw = next(grid_iter)  # [1, 32, 32]
                vision_pos = get_vision_position_ids(
                    current_pos, grid_thw, 1, spatial_merge_size
                )
                # merge后: llm_grid_h = 32//2 = 16, llm_grid_w = 32//2 = 16
                # temporal: 全部 = current_pos (因为T=1)
                # height: [current_pos, current_pos+1, ..., current_pos+15] 每个repeat 16次
                # width:  [current_pos, ..., current_pos+15] repeat 16次

                current_pos += max(32, 32) // 2  # = 16
                # 注意: 视觉token推进的位置 = max(H,W)//merge_size
                # 这使得位置空间被压缩了

    # 最终拼接并返回
    # position_ids: (3, 1, 268)
    # mrope_position_deltas: 位置偏移量，用于自回归续接
```

#### 推演（具体数值）

假设输入序列（简化）：`[BOS(文本), <vs>(文本), <img>×256(图像), <ve>(文本), <vs>(文本), <video>×644(视频), <ve>(文本), 描(文本), 述(文本)]`

mm_token_type_ids = `[0, 0, 1×256, 0, 0, 2×644, 0, 0, 0]`

分组与位置分配：
1. 文本段 (pos 0-3): `current_pos=0`, T=H=W=`[0,1,2,3]`, `current_pos→4`
2. 图像段 (256 tokens): grid_thw=[1,32,32], merge=2 → 16×16=256 tokens
   - T 维: 全部 = 4（当前时序位）
   - H 维: `[4,4,...,4, 5,5,...,5, ..., 19,19,...,19]`（16行，每行16列）
   - W 维: `[4,5,...,19, 4,5,...,19, ..., 4,5,...,19]`（16列，重复16行）
   - `current_pos += max(32,32)//2 = 16` → `current_pos=20`
3. 文本段 (pos 260-262): T=H=W=`[20,21,22]`, `current_pos→23`
4. 视频段 (644 tokens): grid_thw=[2,28,46], merge=2 → 2步×14×23=644 tokens
   - T 维: 第1帧步 `[23]×322`，第2帧步 `[24]×322`
   - H/W 维: 类似图像展开，但 H从23到36，W从23到45。
   - `current_pos += 2*(max(28,46)//2) = 46` → `current_pos=69`
5. 文本段 (pos 907-末尾): 从 69 继续递增。

最终 position_ids shape: `(3, 1, 921)`

### 5. Qwen3.5 删除 DeepStack

```python
# Qwen3.5 VisionModel init:
del self.deepstack_visual_indexes
del self.deepstack_merger_list

# Qwen3.5 Model forward 中:
# 不传递 visual_pos_masks 和 deepstack_visual_embeds 给 language_model
# → LLM 不会在中间层注入视觉特征
```

**Qwen3-VL 的 DeepStack**：从 ViT 的指定层（如第 8/16/24 层）提取特征，经过独立的 PatchMerger，注入到 LLM 的前几个 Decoder 层。Qwen3.5 认为这不再必要（可能因为混合 Decoder 本身已经足够强）。

---

## 关联概念

- [[qwen3.5_前向传播全链路]] — 全链路流转 ✅ 支持
- [[qwen3.5_interleaved_mrope]] — 融合后的位置编码详解 ✅ 支持
- [[qwen3.5_视觉编码器]] — 生成视觉 embedding 的上游 ✅ 支持
- [[mrope_多模态位置编码]] — MRoPE 基础 🔄 演化自
- [[patchmerger_空间降维]] — PatchMerger ✅ 支持

## 参考来源

- `transformers/src/transformers/models/qwen3_5/modular_qwen3_5.py:559-659`
- `transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1033-1257`
