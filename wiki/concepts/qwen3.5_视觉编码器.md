---
title: Qwen3.5 视觉编码器
tags: [Qwen3.5, VisionModel, ViT, PatchEmbed, PatchMerger, 视觉编码]
created: 2026-05-10
updated: 2026-05-10
sources: 2
status: active
---

# Qwen3.5 视觉编码器 (Qwen3_5VisionModel)

## 模块整体说明与架构拆解

Qwen3.5 的视觉编码器继承自 Qwen3-VL 的 `Qwen3VLVisionModel`，核心改动是**删除了 DeepStack**（不再从 ViT 中间层提取特征注入 LLM）。整体是一个标准的 ViT 管线：Conv3d 切 patch → 加位置嵌入 → N 层 Transformer Block → PatchMerger 空间降维。

### 推演样本

沿用 [[qwen3.5_前向传播全链路]] 的统一样本：1 张 448×448 图 + 1 段 4 帧（392×644）视频。

### 全局代码调用顺序与流转概览

```
Qwen3_5VisionModel.forward(pixel_values, grid_thw)
  文件: models/qwen3_5/modular_qwen3_5.py:432
  输入: pixel_values (3600, 3, 2, 14, 14), grid_thw=[[1,32,32], [2,28,46]]
  │
  ├─ 1. patch_embed(pixel_values)
  │     Conv3d(3, 1280, kernel=(2,14,14), stride=(2,14,14))
  │     输入: (3600, 3, 2, 14, 14) → reshape(-1, 3, 2, 14, 14)
  │     输出: (3600, 1280)
  │
  ├─ 2. fast_pos_embed_interpolate(grid_thw)
  │     双线性插值绝对位置嵌入
  │     输出: pos_embeds (3600, 1280)
  │     hidden_states = hidden_states + pos_embeds  → (3600, 1280)
  │
  ├─ 3. rot_pos_emb(grid_thw) → 2D RoPE
  │     输出: rotary_pos_emb (3600, 40)
  │     → emb = cat(rotary, rotary) → (3600, 80)
  │     → position_embeddings = (emb.cos(), emb.sin())
  │
  ├─ 4. cu_seqlens 计算
  │     = [0, 1024, 3600]  # 1张图+1段视频作为独立的attention段
  │
  ├─ 5. for blk in self.blocks:  # 32层
  │     blk(hidden_states, cu_seqlens, position_embeddings)
  │     每层: LN→Attention→residual→LN→MLP→residual
  │     维度不变: (3600, 1280) → (3600, 1280)
  │
  └─ 6. merger(hidden_states)  → PatchMerger
        2×2合并: 4个相邻token拼接
        Norm → Linear(5120,5120) → GELU → Linear(5120,4096)
        输出: (900, 4096)
```

---

## 子模块详解

### 1. Conv3d PatchEmbed

#### 模块说明

将像素立方体切分为 patch embedding。使用 3D 卷积一步完成空间+时间维度的切分和投影。

#### 逻辑链输入与输出
- **输入**：`pixel_values` (3600, 3, 2, 14, 14) — 3600 个 patch group (图1024+视频2576)
- **输出**：`hidden_states` (3600, 1280) — 每个 patch group 的特征向量

#### 核心源码解剖

```python
# 文件: models/qwen3_vl/modeling_qwen3_vl.py:72-89
class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config):
        self.patch_size = 14            # 空间 patch 大小
        self.temporal_patch_size = 2    # 时间 patch 大小
        self.in_channels = 3            # RGB
        self.embed_dim = 1280           # 输出维度 = vision hidden_size

        kernel_size = [2, 14, 14]       # (T, H, W)
        self.proj = nn.Conv3d(
            3, 1280,
            kernel_size=kernel_size,
            stride=kernel_size,   # stride=kernel → 无重叠切分
            bias=True
        )

    def forward(self, hidden_states):
        # hidden_states 进来时是扁平的 patch groups
        hidden_states = hidden_states.view(-1, 3, 2, 14, 14)
        # shape: (num_patches, C=3, T=2, H=14, W=14)

        hidden_states = self.proj(hidden_states)
        # Conv3d 输出: (num_patches, 1280, 1, 1, 1)
        # .view(-1, 1280) → (1024, 1280)
        return hidden_states
```

**推演**：448×448 图，T=1，被补齐为 2 帧，H=32, W=32，共 1024 个 patch group。4帧 392×644 视频，T=4/2=2步，H=28, W=46，共 2×28×46=2576 个 group。总计 3600 个。Conv3d 将每个 (3,2,14,14)=1176 个像素值投影到 1280 维。

### 2. 绝对位置嵌入 (fast_pos_embed_interpolate)

#### 模块说明

使用**可学习的绝对位置嵌入**（`nn.Embedding`）+ **双线性插值**适配任意分辨率。这与标准 ViT 的固定位置嵌入不同 — Qwen3.5 支持动态分辨率。

#### 核心源码解剖

```python
# 文件: models/qwen3_vl/modeling_qwen3_vl.py:702-763
def fast_pos_embed_interpolate(self, grid_thw):
    # self.pos_embed: nn.Embedding(num_position_embeddings, 1280)
    # num_position_embeddings = 2304 = 48×48（预训练分辨率对应的网格）
    # self.num_grid_per_side = 48

    # 对每张图的 (H, W) 网格，在 48×48 的预训练网格上做双线性插值
    h_idxs = torch.linspace(0, 47, h)  # h=32 → 32个采样点
    w_idxs = torch.linspace(0, 47, w)  # w=32

    # 四角插值: floor/ceil × floor/ceil → 4个权重
    # 最终: patch_pos_embeds (H×W, 1280) 对每张图

    # 还需按 spatial_merge_size 重排顺序（因为后续PatchMerger是2×2合并）
    # view(T, H//2, 2, W//2, 2, -1).permute(0,1,3,2,4,5).flatten(0,4)
    # 这让相邻2×2的patch排在一起，方便PatchMerger直接reshape合并

    return patch_pos_embeds  # (1024, 1280)
```

**为什么需要 merge_size 重排？** PatchMerger 将相邻 2×2=4 个 token 拼接，所以位置嵌入的排列顺序必须保证这 4 个 token 在内存中是连续的。`permute(0,1,3,2,4,5)` 将 (T, H_block, W_block, merge_h, merge_w, dim) 变为 (T, H_block, W_block, merge_h×merge_w, dim)，使得每个 block 内的 4 个 patch 连续排列。

### 3. 2D RoPE (rot_pos_emb)

#### 模块说明

为 ViT 内部的 Attention 提供相对位置信息。将 head_dim 分为两半，分别编码行坐标和列坐标。

#### 核心源码解剖

```python
# 文件: models/qwen3_vl/modeling_qwen3_vl.py:662-700
def rot_pos_emb(self, grid_thw):
    merge_size = self.spatial_merge_size  # 2
    # 对每张图:
    merged_h, merged_w = 32//2, 32//2  # = 16, 16
    # 生成 block 坐标 + intra-block 偏移
    row_idx = block_rows * 2 + intra_row  # 还原到全分辨率坐标
    col_idx = block_cols * 2 + intra_col

    coords = stack([row_idx, col_idx])  # (1024, 2)
    # 查表: freq_table = self.rotary_pos_emb(max_hw=32)
    # rotary_pos_emb 是 Qwen3VLVisionRotaryEmbedding(dim=40)
    # dim = head_dim // 2 = (1280/16) // 2 = 40

    embeddings = freq_table[pos_ids]  # (1024, 2, 40)
    embeddings = embeddings.flatten(1)  # (1024, 80)
    # 但实际 rot_pos_emb 返回 (1024, 40)，后续 cat 为 (1024, 80)
```

### 4. VisionBlock (×32)

#### 模块说明

标准的 Pre-Norm Transformer Block：`LayerNorm → Attention → residual → LayerNorm → MLP → residual`。

```python
# 文件: models/qwen3_vl/modeling_qwen3_vl.py:265-296
class Qwen3VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config):
        self.norm1 = nn.LayerNorm(1280, eps=1e-6)
        self.norm2 = nn.LayerNorm(1280, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config)  # 全attention, 无window
        self.mlp = Qwen3VLVisionMLP(config)          # Linear→act→Linear

    def forward(self, hidden_states, cu_seqlens, position_embeddings, **kwargs):
        # Pre-Norm Attention + residual
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens, position_embeddings, **kwargs
        )
        # Pre-Norm MLP + residual
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
        # 维度不变: (1024, 1280) → (1024, 1280)
```

**VisionAttention 关键点**：
- 使用 `cu_seqlens` 做变长 attention（多图可以打包在一个 batch 中）
- 应用 2D RoPE（行/列分别旋转）
- **不是** causal attention（`is_causal=False`）
- QKV 融合投影：`nn.Linear(1280, 1280×3)`

**VisionMLP**：
- `Linear(1280, intermediate_size) → act → Linear(intermediate_size, 1280)`
- 注意这里用的是 `hidden_act`（如 QuickGELU），不是 SwiGLU

### 5. PatchMerger

#### 模块说明

将 ViT 输出的 1024 个 token 通过 2×2 空间合并降至 256 个，同时投影到 LLM 的维度。

```python
# 文件: models/qwen3_vl/modeling_qwen3_vl.py:108-121
class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config):
        merge_size = config.spatial_merge_size  # 2
        self.hidden_size = 1280 * (2**2)  # = 5120
        self.norm = nn.LayerNorm(1280, eps=1e-6)  # use_postshuffle=False时
        self.linear_fc1 = nn.Linear(5120, 5120)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(5120, 4096)  # out_hidden_size = LLM hidden_size

    def forward(self, x):
        # x: (3600, 1280)
        x = self.norm(x)  # 对1280维做LayerNorm
        x = x.view(-1, 5120)  # 每4个相邻token拼接: (900, 5120)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        # (900, 5120) → (900, 5120) → (900, 4096)
        return x
```

**推演**：3600 tokens ÷ 4 = 900 merged tokens (图像256 + 视频644)，每个 merged token 由 4×1280=5120 维 → FC1 → GELU → FC2 → 4096 维。

### 6. Qwen3.5 与 Qwen3-VL VisionModel 的差异

```python
# 文件: models/qwen3_5/modular_qwen3_5.py:421-479
class Qwen3_5VisionModel(Qwen3VLVisionModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        del self.deepstack_visual_indexes   # ← 删除 DeepStack
        del self.deepstack_merger_list      # ← 删除 DeepStack merger

    def forward(self, hidden_states, grid_thw, **kwargs):
        # 与 Qwen3VL 流程相同，但：
        # 1. 不收集 deepstack_feature_lists
        # 2. 返回 BaseModelOutputWithPooling（不含 deepstack_features）
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )
```

---

## 关联概念

- [[qwen3.5_前向传播全链路]] — 全链路流转 ✅ 支持
- [[conv3d_时空切块器]] — Conv3d PatchEmbed 基础 🔄 演化自
- [[patchmerger_空间降维]] — PatchMerger 基础 🔄 演化自
- [[2d_rope_视觉位置编码]] — 2D RoPE 基础 🔄 演化自
- [[window_attention_交错注意力]] — Qwen2.5-VL 用 window attention，Qwen3.5 改为全局 ✅ 支持

## 参考来源

- `transformers/src/transformers/models/qwen3_5/modular_qwen3_5.py:421-479`
- `transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py:59-823`
