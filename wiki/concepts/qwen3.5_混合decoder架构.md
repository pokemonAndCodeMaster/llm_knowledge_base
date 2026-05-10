---
title: Qwen3.5 混合 Decoder 架构
tags: [Qwen3.5, 混合架构, Decoder, Attention, GatedDeltaNet, layer_types]
created: 2026-05-10
updated: 2026-05-10
sources: 2
status: active
---

# Qwen3.5 混合 Decoder 架构

## 模块整体说明与架构拆解

Qwen3.5 的 Text Decoder 采用**混合架构**：通过 `config.layer_types` 列表指定每一层是 **Full Attention** 还是 **Linear Attention (GatedDeltaNet)**。两种层类型共享 MLP 和 LayerNorm，只在 Token Mixer 上不同。

### 推演样本

沿用 [[qwen3.5_前向传播全链路]] 的统一样本：seq_len=268, hidden_size=4096。

### 全局代码调用顺序与流转概览

```
Qwen3_5TextModel.forward(inputs_embeds, position_ids, attention_mask)
  文件: models/qwen3_5/modular_qwen3_5.py:493
  输入: inputs_embeds (1, 268, 4096), position_ids (4, 1, 268)
  │
  ├─ text_position_ids = position_ids[0]   # (1, 268) — 给 causal mask
  │  position_ids = position_ids[1:]        # (3, 1, 268) — 给 MRoPE [T,H,W]
  │
  ├─ causal_mask = create_causal_mask(...)  # (1, 1, 268, 268) 下三角
  │  linear_attn_mask = attention_mask      # (1, 268) 仅 padding mask
  │
  ├─ position_embeddings = self.rotary_emb(hidden_states, position_ids)
  │  # Interleaved MRoPE → (cos, sin) 各 (1, 268, 128)
  │  # 详见 [[qwen3.5_interleaved_mrope]]
  │
  └─ for i in range(64):  # 64层
      │ layer_mask = linear_attn_mask if layer_types[i]=="linear_attention"
      │              else causal_mask
      │
      └─ decoder_layer(hidden_states, position_embeddings, layer_mask, ...)
          │ 文件: models/qwen3_5/modular_qwen3_5.py:357
          │
          ├─ residual = hidden_states                         # (1,268,4096)
          ├─ hidden_states = input_layernorm(hidden_states)   # RMSNorm
          │
          ├─ [full_attention]: self_attn(hidden_states, mask, pos_ids, kv_cache, pos_emb)
          │   输出: (1, 268, 4096)  详见下方 §标准 Attention
          │
          ├─ [linear_attention]: linear_attn(hidden_states, cache, mask)
          │   输出: (1, 268, 4096)  详见 [[qwen3.5_gated_delta_net]]
          │
          ├─ hidden_states = residual + hidden_states         # 第一残差
          ├─ residual = hidden_states
          ├─ hidden_states = post_attention_layernorm(hidden_states)  # RMSNorm
          ├─ hidden_states = MLP(hidden_states)               # SwiGLU
          └─ hidden_states = residual + hidden_states         # 第二残差
```

---

## 子模块详解

### 1. DecoderLayer — 混合层路由

#### 核心源码解剖

```python
# 文件: models/qwen3_5/modular_qwen3_5.py:344-396
class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.layer_type = config.layer_types[layer_idx]  # 读取该层类型

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
            # 注意: 此时没有 self_attn 属性
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
            # 注意: 此时没有 linear_attn 属性

        # MLP 和 LayerNorm 每层都有
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask, position_ids,
                past_key_values, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,      # ← 仅 padding mask
            )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,       # ← 因果 mask
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

**关键设计**：每层只实例化一种注意力模块，节省显存。

### 2. 双 Mask 系统

| Mask 类型 | 适用层 | 形状 | 说明 |
|-----------|--------|------|------|
| `causal_mask` | Full Attention | `(1,1,268,268)` | 标准下三角因果 mask |
| `linear_attn_mask` | GatedDeltaNet | `(1,268)` | 仅 padding mask（无因果性） |

GatedDeltaNet 不需要因果 mask，因为递推天然是因果的。

```python
# 文件: models/qwen3_5/modular_qwen3_5.py:526-539
causal_mask = create_causal_mask(config, inputs_embeds, attention_mask, ...)
linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

for i, decoder_layer in enumerate(self.layers):
    layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" \
                 else causal_mask
```

### 3. 标准 Attention (Qwen3_5Attention) — 完整源码

```python
# Qwen3_5Attention → Qwen3NextAttention → 最终实现在 Qwen3VLTextAttention
# 文件: models/qwen3_vl/modeling_qwen3_vl.py:436-505

class Qwen3VLTextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        self.head_dim = 128                    # 4096 / 32
        self.num_key_value_groups = 32 // 4    # = 8 (GQA: 32 Q heads, 4 KV heads)
        self.scaling = 128 ** -0.5

        self.q_proj = nn.Linear(4096, 32*128, bias=False)   # → 4096
        self.k_proj = nn.Linear(4096, 4*128, bias=False)    # → 512
        self.v_proj = nn.Linear(4096, 4*128, bias=False)    # → 512
        self.o_proj = nn.Linear(32*128, 4096, bias=False)   # 4096 → 4096

        self.q_norm = RMSNorm(128)   # QKNorm: 每个head独立归一化
        self.k_norm = RMSNorm(128)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs):
        # hidden_states: (1, 268, 4096)
        input_shape = hidden_states.shape[:-1]  # (1, 268)

        # 1. 投影 + QKNorm
        query = self.q_norm(self.q_proj(hidden_states).view(1, 268, 32, 128)).transpose(1, 2)
        # q_proj: (1,268,4096) → (1,268,4096) → view → (1,268,32,128)
        # q_norm: per-head RMSNorm on dim=128 → (1,268,32,128)
        # transpose → (1, 32, 268, 128)

        key = self.k_norm(self.k_proj(hidden_states).view(1, 268, 4, 128)).transpose(1, 2)
        # → (1, 4, 268, 128)

        value = self.v_proj(hidden_states).view(1, 268, 4, 128).transpose(1, 2)
        # → (1, 4, 268, 128)  无 norm

        # 2. 应用 RoPE
        cos, sin = position_embeddings  # 各 (1, 268, 128)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # 3. KV Cache 更新
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
            # 自回归时 key/value 包含所有历史 token

        # 4. Attention 计算 (GQA: 4个KV head → repeat到32个)
        attn_output, attn_weights = attention_interface(
            self, query, key, value, attention_mask,
            scaling=self.scaling, **kwargs
        )
        # attention_interface 内部: repeat_kv(key, 8) → (1,32,268,128)
        # attn = softmax(Q @ K^T / sqrt(128) + mask) @ V
        # output: (1, 32, 268, 128) → transpose → (1, 268, 32, 128)

        # 5. 输出投影
        attn_output = attn_output.reshape(1, 268, 4096)
        attn_output = self.o_proj(attn_output)  # → (1, 268, 4096)
        return attn_output, attn_weights
```

**QKNorm 的作用**：在 Q、K 投影后立即做 RMSNorm（每个 head 的 128 维独立归一化），防止 Q、K 向量模长过大导致 attention score 爆炸。这是 Qwen3+ 引入的改进。

### 4. SwiGLU MLP

```python
# 继承自 Qwen3NextMLP
class Qwen3_5MLP(Qwen3NextMLP):
    # gate_proj: Linear(4096, 12288, bias=False)
    # up_proj:   Linear(4096, 12288, bias=False)
    # down_proj: Linear(12288, 4096, bias=False)
    # act_fn = SiLU

    def forward(self, x):  # x: (1, 268, 4096)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # gate: (1,268,12288), up: (1,268,12288)
        # SiLU(gate) × up → (1,268,12288)
        # down → (1,268,4096)
```

### 5. RMSNorm (Zero-Centered)

```python
# 继承自 Qwen3NextRMSNorm → Gemma3RMSNorm
# 特点: weight 初始化为 0, 使用 (1 + weight) × normalized
# 初始时等价于恒等映射
class Qwen3_5RMSNorm(Qwen3NextRMSNorm):
    pass
```

---

## 第一性原理：为什么要混合 Attention 和 Linear Attention？

| 问题 | 纯 Attention | 纯线性注意力 | 混合方案 |
|------|-------------|-------------|---------|
| 复杂度 | $O(N^2)$ | $O(N)$ | 平衡 |
| 精确回忆 | ✅ 完美 | ❌ 有损 | ✅ 关键层用 Attention |
| 超长上下文 | ❌ KV Cache 爆炸 | ✅ 固定大小状态 | ✅ 大部分层用 GDN |

类比：大脑的**短期记忆**（GDN，高效但有损）+ **长期记忆**（Attention，精确但昂贵）。

---

## 关联概念

- [[qwen3.5_前向传播全链路]] — 全链路流转 ✅ 支持
- [[qwen3.5_gated_delta_net]] — GatedDeltaNet 详解 ✅ 支持
- [[qwen3.5_interleaved_mrope]] — MRoPE 位置编码 ✅ 支持
- [[swiglu_门控激活函数]] — MLP ✅ 支持
- [[rmsnorm_归一化]] — LayerNorm ✅ 支持

## 参考来源

- `transformers/src/transformers/models/qwen3_5/modular_qwen3_5.py:344-556`
- `transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py:436-564`
