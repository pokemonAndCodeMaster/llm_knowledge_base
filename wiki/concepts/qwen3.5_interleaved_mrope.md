---
title: Qwen3.5 Interleaved MRoPE 位置编码
tags: [Qwen3.5, MRoPE, 位置编码, RoPE, 交错式, 3D]
created: 2026-05-10
updated: 2026-05-10
sources: 2
status: active
---

# Qwen3.5 Interleaved MRoPE 位置编码

## 模块整体说明与架构拆解

Qwen3.5 的 LLM Decoder 使用 **Interleaved MRoPE**（交错式多模态旋转位置编码）。与 Qwen2.5-VL 的 chunked MRoPE 不同，Qwen3.5 将 T/H/W 三个维度的频率**交错排列**。

### 推演样本

沿用 [[qwen3.5_前向传播全链路]] 的统一样本。

### 配置差异

| 参数 | Qwen2.5-VL/Qwen3-VL | Qwen3.5 |
|------|---------------------|---------|
| `mrope_section` | `[24, 20, 20]` | `[11, 11, 10]` |
| `partial_rotary_factor` | 1.0 | 可配置 |
| 排列 | chunked `[TTT...HHH...WWW]` | interleaved `[THWTHWTHW...]` |

### 全局代码调用顺序

```
Qwen3_5TextModel.forward()
  │
  ├─ position_ids: (4, 1, 268)
  │   [0] = text 位置（用于 causal mask）
  │   [1:] = (T, H, W) 三维位置（用于 MRoPE）
  │
  ├─ text_position_ids = position_ids[0]  # (1, 268)
  │  position_ids = position_ids[1:]       # (3, 1, 268)
  │
  └─ position_embeddings = self.rotary_emb(hidden_states, position_ids)
      │ → Qwen3_5TextRotaryEmbedding.forward()
      │
      ├─ inv_freq: (dim//2,) 基础频率
      │  inv_freq_expanded: (3, 1, dim//2, 1)  # 三维各自的频率
      │
      ├─ freqs = inv_freq @ position_ids  # (3, 1, dim//2, 268)
      │  → transpose → (3, 1, 268, dim//2)
      │
      ├─ freqs = apply_interleaved_mrope(freqs, [11, 11, 10])
      │  → (1, 268, dim//2)  交错后的单一频率
      │
      ├─ emb = cat(freqs, freqs) → (1, 268, dim)
      └─ return (emb.cos() × scaling, emb.sin() × scaling)
```

---

## 子模块详解

### 1. apply_interleaved_mrope — 核心算法

#### 源码解剖

```python
# 文件: models/qwen3_vl/modeling_qwen3_vl.py:370-385
def apply_interleaved_mrope(self, freqs, mrope_section):
    """
    freqs: (3, bs, seq_len, head_dim//2)  # T=0, H=1, W=2
    mrope_section: [11, 11, 10]  # 各维度占用的频率槽数
    """
    freqs_t = freqs[0]  # 以 T 维为基底 (bs, seq_len, head_dim//2)

    for dim, offset in enumerate((1, 2), start=1):  # H=1, W=2
        length = mrope_section[dim] * 3  # 11*3=33
        idx = slice(offset, length, 3)   # H: [1,4,7,10,...,31], W: [2,5,8,11,...,32]
        freqs_t[..., idx] = freqs[dim, ..., idx]

    return freqs_t  # (bs, seq_len, head_dim//2)
```

#### 推演（以 head_dim=128, head_dim//2=64 为例）

假设 `mrope_section = [11, 11, 10]`（总共 11+11+10=32 个交错组）。

频率索引分配（前 33 个位置，`length=11×3=33`）：
```
索引:  0  1  2  3  4  5  6  7  8  ... 30 31 32
维度:  T  H  W  T  H  W  T  H  W  ...  T  H  W
```

索引 33-63（剩余位置，`length=10×3=30` 只覆盖到 32）：
```
索引: 33 34 35 ... 62 63
维度:  T  T  T  ...  T  T  （未被 H/W 覆盖，保持 T 的频率）
```

**物理意义**：低频维度（索引靠前）被 T/H/W 交错覆盖，高频维度（索引靠后）仅用 T。这使得每个维度都能获得低频和高频的旋转信号，而不是像 chunked 方式那样某些维度只分配到高频。

### 2. Qwen3_5TextRotaryEmbedding

```python
# 文件: models/qwen3_5/modular_qwen3_5.py:166-187
class Qwen3_5TextRotaryEmbedding(Qwen3VLTextRotaryEmbedding):
    def __init__(self, config):
        super().__init__()
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = config.hidden_size // config.num_attention_heads  # 128
        dim = int(head_dim * partial_rotary_factor)
        # partial_rotary_factor < 1.0 → 只对部分维度做 RoPE（其余维度无位置信息）

        inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))
        # shape: (dim//2,) 例如 (64,)
        return inv_freq, 1.0
```

### 3. 4D position_ids 的含义

| 维度 | 含义 | 文本 token | 图像 token |
|------|------|-----------|-----------|
| `[0]` | text 位置 | 标准递增 | 标准递增 |
| `[1]` | Temporal (T) | 同 text | 由 grid_thw 计算 |
| `[2]` | Height (H) | 同 text | 行号 |
| `[3]` | Width (W) | 同 text | 列号 |

纯文本时 T=H=W=text_pos，三维退化为一维。

---

## 关联概念

- [[mrope_多模态位置编码]] — MRoPE 基础原理 🔄 演化自
- [[2d_rope_视觉位置编码]] — ViT 内 2D RoPE ✅ 支持
- [[qwen3.5_前向传播全链路]] — 全链路流转 ✅ 支持
- [[qwen3.5_多模态融合机制]] — position_ids 的计算来源 ✅ 支持

## 参考来源

- `transformers/src/transformers/models/qwen3_5/modular_qwen3_5.py:166-187`
- `transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py:299-385`
