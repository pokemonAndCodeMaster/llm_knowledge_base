---
title: ViT 视觉骨干网核心原理与结构
tags: [概念, ViT, 视觉编码器, Transformer, Qwen2.5-VL, 架构统一, Tensor 流转]
created: 2026-05-03
updated: 2026-05-04
sources: 5
status: active
---

# ViT 视觉骨干网核心原理与结构

## 模块整体说明与架构拆解

视觉骨干网（Vision Backbone）是多模态大模型的“语义核心熔炉”。以 Google 于 2020 年提出的原始 Vision Transformer (ViT) 为开端，它首次证明了：**不需要卷积神经网络 (CNN) 的强空间归纳偏置，纯 Transformer 也能在图像领域提取出极具表达力的全局语义特征。** 

在 Qwen-VL 系列中，ViT 被深度改造并“LLM 化”，其职责是接收预处理后的视觉序列，通过 **32 层** 高度统一的 Transformer Block 深度堆叠，完成从“像素光影”到“高级语义（物体、动作、场景）”的跨越，为最终进入大语言模型（LLM）做好准备。

### 渊源与演化：原始 ViT (2020) 的奠基流转
![](<images/ViT(2020, Google Research)-image.png>)
![](<images/ViT(2020, Google Research)-vit流程图.png>)

原始 ViT 的架构完全复用了 NLP 领域的标准 Transformer Encoder：
1. **Linear Projection (Patch Embedding)**：将二维图片切分成无重叠的 Patch，展平并通过全连接层映射为 1D Token 序列。
2. **[CLS] Token 与 Positional Embedding**：在序列头部追加一个用于分类的 Token，并为所有 Token 加上 1D 可学习绝对位置编码。
3. **Transformer Encoder**：由交替的 `LayerNorm -> Multi-Head Attention (MSA)` 和 `LayerNorm -> MLP (Inverted Bottleneck)` 组成，循环 $L$ 层。
4. **MLP Head**：仅提取 `[CLS]` Token 的特征，用于图像分类。

### 架构演化对比：Qwen2.5-VL 的“LLM 化”颠覆
为了让视觉特征与语言大模型能够更顺畅地对接（降低跨模态对齐的“方言差”），Qwen 团队对 ViT 进行了彻底的改造：

| 组件 | 原始 ViT (2020) | Qwen2.5-VL ViT | 演化本质与原理解读 |
| :--- | :--- | :--- | :--- |
| **位置编码** | 1D 可学习绝对位置，**直接加 (Add)** 在输入上。数学本质是 $W(I+P)=WI+WP$，相加等价于一种特殊的高效 Concatenate | **2D-RoPE**，在 Attention 内部进行复数旋转 | 绝对位置编码锁死了分辨率，2D-RoPE 天然支持任意分辨率和长宽比外推。 |
| **全局聚合** | 依赖序列开头的 `[CLS]` Token | **抛弃 `[CLS]` Token**，全部 Token 降维后送入 LLM | MLLM 不需要做简单的图像分类，而是需要密集的局部和全局语义输入给 LLM。 |
| **归一化策略** | LayerNorm | **RMSNorm** | 抛弃均值平移，与 Qwen2.5 文本端完全对齐，提升计算速度。 |
| **非线性激活** | GELU | **SwiGLU (且 bias=True)** | 文本端使用无偏置 SwiGLU，视觉端为了对抗传感器直流偏移底噪开启偏置。 |
| **注意力范围** | 每层都是全图 Global Attention | **Window Attention** 为主，1/8 的层穿插 Global | 在原生极高分辨率下，$O(N^2)$ 全图计算会 OOM，交错窗口既省算力又保住了宏观意图。 |

---

## 核心底层神经元级别的微观放大镜 (Tensor Flow)

为了破除“黑盒”迷信，我们以原始 ViT-Base 处理一张 $224 \times 224 \times 3$ 的图像为例，用显微镜观察数据在每一层神经元中的张量形变（Tensor Shape）。

### 1. 物理切块与 Patch Embedding
*   **输入**：Image $X \in \mathbb{R}^{B \times 3 \times 224 \times 224}$
*   **动作**：使用 `kernel_size=16`, `stride=16` 的 2D 卷积核，将图像切分为 $14 \times 14 = 196$ 个 Patch。每个 Patch 区域（$16 \times 16 \times 3 = 768$ 个像素值）被映射为一个 768 维的向量。
*   **张量形变**：`[B, 3, 224, 224] -> Conv2d -> [B, 768, 14, 14] -> flatten -> [B, 768, 196] -> transpose -> [B, 196, 768]`

### 2. [CLS] 拼接与位置编码融合
*   **[CLS] Token**：这是一个 `nn.Parameter(torch.randn(1, 1, 768))`，拓展到 Batch 后与 Patch 序列 `cat`，长度变为 197。`[B, 197, 768]`
*   **位置编码**：生成一个形状为 `[1, 197, 768]` 的可学习参数。直接与输入相加 `x = x + pos_embed`。
*   **第一性原理**：为什么要相加而不是 Concat？因为 $W(I+P) = WI+WP$，相加等效于在特征拼接后再做一次联合线性变换，既保留了空间信息，又极其节约显存。

### 3. 多头注意力机制 (Multi-Head Attention, MSA) 的微观解剖
这是 Transformer 的跨空间融合核心。
*   **输入**：`X`，形状 `[B, 197, 768]`。
*   **QKV 线性映射**：通过一个 `nn.Linear(768, 768*3)`，将输入扩展为 `[B, 197, 2304]`。
*   **多头拆分**：将 2304 劈开为 Q, K, V。假设 Head=12，则每个 Head 的维度 `head_dim = 768/12 = 64`。
    `[B, 197, 3, 12, 64] -> permute -> Q, K, V 分别为 [B, 12, 197, 64]`
*   **注意力分数矩阵 (Attention Matrix)**：
    计算 $Q \cdot K^T$：`[B, 12, 197, 64] @ [B, 12, 64, 197] -> [B, 12, 197, 197]`
    这个 $197 \times 197$ 的矩阵就是全局注意力热力图。除以 $\sqrt{64}$ (Scale) 防止 Softmax 梯度消失，然后算 Softmax，得出权重。
*   **特征加权 (Weighted Sum)**：
    `[B, 12, 197, 197] @ [B, 12, 197, 64] -> [B, 12, 197, 64]`。随后把 12 个头拼回去，变回 `[B, 197, 768]`。
*   **物理意义**：经过这一步，原本只代表“猫耳朵”的 Patch 向量内部，按照 Softmax 的权重，被狠狠地“混入”了“猫眼睛”和“猫尾巴”的数值。孤立的像素碎片演化成了连贯的“猫”。

### 4. MLP 层：倒瓶颈结构 (Inverted Bottleneck) 的通道提纯
如果说 Attention 是让所有 Token “串门互通”，那么 MLP 就是让每个 Token “关起门来内部消化”。
*   **网络结构**：两层全连接层 `Linear(768, 3072) -> GELU -> Linear(3072, 768)`。
*   **第一性原理 (倒瓶颈)**：不同于 ResNet 先降维再升维的 Bottleneck（为了省算力）。ViT 采用**先将通道数放大 4 倍，再缩减回来**的 Inverted Bottleneck 结构。这是因为在高维空间中，特征之间更容易被非线性激活函数（如 GELU/SwiGLU）解耦和过滤，从而极大地增强模型对复杂视觉概念的表达能力。

### 5. 残差连接 (Residual) 与高速公路
每一层 Attention 和 MLP 的外面都包着一个残差 `x = x + module(norm(x))`。
*   **物理意义**：不仅防止了前向传播时原始物理像素特征在深层网络中丢失，更在**反向传播**中充当了“梯度高速分配器”，将 Loss 无损地传导回第一层的切块器，根绝梯度消失。

---

## 核心源码解剖 (从原始 ViT 到 Qwen)

要真正破除黑盒，没有什么比直接看 PyTorch 源码更直观的了。以下是基础组件的标准代码实现。

### 1. 原始 ViT 的 Patch Embedding
将卷积切图和张量重塑（展平、转置）一气呵成：
```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        # 利用 2D 卷积的无重叠滑动，直接切出 Patch 并映射到 768 维
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)      # [B, 3, 224, 224] -> [B, 768, 14, 14]
        x = x.flatten(2)      # [B, 768, 14, 14] -> [B, 768, 196]
        x = x.transpose(1, 2) # [B, 768, 196]    -> [B, 196, 768] (这就是标准的 Transformer 序列格式)
        return x
```

### 2. 原始 ViT 的多头注意力 (MultiheadAttention)
重点观察 `qkv` 的剥离和最后合并多头的形变：
```python
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        # 一次性生成 Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape  # 例如 B, 197, 768
        
        # 1. 映射并拆分为 3 份 (Q,K,V) 和多头 (heads)
        # [B, 197, 3, 12, 64] -> permute -> [3, B, 12, 197, 64]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 [B, 12, 197, 64]
        
        # 2. Q * K^T 计算热力图
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, 12, 197, 197]
        attn = attn.softmax(dim=-1)
        
        # 3. 乘以 Value 并合并多头
        # [B, 12, 197, 64] -> transpose -> [B, 197, 12, 64] -> reshape -> [B, 197, 768]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # 4. 最终投影
        return self.out_linear(x)
```

### 3. Qwen2.5-VL 的视觉块 (Vision Block)
Qwen 的架构中引入了动态窗口切换和 RMSNorm：
**文件路径**：`transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
```python
class Qwen2_5_VLVisionBlock(nn.Module):
    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb):
        # 1. 第一层归一化 + 注意力 (Residual Path 1)
        # 注意：cu_seqlens 决定了视野是 Window 还是 Global；rotary_pos_emb 注入 2D-RoPE
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens, rotary_pos_emb
        )
        # 2. 第二层归一化 + MLP (Residual Path 2)
        # 注意：视觉侧 MLP 开启了 bias=True 对抗底噪
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
```

---

## 参数量与显存极限推演 (Memory & Parameters)

大模型时代，不仅要懂架构，更要能精准预估算力边界。
以经典的 ViT-Base 为例（$D=768$, 12层, 12头），其参数量约为 **86M**。
*   **参数分布大头**：主要集中在每一层的 Attention 投影矩阵（$768 \times 768 \times 4 \times 12$）和 FFN 升降维矩阵（$768 \times 3072 \times 2 \times 12$）中。

**显存开销公式推演**：
假设模型总参数量为 $M$ (十亿, Billion)。
1.  **训练阶段 (Training)**：
    *   显存不仅要装载**模型权重**，还要装载**梯度 (Gradients)**、**优化器状态 (Optimizer States, 如 Adam 的动量和方差)**，以及反向传播必需的**激活值 (Activations)**。
    *   **单精度 (FP32)** 训练：消耗约 $20M$ (GB)。
    *   **混合精度 (Mixed Precision: FP16/BF16 + FP32)**：注意！这是一个常见误区。开启混合精度并**不能**节省显存！因为为了保证更新精度，内存中必须同时保留 FP32 的权重副本和 FP16 的前向副本，实际消耗依然在 $16M \sim 20M$ (GB) 甚至更高。
    *   **纯 BF16 训练**：消耗约 $10M$ (GB)。
2.  **推理阶段 (Inference)**：
    *   无需保存梯度、优化器状态和长期的中间激活值。
    *   **半精度 (FP16/BF16)** 推理：显存消耗仅约为 $2M$ (GB)。
    *   **单精度 (FP32)** 推理：显存消耗约为 $4M$ (GB)。
*(注：对于 LLM 还需要额外考虑 KV Cache 的巨大开销，但纯粹的 ViT 骨干网由于是一次性前向提取，不存在自回归过程，所以不产生 KV Cache 累积)*

---

## 质量自我审查与准出标准

1.  **流转张量看懂了吗？**：必须能闭眼描述出 `[B, 197, 768]` 是如何依次在 MSA 内部劈开成多头计算 Attention，再在 MLP 中膨胀 4 倍又收缩回来的。
2.  **融合机制清楚了吗？**：能解释 Attention 是如何通过 $QK^T$ 让一个 Patch 获取全局特征的，而 MLP 只是在单个 Patch 内部混合通道。
3.  **理解反向传播的高速公路了吗？**：知道在 32 层深的网络中，残差连接是如何保证梯度不消失的。

---

## 关联概念

- [[conv3d_时空切块器]]：Qwen 体系下的上游输入，提供原始 5D 时空嵌入。
- [[window_attention_交错注意力]]：核心注意力机制升级，实现算力节省。
- [[swiglu_门控激活函数]]：核心非线性单元，负责特征过滤。
- [[rmsnorm_归一化]]：数值稳定性的基石。
- [[patchmerger_空间降维]]：下游衔接，将提纯后的 32 层特征进行最后的 4 倍压缩。

## 参考来源
- 论文：*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2020)*
- 原始资料：`knowledge_base/raw/ViT(2020, Google Research)/ViT(2020, Google Research).md`
- `knowledge_base/raw/万字长文图解Qwen2.5-VL实现细节_猛猿_2025-06-25/index.md`