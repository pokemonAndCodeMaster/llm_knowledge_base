---
title: LLM Backbone 大语言模型基座演进
tags: [概念, LLM, Qwen2.5, Qwen3, Qwen3.5, Attention, MoE, DeltaNet, 底层神经元]
created: 2026-05-04
updated: 2026-05-04
sources: 4
status: active
---

# LLM Backbone 大语言模型基座演进

## 1. 模块整体说明与 Qwen3.5 全局宏观架构

在多模态大模型（VLM）中，视觉编码器（ViT）负责提取特征，而大语言模型基座（LLM Backbone）则是整个系统的“大脑”，专职负责**因果逻辑推理、常识调取和最终的文本预测生成（Autoregressive Decoding）**。

随着 Qwen 系列的演进，由于多模态数据（尤其是长视频和高分辨率图像）使得上下文长度轻松突破 256K 甚至 1M，传统 Transformer 的 $O(N^2)$ 计算复杂度和爆炸的 KV Cache 成为了致命瓶颈。因此，Qwen3 / Qwen3.5 在架构上进行了激进的革命。

### 1.1 全局架构与上下游串联流转 (以 Qwen3.5 文本底座为例)
Qwen3.5 采用的是 **Gated DeltaNet (线性注意力)** + **Gated Attention (全注意力)** 混合，搭配 **MoE (混合专家)** 的极致稀疏架构。

- **上游输入**：经过词表 Embedding 和视觉投影桥接器处理后的隐状态序列。张量形状为 `[Batch, Seq_Len, Hidden_Size]` (例如 Qwen3.5-397B 中是 `[B, S, 4096]`)。
- **总层数与 3:1 混合交错**：总计 48 或 60 层。以 4 层为一个循环周期：前 3 层是 Linear Attention，第 4 层是 Full Attention（比例为 3:1）。这种设计既保留了线性注意力的 $O(N)$ 推理速度，又通过少数全注意力层保住了强大的上下文召回能力。
- **MoE 穿插**：除了第一层等极少数固定层外，所有的 MLP 都被替换为了极度稀疏的 MoE 层。
- **下游输出**：每一层输出依然维持 `[B, S, 4096]`。最终通过 `LM_Head` 映射为词表大小的 Logits 概率分布，进行预测。

### 1.2 架构演进流转图示
![](<images/v2-09508bd96a0a2ea00b093944525e689c_r.jpg>)

---

## 2. 神经元级微观解剖一：Gated DeltaNet (线性注意力)

传统的 Self-Attention 需要计算每一个 Token 与过去所有 Token 的关系。而 Gated DeltaNet 利用一个固定大小的**记忆矩阵 $S$**，将过去的特征压缩存储起来，使得推理时间与历史长度解耦。

### 2.1 架构与数据流转 (Linear Attention 模块)
![](<images/v2-dc4598ae409deefacc4d75008ba1aeef_r.jpg>)

1. **零中心化归一化 (Zero-Centered RMSNorm)**：
   - **公式**：$y = \frac{x}{\text{RMS}(x)} \cdot (1 + w)$，其中参数 $w$ 初始为 0。
   - **物理意义**：保证在训练初期，归一化层表现出完美的恒等映射（Identity Mapping），防止深层网络梯度爆炸。
2. **QKV 与门控映射**：
   - 输入 `[B, S, 4096]`。经过线性层，一次性得到 $Q, K, V$ 以及门控参数 $z$（输出门），$\beta$（写入步长门控），$g$（历史遗忘衰减门控）。
3. **局部因果卷积 (Conv1D)**：
   - 在进入注意力前，对 $Q, K, V$ 进行核大小为 4 的 1D 深度卷积（Depthwise Conv1D），让当前 Token 融合过去 3 个 Token 的局部信息。
4. **核心更新：Gated Delta Rule** (见下文详解)。
5. **门限控制输出**：
   - 最终输出结果需要再乘以 $\text{SiLU}(z)$，让模型决定是否激活这段记忆。

### 2.2 核心机制：Gated Delta Rule 的张量形变
![](<images/v2-22d0c83fab026a69e62090197a037349_r.jpg>)

Delta Rule 是 RNN 与现代 Transformer 的完美结合：
- **状态矩阵 $S$**：在自回归推理时，我们不再保存庞大无限的 KV Cache，而是仅仅保存一个形状为 `[B, Num_Heads, K_Dim, V_Dim]` (如 `[B, 32, 128, 128]`) 的隐状态矩阵。
- **衰减与更新公式**：
  $$ S_t = g_t \cdot S_{t-1} + \beta_t \cdot k_t \cdot (v_t - S_{t-1}^T k_t)^T $$
  这表明：新的状态 $=$ 衰减后的历史状态 $+$ (新输入 $v_t$ $-$ 历史状态对当前 $k_t$ 的预测偏差) $\times$ 步长。
- **输出查询**：
  $$ o_t = S_t^T q_t $$

### 2.3 核心源码解剖 (流式推理的 $O(1)$ 实现)
```python
def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state):
    # 张量形状: Q/K/V 均为 [Batch=B, Heads=32, Seq_Len=1, Dim=128]
    # initial_state 记忆矩阵: [B, 32, 128, 128]
    last_recurrent_state = initial_state
    
    # 获取当前 Token 的特征 (因为 Seq_Len=1)
    q_t = query[:, :, 0] # [B, 32, 128]
    k_t = key[:, :, 0]
    v_t = value[:, :, 0]
    
    # 门控系数形变，适配广播乘法
    g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1) # [B, 32, 1, 1]
    beta_t = beta[:, :, 0].unsqueeze(-1) # [B, 32, 1]
    
    # 1. 记忆衰减
    last_recurrent_state = last_recurrent_state * g_t
    
    # 2. 计算记忆对当前 K 的响应，并与真实 V 比较得到预测误差 Delta
    # kv_mem = S_{t-1} * K_t
    kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2) 
    delta = (v_t - kv_mem) * beta_t  # [B, 32, 128]
    
    # 3. 将 Delta 作为增量，通过外积更新回记忆矩阵中
    # S_t = S_t + K_t ⊗ Delta
    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    
    # 4. Q 查询记忆矩阵得到输出
    core_attn_out = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    
    return core_attn_out, last_recurrent_state
```
*(注：对于 Prefill 并行训练阶段，底层使用了复杂的 Chunkwise 算法，利用块内并行和块间 RNN 来达到并行的目的。)*

---

## 3. 神经元级微观解剖二：Gated Attention (全注意力)

![](<images/v2-038c12b1b566bc1f8f40aabb286f85c3_r.jpg>)

在 3:1 的架构中，每隔 3 层的 Gated DeltaNet 会穿插一层 Gated Attention。这层 Attention 采用标准的 GQA (Grouped-Query Attention)，但在输出端做了一个精妙的修改。

**核心创新：输出门控 (Gate)**
在传统的 Softmax Attention 算完得到 `attn_output` 后，并不是直接丢给 MLP。
模型根据输入计算出了一个门控矩阵 $Z_t$：
```python
# query_states 和 gate 是通过同一个 Linear 映射出来的
query_states, gate = torch.chunk(self.q_proj(hidden_states), 2, dim=-1)

# 在传统 Attention 算出 attn_output 后，强行用 Sigmoid 过滤
attn_output = attn_output * torch.sigmoid(gate)
```
**物理意义**：允许模型动态地决定“我当前这个 Token，到底要不要接受 Attention 融合过来的庞大全局上下文”。如果不重要，模型可以关闭门控，保持自身原始特征。

---

## 4. 神经元级微观解剖三：高度稀疏 MoE (混合专家)

![](<images/v2-81c6630e93744475cbd711c2f3248350_r.jpg>)

大模型如果完全是稠密的（Dense），那么推理时显存和算力开销将是灾难性的。Qwen3.5 397B 采用的是极致稀疏的 MoE，每次推理仅激活约 17B 参数！

### 4.1 架构与张量流转
- **输入**：`hidden_states` `[B, S, 4096]`。
- **Router (路由器)**：通过一个 `Linear(4096, 512)` 算出一个 `[B, S, 512]` 的概率 Logits 矩阵。
- **Top-K 选择**：对 Logits 进行 Softmax 后，使用 `torch.topk` 只挑出概率最大的 10 个专家（总共 512 个），其余 502 个专家的计算全部跳过。
- **Shared Expert (共享专家)**：除了这 10 个路由专家，所有 Token 还必须同时经过 1 个“共享稠密专家”。
- **融合输出**：
  $$ Out = \sum_{i \in Top10} weight_i \times Expert_i(x) + \text{sigmoid}(shared\_gate) \times Shared\_Expert(x) $$

### 4.2 物理意义与归纳偏置
- **512 个稀疏路由专家**：各自学习极度细分的专业知识（如某个专家可能专精 Python 代码，另一个专精法语语法），极大地拉升了模型的参数量（知识库上限），而不增加每次推理的计算量。
- **1 个共享专家**：兜底通用常识（如“猫是动物”），防止知识碎片化，维持模型的基础稳定。

---

## 5. 多模态与解码提效前沿

### 5.1 DeepStack 跨层特征融合 (Qwen3-VL 专属)
传统的 VLM 只把 ViT 最后一层的特征送入 LLM。
Qwen3-VL 采用了 **DeepStack** 机制：提取 ViT 中间层（如第 8, 16, 24 层）的特征，直接跨越网络，强行残差注入到 LLM 的前 K 层中。
**物理意义**：低级的视觉特征（边缘、轮廓）直接送入底层 LLM 参与逻辑重组，极大增强了模型对高分辨率细节的“抠图”和“视觉定位”能力。

### 5.2 MTP (Multi-Token Prediction)
![](<images/v2-4f3666b39af0db1098080132d8481efa_r.jpg>)
如同 DeepSeek-V3，Qwen3.5 在主模型（如前 60 层）之上，额外增加了一个草稿层（第 61 层）。
在一次前向推理中，主网络预测下一个 Token，而草稿层顺手预测“下下个” Token。这种投机解码（Speculative Decoding）的内置设计，让系统的吞吐量暴增了近 1.5 倍，是极致压缩推理成本的杀手锏。

---

## 质量自我审查与准出标准
1. **架构串联清楚了吗？**：必须知道 `[3层 DeltaNet + 1层 Full Attention]` 的交替比例以及它们内部是怎么使用 `Zero-Centered RMSNorm` 的。
2. **$O(1)$ 推理看懂了吗？**：能写出 `Gated Delta Rule` 的核心更新公式，并知道在流式推理时，记忆矩阵 $S$ 只有 `[B, 32, 128, 128]` 这么小，永远不会像 KV Cache 那样随上下文爆炸。
3. **稀疏度明白了吗？**：理解 512 选 10 + 1 共享专家的机制，明白为何 397B 的模型每次只激活 17B 参数。

## 关联概念
- 🔙 接收前置多模态输入：[[patchmerger_空间降维]] (桥接视觉特征)
- 🤝 同级 3D 空间赋权：[[mrope_多模态位置编码]] (在 Attention 中注入时间/行/列信息)