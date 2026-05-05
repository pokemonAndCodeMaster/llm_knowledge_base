---
title: Qwen3.5 原生多模态训练范式 (Native Multimodal)
tags: [概念, 训练范式, 原生多模态, MTP, 异构并行, Qwen3.5, ms-swift]
created: 2026-05-04
updated: 2026-05-04
sources: 5
status: active
---

# Qwen3.5 原生多模态训练范式 (Native Multimodal)

## 模块整体说明

当我们称呼 Qwen3.5 为“原生多模态大模型”时，并不是指它的视觉骨干网（Vision Backbone）或桥接器（Projector）在结构上发生了颠覆（实际上它完全沿用了 2.5-VL 的 `NaViT + PatchMerger` 极简架构）。真正的革命发生在其**底层并行工程（Heterogeneous Infrastructure）**与**联合训练策略（Joint Training Paradigm）**上。

Qwen3.5 摒弃了过去“先学文本，再挂载视觉，分阶段逐渐解冻”的缝合怪路线，转而在极早期的预训练阶段，就将交错图文、视频、纯文本放在同一个池子里联合优化，并搭配 MTP (多 Token 预测) 加速收敛。

---

## 核心训练阶段拆解与数据配比

根据公开材料与模型权重分析，Qwen3.5 的训练体系彻底打破了 2.5 时代的隔阂：

### Stage 0：视觉基座连续预训练 (CPT)
在接入 LLM 之前，单独对基于 SigLIP2 思想的 ViT 进行继续预训练。
- **目标**：拓展其在原生动态分辨率下的抗形变能力。
- **数据量**：数百亿 Token 级别的纯视觉或粗粒度图文对。

### Stage 1：原生联合预训练 (Native Joint Pre-training)
这是 Qwen3.5 最核心的“原生”阶段。视觉编码器、PatchMerger 和语言大模型（DeltaNet + MoE）**全部解冻，全参数端到端联合训练**。
- **输入特征**：混合了视觉 Instruction 数据、纯文本数据、交错图文、STEM（理工科）数据。
- **数据量级**：高达 **~1T Tokens**（包含数十亿级别的图像/视频 Token）。
- **训练方式**：从一开始就不区分“这是文字”还是“这是图片”，所有特征被统一投影为 4096 维隐状态，在 DeltaNet 的 $S$ 记忆矩阵中平权流转。

### Stage 2 & 3：长上下文与极长视频扩展
- **Stage 2**：序列长度拉升至 `32768` (32K)，加入长视频和多步思维链任务，数据量约 **~1T Tokens**。
- **Stage 3**：序列长度逼近极限的 `262144` (256K)，聚焦百万像素巨图与小时级长视频的理解。

---

## 架构解耦：异构基础设施 (Heterogeneous Infrastructure)

Qwen3.5 在工程上最大的挑战在于：**它的前端（ViT）是极度消耗显存的稠密网络（Dense），而后端（LLM）是极度依赖通信带宽的稀疏网络（MoE）加线性注意力（DeltaNet）。**
如果用传统的 Megatron-LM 流水线将它们绑在一起，GPU 会因为等待对方而产生巨大的气泡。

### Qwen 团队的解法
实现了**视觉与语言组件的解耦并行策略**：
- **Vision 端**：主要使用传统的张量并行（Tensor Parallel）和数据并行（Data Parallel）跑稠密的 3D 卷积和 2D-RoPE 注意力。
- **LLM 端**：启动专家并行（Expert Parallelism）和序列并行（Sequence Parallelism）。
- 两者通过高速的 NVLink / RDMA 异步传递 `PatchMerger` 的隐状态，彻底消灭了统一架构带来的低效。

---

## 微调框架 (ms-swift) 适配源码解剖

在下游框架（如 `ms-swift` 和 `LLaMA-Factory`）中，为了支撑这种极端的架构，工程师们注入了大量特化补丁：

### 1. 序列并行补丁 (Sequence Parallel Patch)
为了让 Gated DeltaNet 在多卡下支持极长文本，必须重写它的底层算子。
**代码路径**：`swift/model/models/qwen.py`
```python
def _patch_qwen3_5_linear_attention_sequence_parallel() -> None:
    # 动态拦截 Qwen3_5GatedDeltaNet
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    
    if getattr(Qwen3_5GatedDeltaNet, '_ms_swift_sp_linear_patched', False):
        return
        
    origin_forward = Qwen3_5GatedDeltaNet.forward
    
    # 重写 forward，将其分发给支持序列并行的底层 C 算子
    def sp_linear_forward(self, hidden_states, ...):
        return _run_qwen3_5_gated_delta_net_sequence_parallel_forward(...)
        
    Qwen3_5GatedDeltaNet.forward = sp_linear_forward
    Qwen3_5GatedDeltaNet._ms_swift_sp_linear_patched = True
```

### 2. FP8 低精度解耦训练 (Linear Decoupled in_proj)
针对 397B 的庞然大物，即使是微调也必须借助 FP8。但是在 DeltaNet 中，如果把包含门控权重的线性层全转 FP8 会导致严重的精度坍塌。
**解决方案**：`linear_decoupled_in_proj=True`
- 将原有的 $Q,K,V,Z$ 与 $\beta, \alpha$ 分离开来：
  - `in_proj_qkvz`：可以使用 FP8 进行极限加速。
  - `in_proj_ba`：由于 $\beta$（写入衰减）和 $\alpha$（门控衰减）对数值极度敏感，它们必须被**解耦出来，保留 BF16/FP32 原始精度**进行微调。

---

## MTP (Multi-Token Prediction) 联合提速训练

Qwen3.5 不仅在推理时用 MTP，在**预训练和 SFT 时同样启用了 MTP 联合训练**。

### 逻辑链输入与输出
- **输入**：正常的前向隐状态 `[B, S, D]`。
- **操作**：
  在主干网络（如前 60 层）的顶部，堆叠一个额外的“草稿层”（第 61 层）。
  主干网络预测第 $t+1$ 个 Token，计算 $Loss_1$。
  草稿网络基于主干的输出，预测第 $t+2$ 个 Token，计算 $Loss_2$。
- **输出**：联合损失 $Loss_{total} = Loss_1 + \lambda Loss_2$。

### 第一性原理与原理解读
传统模型学的是“看图说话的第一反应”，而引入 MTP 后，模型被逼着去预测**更长远的未来**。这不仅能让解码吞吐量翻 1.5 倍，更重要的是，它**倒逼前面的视觉编码器提取出更深刻、更具全局逻辑连贯性的图像特征**，这也是“原生多模态”在效果上碾压拼接式 VLM 的秘密武器。

---

## 关联概念
- 🔙 它的极简化前置架构：[[patchmerger_空间降维]]
- 🤝 它的底座支撑：[[llm_backbone_大语言模型基座]]