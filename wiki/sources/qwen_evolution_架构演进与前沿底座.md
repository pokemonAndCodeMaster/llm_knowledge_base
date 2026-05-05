---
title: Qwen 架构演进与前沿底座（来源摘要）
tags: [来源, Qwen2.5, Qwen3, Qwen3.5, MoE, DeltaNet]
created: 2026-05-04
updated: 2026-05-04
sources: 6
status: active
source_type: article
---

# Qwen 架构演进与前沿底座

## 来源信息
本来源摘要整合了多篇探讨 Qwen 系列多模态大模型从早期 Qwen-VL 一路狂奔至 Qwen3.5 的架构演进、源码剖析和技术报告解析：
1. **面试官!从Qwen-VL到Qwen3.5技术改进？**：从宏观上串联了从最初直接套用 OpenCLIP，到自己训练原生动态分辨率 ViT，再到 Qwen3 的 DeepStack 融合与 MoE 引入的整条演化时间线。
2. **Qwen3-VL 技术报告 / Preview**：揭示了 Qwen3-VL 的 Interleaved MRoPE、DeepStack 多层视觉注入机制，以及丰富的多模态 RL（SAPO 算法）训练手段。
3. **Qwen3-Next 模型结构与源码逻辑** / **Qwen3.5 源码逻辑与模型架构**：硬核解析了 Qwen3-Next 及 Qwen3.5 的 Python 源码，揭露了 `Zero-Centered RMSNorm`、`Gated DeltaNet`（线性注意力）、`MTP (Multi-Token Prediction)` 等革命性创新的运算细节。
4. **【LLM】Qwen3.5解剖**：进一步深入到了并行（Chunkwise）和串行（Recurrent）状态更新的底层算法实现。

## 核心要点摘要

### 1. 视觉端的进化之路
- **Qwen1-VL**：照搬 CLIP 的 ViT-bigG 和 1D 可学习绝对位置编码，使用单层 Cross-Attention 作为 Projector，分辨率死板。
- **Qwen2-VL & Qwen2.5-VL**：拥抱 NaViT 理念，全面转向 2D-RoPE 与 PatchMerger 桥接器，在视觉端使用了 Window Attention（交错窗口）与带 Bias 的 SwiGLU，彻底激活了原生动态分辨率。
- **Qwen3-VL**：不仅限于将最终特征送给 LLM，还引入了 **DeepStack** 机制。ViT 中间层的特征被提取出来，直接残差注入到了 LLM 的前 K 层中，让低频像素细节与高频语言抽象进行强强融合。

### 2. LLM 基座的革命 (Qwen3 / 3.5)
- **计算瓶颈的突破 (Gated DeltaNet)**：为了容纳 256K 超长文本和无穷无尽的视频流，传统 Self-Attention 的 $O(N^2)$ KV Cache 是致命的。Qwen3.5 采用 3:1 比例引入线性注意力，通过 `RNN` 式的衰减记忆矩阵，让解码开销变成了 $O(1)$。
- **门控哲学的登峰造极**：从带有 `Zero-Centered` 初值的 RMSNorm（确保初期完美恒等映射），到 Attention 输出结果的 `sigmoid(gate)` 过滤，再到 Shared Expert 强制经过的门控衰减，模型对“信息摄入”变得极其克制与精准。
- **稀疏计算 (MoE) 与 MTP**：庞大的 397B 模型实际上是一个拥有 512 个专家的重型 MoE，每次仅激活 10 个专家（约 17B 参数）。此外，引入 MTP 额外预测下下个 Token，实现吞吐量的倍增。

## 关联概念卡片
- 👉 大语言模型底座全景：[[llm_backbone_大语言模型基座]]
- 👉 时空坐标系的融合演进：[[mrope_多模态位置编码]]
- 👉 视觉端的压缩桥接：[[patchmerger_空间降维]]