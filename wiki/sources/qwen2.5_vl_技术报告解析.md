---
title: Qwen2.5-VL 技术报告解析（来源摘要）
tags: [来源, Qwen2.5-VL, 技术报告, 多模态]
created: 2026-05-03
updated: 2026-05-03
sources: 1
status: active
source_url: https://arxiv.org/pdf/2502.13923
source_type: article
---

# Qwen2.5-VL 技术报告解析

## 来源信息

- **原始标题**：Qwen2.5-VL Technical Report
- **链接**：https://arxiv.org/pdf/2502.13923
- **知识库原始资料**：`knowledge_base/Qwen2.5-VL/Qwen2.5-VL.md`
- **预览版解析**：`knowledge_base/Qwen2.5-VL (preview)/Qwen2.5-VL (preview).md`

## 核心要点摘要

### 架构改进（相比 Qwen2-VL）

1. **ViT 改进**（最大更新）：
   - 使用交错 [[window_attention_交错注意力]]（7:1 比例，32 层中 4 层全局）
   - MLP 升级为 [[swiglu_门控激活函数]]（`silu` + `bias=True`）
   - 归一化升级为 [[rmsnorm_归一化]]
   - 保留 [[2d_rope_视觉位置编码]]（与 Qwen2-VL 相同）

2. **PatchMerger**（微改）：
   - 仅修改 LayerNorm 为 RMSNorm，详见 [[patchmerger_空间降维]]

3. **LLM 端**：
   - 语言模型升级为 Qwen2.5 LLM
   - [[mrope_多模态位置编码]] 对齐绝对时间（Time-Absolute MRoPE）

### 训练策略

三阶段预训练 + SFT + DPO，详见 [[qwen2.5_vl_三阶段预训练]]。

### Benchmark 表现

- 视觉任务和文本任务均达到 SOTA
- 在 OCR 和 Agent 子任务上表现突出

## 关联概念

- [[navit_动态分辨率]]、[[conv3d_时空切块器]]、[[window_attention_交错注意力]]
- [[swiglu_门控激活函数]]、[[rmsnorm_归一化]]、[[patchmerger_空间降维]]
- [[mrope_多模态位置编码]]、[[2d_rope_视觉位置编码]]
- [[qwen2.5_vl_三阶段预训练]]
- 综合分析：[[qwen2.5_vl_深度剖析学习指南]]
