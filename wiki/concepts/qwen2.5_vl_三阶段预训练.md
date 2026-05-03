---
title: Qwen2.5-VL 三阶段预训练
tags: [概念, 训练策略, 预训练, SFT, DPO, Qwen2.5-VL]
created: 2026-05-03
updated: 2026-05-03
sources: 3
status: active
---

# Qwen2.5-VL 三阶段预训练

## 模块整体说明

Qwen2.5-VL 的训练遵循**三阶段预训练 + 后训练（SFT+DPO）**的策略。核心思路是逐步解冻参数、逐步提升数据复杂度，让模型先学会"看"（视觉对齐），再学会"理解"（多模态融合），最后学会"遵循指令"（后训练）。

---

## 权重初始化

| 组件 | 初始化来源 |
|------|-----------|
| LLM | Qwen2.5 LLM 预训练权重 |
| ViT | **从零使用 DataComp 数据训练**（Stage 0） |
| PatchMerger | 随机初始化 |

ViT 的预训练（Stage 0）包括三阶段 CLIP pre-training，训练中会进行分辨率的动态抽样。

---

## 三阶段预训练

### Stage 1：视觉-语言对齐

| 项目 | 详情 |
|------|------|
| **可训练** | ViT + [[patchmerger_空间降维]] |
| **冻结** | LLM |
| **数据** | Image Caption、Knowledge、OCR |
| **目标** | 使视觉部分与文本数据进行对齐 |

这个阶段的核心是让 [[patchmerger_空间降维]] 学会把视觉特征"拽"到语言模型的表征空间。

### Stage 2：多模态融合

| 项目 | 详情 |
|------|------|
| **可训练** | **全部参数**（ViT + PatchMerger + LLM） |
| **数据** | 更多样的多模态数据：视频、VQA 等 |
| **目标** | 提升多模态处理能力 |

### Stage 3：长上下文与复杂推理

| 项目 | 详情 |
|------|------|
| **可训练** | **全部参数**（ViT + PatchMerger + LLM） |
| **数据** | 增加序列长度 + 视频 + Agent 数据 |
| **目标** | 处理更长上下文和更复杂的推理任务 |

---

## 预训练数据规模

相比 Qwen2-VL 的 1.2T tokens，Qwen2.5-VL 预训练数据扩展到约 **4T tokens**。数据类型包括：

| 数据类型 | 说明 |
|---------|------|
| Image captions | 图像-文本配对 |
| Interleaved image-text | 交错图文数据（注重文本-图像相关性和信息密度平衡） |
| OCR data | 多语言 OCR 数据集（法、德、意、西、葡、阿、俄、日、韩、越） |
| Visual knowledge | 名人、地标、动植物识别 |
| Multi-modal QA | 多模态学术问答 |
| Localization data | 物体定位与空间推理 |
| Document parsing | 表格、图表、公式解析（整合进 HTML 标签结构） |
| Video data | 动态帧率采样的视频描述和定位 |
| Agent data | 移动/Web/桌面平台截图 + UI 元素定位 |

---

## 后训练（SFT + DPO）

**关键约束**：后训练阶段**均保持 ViT 冻结**。

### SFT（监督微调）

- **数据格式**：ChatML format 的 Instruction 数据
- **数据过滤**：
  1. Domain-Specific Categorization：训练 Qwen2-VL-Instag 分类器，分 8 主类 + 30 子类
  2. Domain-Tailored Filtering：rule-based（删除不完整/格式错误的响应）+ model-based（Reward Model）
- **Reasoning 数据**：使用 RM 进行 Rejection Sampling 过滤

### DPO（直接偏好优化）

- ViT 仍冻结
- 仅使用 image-text 和纯文本数据
- 每个样本只处理一次

---

## 与 Qwen2-VL 训练策略的关键差异

| 维度 | Qwen2-VL | Qwen2.5-VL |
|------|---------|------------|
| 数据量 | 1.2T tokens | **4T tokens** |
| Stage 1 冻结 | LLM 冻结 | LLM 冻结 |
| Stage 3 | 训练 LLM | **全参数训练** |
| 后训练 | 无 DPO | **新增 SFT + DPO** |
| ViT 初始化 | 675M 重新训练 | DataComp 从零训练 |

---

## 关联概念

- ✅ 支持 [[qwen2.5_vl_技术报告解析]]：训练策略是理解模型的关键。
- [[patchmerger_空间降维]]：Stage 1 重点训练的组件。
- [[conv3d_时空切块器]]：随 ViT 在 Stage 1-3 训练。
- [[window_attention_交错注意力]]：随 ViT 在 Stage 1-3 训练。
- [[swiglu_门控激活函数]]：随 ViT 在 Stage 1-3 训练。

## 参考来源

- 原始资料：`knowledge_base/Qwen2.5-VL/Qwen2.5-VL.md` §2-3
- 原始资料：`knowledge_base/面试官!从Qwen-VL到Qwen3.5技术改进？(26年2月版)/` §3.2
