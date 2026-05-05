---
title: 动态分辨率输入方案解读（来源摘要）
tags: [来源, 动态分辨率, NaViT, InternVL, Token压缩]
created: 2026-05-04
updated: 2026-05-04
sources: 3
status: active
source_type: article
---

# 动态分辨率输入方案与视觉 Token 压缩

## 来源信息
本来源摘要整合了关于多模态大模型（VLM）处理不同分辨率图像的进阶技术与痛点讨论文章：
1. **NaViT (Native Resolution ViT)**：Google 提出的保留原始分辨率与宽高比的序列打包策略（Patch n' Pack）。
2. **[qwen2vl-internvl2.5] 动态分辨率输入方案解读**：详细对比了 Qwen2-VL（原生动态分辨率/整体缩放）与 InternVL2.5（基于切片 Dynamic Tiling）在源码实现上的差异、Padding 逻辑和 Token 消耗。
3. **面试官：VLM 的动态分辨率是怎么做的？** / **为什么qwen2-vl 比qwen-vl 更多保留图片信息？**：系统总结了 VLM 动态分辨率的两大流派（Native vs Tiling），以及为解决视觉 Token 过长问题而衍生的四大类视觉 Token 压缩技术（Transformation, Similarity, Attention, Query）。

## 核心要点摘要

### 1. 为什么需要动态分辨率？
传统的 ViT 会把所有图像 Resize 到固定的尺寸（如 224x224）。对于长宽比极端的图像，这会造成严重扭曲；对于高分辨率图像（如包含密密麻麻文字的 OCR 图片），会造成细节的毁灭性丢失。

### 2. 两大主流流派对比
*   **原生动态分辨率 (Native Dynamic Resolution)**：
    *   **代表模型**：Qwen2-VL, Qwen2.5-VL, Qwen3-VL, GLM OCR 等。
    *   **做法**：废除固定的绝对位置编码，改用 2D-RoPE。直接按原始比例对图片进行一定范围的缩放（限制最大/最小像素数），随后直接按 Patch 序列化，送入 ViT。
    *   **优劣**：结构简单，保留全局感受野；但在组 Batch 时由于各序列长度不一，需要像文本一样进行 Pad 对齐，或者采用 NaViT 的 Patch n' Pack 打包策略，辅以复杂的 Attention Mask 来隔离。
*   **基于切片的动态分辨率 (Dynamic Tiling)**：
    *   **代表模型**：LLaVA-NeXT, InternVL 2/3, STEP3-VL 等。
    *   **做法**：将原始图像切分为若干个固定的标准块（例如 448x448 的 Tile），并额外保留一张全局缩略图（Thumbnail）。这些切块和缩略图独立通过 ViT 后，再拼接送入 LLM。
    *   **优劣**：能复用现成的高分辨率 ViT 权重，工程实现友好，局部高清细节保留极好；但切片边缘处容易产生割裂感，且引入了大量的额外 Token（尤其是带重叠的切片）。

### 3. 视觉 Token 压缩 (Vision Token Compression)
引入动态分辨率后，高分辨率图像会产生海量的视觉 Token（动辄数千上万），严重拖慢 LLM 的推理并挤占上下文空间。目前有四大压缩方向：
1.  **基于变换 (Transformation-based)**：如 Qwen 系列的 PatchMerger (2x2 合并)、InternVL 的 Pixel Unshuffle，简单粗暴但压缩率固定。
2.  **基于相似性 (Similarity-based)**：利用聚类或余弦相似度，将背景或冗余的 Token 融合（如 ToMe）。
3.  **基于注意力 (Attention-based)**：利用 ViT 内部的 Attention 权重，剪除不被关注的背景 Token（如 FastV）。
4.  **基于查询 (Query-based)**：用极少量的 Query 向量主动去提取图像特征（如 Q-Former），压缩率极高，但可能丢失细粒度信息。

## 关联概念卡片
- 👉 原生分辨率的理论与实战：[[navit_动态分辨率]]
- 👉 跨模态桥接与初级压缩：[[patchmerger_空间降维]]