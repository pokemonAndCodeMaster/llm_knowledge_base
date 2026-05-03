# 🧠 知识库索引 (Index)

> 导航规则：先看📌总纲，再按需跳入对应分类。

---

## 📌 综合分析（Synthesis）

| 文件 | 摘要 |
|---|---|
| [[qwen2.5_vl_深度剖析学习指南]] | Qwen2.5-VL 全链路硬核学习指南：从 Processor 预处理到 LLM 自回归解码的完整五模块无死角串联 |

---

## 💡 核心概念（Concepts）

### 🏗️ 视觉编码器组件

| 文件 | 摘要 |
|---|---|
| [[navit_动态分辨率]] | NaViT 原生动态分辨率方案：保持原始分辨率和宽高比，Patch n' Pack 序列打包，分解式位置编码 |
| [[conv3d_时空切块器]] | 3D 卷积时空切块器 (VisionPatchEmbed)：将像素立方体切分为 1152 维特征向量，kernel=(2,14,14) |
| [[window_attention_交错注意力]] | 交错窗口注意力：32 层中 4 层全局 / 28 层窗口，7:1 黄金比例，降低 $O(N^2)$ 到 $O(N \times W)$ |
| [[swiglu_门控激活函数]] | SwiGLU 门控 MLP：gate×up→down 三路结构，ViT 中开启 bias=True 对抗 DC Offset |
| [[rmsnorm_归一化]] | RMSNorm 归一化：省去均值计算，权重初始化为 1，后续演化为 Zero-Centered RMSNorm |
| [[patchmerger_空间降维]] | PatchMerger 空间降维桥接器 (Projector)：2×2 合并 + MLP 投影，4608→4096，Token 数砍 75% |

### 🧭 位置编码

| 文件 | 摘要 |
|---|---|
| [[2d_rope_视觉位置编码]] | 2D-RoPE 视觉位置编码：ViT 内部使用，将 head_dim 均分为 X/Y 两轴独立旋转 |
| [[mrope_多模态位置编码]] | MRoPE 多模态旋转位置编码：LLM 内部的三维时空编码 (T,H,W)，Qwen2.5-VL 对齐绝对物理时间 |

### 📐 架构对比

| 文件 | 摘要 |
|---|---|
| [[动态分辨率方案对比]] | 原生动态分辨率 vs Dynamic Tiling，视觉 Token 压缩四分类对比 |

### 🎓 训练策略

| 文件 | 摘要 |
|---|---|
| [[qwen2.5_vl_三阶段预训练]] | 三阶段预训练（ViT→全参→长上下文）+ 后训练（SFT+DPO），4T tokens 数据规模 |

---

## 📄 来源摘要（Sources）

| 文件 | 来源 | 摘要 |
|---|---|---|
| [[qwen2.5_vl_技术报告解析]] | arXiv 技术报告 | Qwen2.5-VL 官方技术报告的架构改进、训练策略和实验结果摘要 |

---

## 🔗 原始资料总览（Raw）

知识库中的原始资料目录（不可修改）：
- `Qwen2.5-VL/` — 官方技术报告解析
- `Qwen2.5-VL (preview)/` — Transformers PR 阶段预览解析
- `NaViT/` — NaViT 论文解读
- `Qwen_Architecture_Guides/` — 学习指南 Phase 1（原始版）
- `面试官!从Qwen-VL到Qwen3.5技术改进？(26年2月版)/` — Qwen 系列面试题对比
- `面试官：VLM 的动态分辨率是怎么做的？/` — 动态分辨率面试题
- `Qwen3-Next模型结构与源码逻辑_骑虎南下_2026-01-06/`
- `Qwen3-Next源码解析：一文看清架构中的核心点实现_李先生_2025-10-20/`
- `Qwen3.5源码逻辑与模型架构_骑虎南下_2026-03-17/`
- `【LLM】Qwen3.5解剖_Plunck_2026-02-15/`
- `MCP_Guides/`
