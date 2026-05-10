# 📅 知识库操作日志 (Log)

> append-only 时间线日志。每条记录使用一致前缀，方便 `grep "^## \[" log.md` 解析。

---

## [2026-05-03] init | 知识库初始化与 Qwen2.5-VL 全线摄入

- 初始化: schema.md, ai-librarian-SKILL.md, wiki/ 目录结构
- 新建 (concepts): navit_动态分辨率.md, conv3d_时空切块器.md, window_attention_交错注意力.md, swiglu_门控激活函数.md, rmsnorm_归一化.md, patchmerger_空间降维.md, mrope_多模态位置编码.md, 2d_rope_视觉位置编码.md, 动态分辨率方案对比.md, qwen2.5_vl_三阶段预训练.md
- 新建 (sources): qwen2.5_vl_技术报告解析.md
- 新建 (synthesis): qwen2.5_vl_深度剖析学习指南.md（完整保留 Phase 1 学习指南内容，注入 [[wikilinks]] 双向链接）
- 新建: index.md, log.md
- 触发: 用户请求将 ai-knowledge-base 的知识管理 skill 搬入 llm_base 项目，并对 Qwen2.5-VL 学习线路全面编译
- 摄入来源:
  - knowledge_base/Qwen2.5-VL/Qwen2.5-VL.md
  - knowledge_base/Qwen2.5-VL (preview)/Qwen2.5-VL (preview).md
  - knowledge_base/NaViT/NaViT.md
  - knowledge_base/Qwen_Architecture_Guides/qwen_learning_guide_phase1.md
  - knowledge_base/面试官!从Qwen-VL到Qwen3.5技术改进？(26年2月版)/
  - knowledge_base/面试官：VLM 的动态分辨率是怎么做的？/
  - qwen_learning_guide_phase0.md (前置知识)

## [2026-05-10] ingest | Qwen 全系列代码地图构建

- 新建 (concepts): qwen_代码地图.md — 跨 4 仓库（Transformers/vLLM/Swift/LLaMA-Factory）的 Qwen 全系列代码索引
- 更新: index.md — 新增「💻 代码导航」分类，新增 qwen3.5_深度剖析学习指南 索引
- 覆盖范围: 16 个 Transformers 模型目录、20+ vLLM 模型文件、Swift 加载器/模板、LLaMA-Factory 模板/脚本
- 重点: Qwen3.5 架构继承链（Qwen3-Next Text + Qwen3-VL Vision）、GatedDeltaNet 混合架构、阅读路线图
- 触发: 用户请求梳理 Qwen 代码文件并创建可索引的代码查阅文档

## [2026-05-10] ingest | Qwen3.5 前向传播全链路代码拆解

- 新建 (concepts): qwen3.5_前向传播全链路.md — 从 input 到 output 的完整 forward 函数调用链，含 5 大阶段、张量形状追踪、Mermaid 架构图
- 新建 (concepts): qwen3.5_gated_delta_net.md — GatedDeltaNet 线性注意力完整解剖：投影拆分、因果卷积、门控信号、Delta Rule 算法、RMSNormGated
- 新建 (concepts): qwen3.5_混合decoder架构.md — 混合 Decoder 路由机制、双 Mask 系统、KV Cache 二态管理、标准 Attention (QKNorm) 详解
- 新建 (concepts): qwen3.5_多模态融合机制.md — masked_scatter 占位符替换、Qwen3.5 删除 DeepStack 的差异、特殊 token ID
- 新建 (concepts): qwen3.5_interleaved_mrope.md — 交错式 MRoPE 频率排列算法、4D position_ids、视频时间戳分帧
- 更新: index.md — 新增「🧬 Qwen3.5 架构组件」分类，新增 5 张卡片索引
- 源码阅读范围:
  - `models/qwen3_5/modular_qwen3_5.py` (701行) — 完整阅读
  - `models/qwen3_vl/modeling_qwen3_vl.py` (1773行) — 完整阅读
- 触发: 用户请求以 Qwen3.5 为例串讲完整 forward 流经的每个函数/类

## [2026-05-10] update | 知识卡片质量升级 (Round 1)

- 更新: GEMINI.md v1.1 — 新增「知识优先查询规范」(Knowledge-First Protocol) + 「具体样本推演」写入要求
- 重写 (concepts): qwen3.5_前向传播全链路.md — 以 448×448 图+文本为具体样本推演每一步 I/O 张量形状
- 新建 (concepts): qwen3.5_视觉编码器.md — 视觉编码全链路：Conv3d→位置嵌入→RoPE→VisionBlock→PatchMerger 含完整源码
- 更新: index.md — 新增视觉编码器卡片索引
- 触发: 用户6点反馈（知识优先查询/具体推演/完整代码/SKILL标准对齐）
- 待续: GatedDeltaNet/混合Decoder/多模态融合/MRoPE 卡片扩充，现有卡片增补 Qwen3.5 内容

## [2026-05-10] update | 知识卡片质量升级 (Round 2)

- 重写 (concepts): qwen3.5_gated_delta_net.md — 具体推演(268 tokens)、四路投影/Conv1d/门控/Delta Rule 完整源码、数值示例
- 重写 (concepts): qwen3.5_混合decoder架构.md — 标准 Attention 完整源码(QKNorm+GQA)、双Mask、SwiGLU MLP
- 重写 (concepts): qwen3.5_多模态融合机制.md — get_rope_index 逐token位置分配数值推演、DeepStack删除说明
- 重写 (concepts): qwen3.5_interleaved_mrope.md — 频率索引交错分配数值推演、partial_rotary_factor
- 触发: 用户反馈要求完整代码详解+具体推演
- 待续: 现有卡片(conv3d/patchmerger/mrope)增补 Qwen3.5 内容

## [2026-05-10] update | 现有卡片增补 Qwen3.5 内容 (Round 3)

- 更新 (concepts): conv3d_时空切块器.md — 新增 Qwen3.5 章节：448×448推演、embed_dim=1280/bias=True配置差异、继承链源码
- 更新 (concepts): mrope_多模态位置编码.md — 新增 Qwen3.5 Interleaved MRoPE 章节：配置差异表、交错排列物理意义、链接到专用卡片
- 更新 (concepts): patchmerger_空间降维.md — 新增 Qwen3.5 双向链接（内容已有完整 Qwen3.5 源码解剖）
- 触发: 用户要求继续执行升级计划

## [2026-05-10] update | 学习指南合并与丰富 (Round 4)

- 合并重写 (synthesis): qwen3.5_深度剖析学习指南.md (513行)
  - 合并了 qwen3.5_前向传播全链路.md 的全链路代码调用内容
  - 对标 qwen2.5_vl_深度剖析学习指南.md (536行) 的深度和结构
  - 8 个章节: 全景架构、Processor预处理、文本嵌入、视觉编码、多模态融合、混合Decoder、训练范式、版本演化
  - 每章含: 架构解读+流转图+具体I/O推演+多层级概念学习+Q&A+源码路径
  - 修正 token 数量推演: chat template 展开后约 272 tokens
  - 插入官方架构图: raw/Qwen-3.5 Preview/ → synthesis/images/qwen3.5/
- 更新: index.md — 标注前向传播全链路已合并至学习指南
- 触发: 用户5点反馈（合并文档/丰富Processor/修正token数/概要说明/插图）

## [2026-05-10] create | 补全学习指南缺失的概念卡片 (Round 5)

- 新建 (concepts): qwen3.5_processor预处理.md — mm_token_type_ids 模态标记、图像占位符展开源码、视频帧时间戳 _calculate_timestamps 推演、Qwen2.5-VL 差异表
- 新建 (concepts): qwen3.5_文本嵌入与特殊token.md — nn.Embedding 查表原理与参数量、ChatML 模板逐 token 拆解、特殊 Token 角色表、占位 Embedding 生命周期
- 更新 (synthesis): qwen3.5_深度剖析学习指南.md — §2/§3 补充链接到新卡片
- 更新: index.md — 新增两张卡片到 Qwen3.5 架构组件区
- 触发: 用户要求补全所有学习概念的卡片跳转
