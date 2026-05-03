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
