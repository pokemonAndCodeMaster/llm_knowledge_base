# 统一知识系统维护法则 (Schema)

> **版本**：v1（2026-05-03 按 ai-librarian Skill 规范初始化）
> **演进规则**：你（AI）与用户在实践中发现更好的模式时，主动提出修改此文件。

---

## 🏛️ 目录结构约定

```
knowledge_base/
├── schema.md          # 本文件：约束法则，随使用经验共同演进
├── ai-librarian-SKILL.md  # AI 图书管理员技能定义
├── Qwen2.5-VL/        # 原始资料：Qwen2.5-VL 技术报告解析
├── Qwen2.5-VL (preview)/  # 原始资料：PR 阶段的预览解析
├── NaViT/             # 原始资料：NaViT 论文解读
├── 面试官!.../         # 原始资料：Qwen 系列面试题解析
├── 面试官：VLM.../    # 原始资料：动态分辨率面试题
├── Qwen_Architecture_Guides/  # 原始资料：学习指南 Phase 1
├── Qwen3-Next.../     # 原始资料：Qwen3-Next 源码解析
├── ...                # 其他原始资料目录
└── wiki/              # AI 完全拥有的编译区
    ├── index.md       # 全局知识目录
    ├── log.md         # append-only 时序操作日志
    ├── sources/       # 每篇原始来源的精炼摘要
    ├── entities/      # 实体卡片（人物、组织、产品、项目）
    ├── concepts/      # 概念卡片（理论、方法、术语、工具）
    └── synthesis/     # 综合分析页（跨多来源的对比/整合/框架）
```

---

## 📝 卡片 Frontmatter 规范

每张 wiki/ 下的 Markdown 文件**必须**包含以下 YAML Frontmatter：

```yaml
---
title: 卡片完整标题
tags: [分类标签, 主题标签]
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: 引用来源数量（整数）
status: active        # active | stale | superseded
---
```

来源摘要卡（`sources/`）额外包含：
```yaml
source_url: https://...
source_type: article | twitter | youtube | github | gist | reddit
```

---

## 🔤 文件命名规范

- 文件名使用**小写中文或英文**，空格替换为下划线 `_`。
- 实体卡片：`entities/人名或组织名.md`
- 概念卡片：`concepts/概念名.md`
- 来源摘要：`sources/简短英文标识.md`
- 综合分析：`synthesis/分析主题.md`

---

## 🔗 双向链接规范

- 使用 `[[文件名（不含路径和扩展名）]]` 格式。
- 跨目录链接时只写文件名，Obsidian 会自动解析。
- 链接类型标注（可选但推荐）：
  - `✅ 支持`：新内容印证此观点
  - `❌ 反驳`：新内容与此矛盾
  - `🔄 演化自`：此观点从另一观点发展而来

---

## 📅 log.md 格式规范

每条日志必须以 `## [YYYY-MM-DD] <操作类型> | <标题>` 开头：

操作类型枚举：`init` | `ingest` | `query` | `lint` | `migrate` | `update`

---

## 🌏 语言约束

**所有 wiki/ 下的文件内容必须使用中文撰写**。即便原始资料是英文，摘要卡、概念卡的正文也需要翻译为中文。英文术语可保留原词，但解释文字必须是中文。

---

## 🔄 共同演进提示

当出现以下情况时，建议更新此 Schema：
- 某类卡片频繁出现但缺乏对应规范。
- 当前文件命名规范造成歧义。
- 某个标签体系需要扩展。
