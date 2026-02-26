# Acne RAG Assistant (MVP)

## 1. 项目简介
一个基于检索增强生成（RAG）的痤疮科普助手，用于提供**有引用依据**的通用健康科普信息与就医分流建议。

> 目标不是诊断或处方，而是“基于权威资料的科普 + 引用 + 拒答”。

---

## 2. 项目目标（MVP）
MVP 只解决这几件事：
- 回答痤疮相关基础科普问题（概念、常见诱因、一般护理）
- 给出引用来源（文档 / 页码 / 章节）
- 对高风险问题或证据不足问题进行拒答
- 输出结构化 JSON（便于后续接 API / 前端 / 评测）

---

## 3. 项目边界（必须严格遵守）

### ✅ In Scope（做）
- 痤疮基础科普（定义、类型、常见误区）
- 常见治疗方式的通用科普解释（不涉及处方剂量）
- 一般护肤与生活方式建议（基于权威资料）
- 何时建议咨询药师 / GP / 皮肤科（就医分流）
- “这个结论来自哪里”的引用展示

### ❌ Out of Scope（不做）
- 诊断（不能判断“你这是不是痤疮/毛囊炎/玫瑰痤疮”）
- 处方建议（不能推荐具体处方药、剂量、疗程）
- 个体化用药方案调整
- 图片诊断（MVP阶段不支持）
- 紧急医疗处理建议

---

## 4. 安全策略（Safety Policy）
本项目为**科普助手**，不是医生替代品。

### 基本原则
- 非诊断（Non-diagnostic）
- 非处方（Non-prescription）
- 仅科普（Education only）
- 证据不足时拒答（Insufficient evidence -> refuse）

### 拒答/转介场景（示例）
当用户请求以下内容时，应拒答或建议线下就医：
- “帮我诊断是不是痤疮”
- “给我开药 / 告诉我剂量”
- “我怀孕了能不能吃XX药”
- “症状严重/快速恶化/疼痛明显/可能感染”
- “根据图片判断皮肤问题”

---

## 5. 输入输出定义（I/O Contract）
后续所有代码都围绕这里的结构实现。

### 输入（Request）
至少包含：
- `request_id`
- `question`
- `language`
- `attachments`（MVP可为空）
- `mode`（如 `education`）

### 输出（Response）
至少包含：
- `request_id`
- `status`（`answer` / `refuse` / `insufficient_evidence`）
- `answer`
- `citations`
- `safety`
- `evidence_confidence`

> 详细 JSON Schema 见：
- `schemas/request.schema.json`
- `schemas/response.schema.json`

---

## 6. 数据来源策略（Knowledge Base Policy）
为保证可信度，知识库优先使用：
- 临床指南 / 官方医疗机构资料
- 专业皮肤科协会 / 国家健康机构患者教育资料
- 可追溯来源（保留 URL、标题、页码/章节）

### 数据要求
- 每条 chunk 必须可回溯到原文（doc/page/section）
- 禁止使用无法验证来源的内容作为核心知识
- 不编造引用

---

## 7. 项目结构（当前阶段）
```text
project/
├─ README.md
├─ schemas/
│  ├─ request.schema.json
│  └─ response.schema.json
├─ data/
│  ├─ raw_docs/
│  │  └─ acne/
│  │     ├─ pdf/
│  │     ├─ web_html/
│  │     ├─ web_md/
│  │     └─ manifest.jsonl
│  └─ processed/
│     └─ acne/
│        ├─ chunks.jsonl
│        └─ ingest_report.json
└─ src/
   └─ ingest/