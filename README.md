# Acne Guideline RAG Agent

基于 NICE NG198 Acne Guideline 的 guideline-first clinical RAG 项目。系统目标不是开放式医学聊天，而是尽量基于 guideline recommendation 回答 acne 相关问题；当 primary guideline 证据不足时，才引入 support 文档补充，并在证据不足时拒答。

这个 README 同时承担两件事：
- 帮当前使用者快速跑通项目
- 让下一个 Codex 会话能立刻理解项目现状和下一步工作

## Current Status

当前项目已经完成：
- 主链路整理：`ingest -> split -> build_index -> retrieve -> judge -> rewrite -> answer/refuse`
- 路径与产物统一：集中在 [src/config.py](/Users/jessechen/project-大模型应用/course_rag_mvp - ai/src/config.py)
- Agent 检索资源改为按需加载，避免 import 时直接加载大索引
- 分层评测体系已建立
- 最终版 benchmark `final_v1` 已创建
- 旧版小样本 baseline 已完成并归档

当前还没有完成：
- `final_v1` 上的正式 baseline 复跑
- 第一轮新的 retrieval 优化实验，建议从 `hybrid retrieval` 或 `metadata filtering` 开始

## System Architecture

系统采用 guideline-first retrieval：

```text
User Query
-> Retrieve from main guideline
-> Rerank
-> Judge evidence sufficiency
-> sufficient: answer
-> insufficient: rewrite query
-> Retrieve from main + support
-> Rerank
-> Judge again
-> sufficient: answer
-> insufficient: refuse
```

设计原则：
- `main guideline` 是 primary authority
- `support` 只能补充，不能替代 recommendation
- 若没有足够 primary evidence，系统应倾向拒答

## Main Entry Points

正式问答入口：

```bash
python -m src.rag_answer_siliconflow "your question"
```

调试入口：

```bash
python scripts/debug_agent.py
```

离线数据构建：

```bash
python -m src.ingest
python -m src.split_chunk
python -m src.build_index --target main
python -m src.build_index --target support
```

## Repository Layout

```text
src/
  agent_graph.py              Agent decision flow
  rag_core.py                 Retrieval, context building, LLM calls
  build_index.py              FAISS index construction
  ingest.py                   PDF parsing and chunk creation
  split_chunk.py              Split main vs support corpora
  config.py                   Shared paths and repo-level configuration

data/
  raw_docs/                   Source PDFs and manifests
  processed/                  Chunk JSONL files

artifacts/
  index_main/                 Main guideline FAISS index
  index_support/              Support corpus FAISS index
  index/                      Historical unified index

eval/
  datasets/                   Evaluation datasets
  artifacts/                  Baseline and experiment outputs
  *.py                        Evaluation scripts

tests/
  Minimal unit tests

scripts/
  Debug helpers

archive/
  experiments/                Historical experiment records
```

## Environment Setup

推荐环境：
- Python 3.11

创建环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

必须环境变量：

```bash
export SILICONFLOW_API_KEY="your_key"
```

可选环境变量：

```bash
export OPENAI_API_KEY="your_key"
export LOCAL_EMBED_MODEL="all-MiniLM-L6-v2"
```

## Tests And Validation

运行单元测试：

```bash
python -m unittest discover -s tests
```

校验最终 benchmark：

```bash
python eval/validate_datasets.py --index-path eval/datasets/final_v1/index.json
```

## Evaluation Overview

项目评测分成四层，不再只看单一 retrieval 指标：

1. `retrieval_core`
测能否找回正确 guideline 证据，并排到前面。

2. `qa_grounded`
测在证据存在时，回答是否抓住 recommendation，且没有越出证据乱说。

3. `support_governance`
测 support 文档何时只能补充、不能越权。

4. `refusal_boundary`
测系统是否能在越界、个体化建议、证据不足时拒答。

## Final Benchmark

后续正式实验统一使用：
- [eval/datasets/final_v1/index.json](/Users/jessechen/project-大模型应用/course_rag_mvp - ai/eval/datasets/final_v1/index.json)
- 说明文档：[eval/datasets/final_v1/README.md](/Users/jessechen/project-大模型应用/course_rag_mvp - ai/eval/datasets/final_v1/README.md)

`final_v1` 当前规模：
- retrieval: 56 questions
- grounded QA: 24 questions
- support governance: 12 questions
- refusal boundary: 20 questions

说明：
- `final_v1` 是后续统一 benchmark
- 如果未来继续扩题，建议新增 `final_v2`
- 不要覆盖 `final_v1`

## Existing Baseline

目前已经完成的是旧版小样本分层评测 baseline，汇总在：
- [eval/artifacts/baseline/metrics.json](/Users/jessechen/project-大模型应用/course_rag_mvp - ai/eval/artifacts/baseline/metrics.json)

当前 baseline 摘要：

- retrieval_core_v2
  - `Page Hit@3 = 0.933`
  - `Page MRR@3 = 0.822`
  - `Rec Hit@3 = 0.933`
  - `Rec MRR@3 = 0.822`

- qa_grounded_v1
  - `Action accuracy = 0.583`
  - `Must-include pass = 0.000`
  - `No forbidden claims = 1.000`
  - `Primary-source pass = 0.583`

- support_governance_v1
  - `Action accuracy = 0.750`
  - `Must-include pass = 0.000`
  - `No forbidden claims = 1.000`
  - `Primary-source pass = 0.667`

- refusal_boundary_v1
  - `Refusal accuracy = 0.917`
  - `Actual refuse rate = 0.917`

对应明细文件在：
- [eval/artifacts/baseline](/Users/jessechen/project-大模型应用/course_rag_mvp - ai/eval/artifacts/baseline)

## Recommended Next Step

后续建议严格按这个顺序推进：

1. 用 `eval/datasets/final_v1/` 重跑一版正式 baseline
2. 固定这组结果作为后续统一对照
3. 选择第一个真正的新优化项

推荐第一个优化：
- `hybrid retrieval`
- 或 `metadata filtering`

原因：
- 它们最容易和当前 retrieval baseline 做清晰对比
- 改动范围适中
- 面试价值高，容易讲清楚

## Handoff For Next Codex

如果你是下一个接手这个仓库的 Codex，请直接假设以下事实成立：

- 仓库已经做过结构整理，`src/config.py` 是路径真源
- 当前主系统是 guideline-first clinical RAG，不是泛化聊天机器人
- `eval/datasets/final_v1/` 是后续正式实验应使用的统一 benchmark
- `eval/artifacts/baseline/metrics.json` 是旧版小样本 baseline，不是最终 benchmark baseline
- 下一步最值得做的是：在 `final_v1` 上重跑 baseline，然后开始第一轮 retrieval 优化实验
- 除非明确需要升级 benchmark，否则不要覆盖 `final_v1`

## Notes

- 当前仓库保留了部分历史实验和历史索引，用于回溯，不代表它们仍是正式实验入口
- 简历和最终项目描述应优先采用统一 benchmark 下的最终指标，不要混用不同题集口径
