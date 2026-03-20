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
- Retrieval 统一入口已抽到 [src/retrieval.py](/Users/jessechen/project-大模型应用/course_rag_mvp - ai/src/retrieval.py)
- Retrieval profile 统一入口已抽到 [src/retrieval_profiles.py](/Users/jessechen/project-大模型应用/course_rag_mvp - ai/src/retrieval_profiles.py)
- 分层评测体系已建立
- 最终版 benchmark `final_v1` 已创建
- `final_v1` 正式 baseline 已完成
- 第一轮 retrieval 实验已完成：`hybrid retrieval`、`metadata filtering`、`query routing v1`
- `query routing v1` 已完成 retrieval / grounded QA / support governance 三组评测

当前还没有完成：
- 将实验结果正式沉淀到统一 metrics/实验记录
- 后续可以继续做更严格的 `support governance`

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

推荐优先用 `retrieval_profile` 跑实验，而不是手动拼很多 flags：

```bash
python -m src.rag_answer_siliconflow "your question" --retrieval-profile runtime_dense
python -m src.rag_answer_siliconflow "your question" --retrieval-profile dense_metadata_v1
python -m src.rag_answer_siliconflow "your question" --retrieval-profile dense_routing_v1
python -m src.rag_answer_siliconflow "your question" --retrieval-profile hybrid_v1
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

旧版小样本 baseline 汇总在：
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

## Final_v1 Baseline

`final_v1` 正式 baseline 明细在：
- [eval/artifacts/final_v1_baseline](/Users/jessechen/project-大模型应用/course_rag_mvp - ai/eval/artifacts/final_v1_baseline)

当前正式 baseline 摘要：

- retrieval_core_final_v1 reranked
  - `Page Hit@3 = 0.964`
  - `Page MRR@3 = 0.923`
  - `Rec Hit@3 = 0.893`
  - `Rec MRR@3 = 0.845`

- qa_grounded_final_v1
  - `Action accuracy = 19/24 = 0.792`
  - `Must-include pass = 8/24 = 0.333`
  - `No forbidden claims = 24/24 = 1.000`
  - `Primary-source pass = 19/24 = 0.792`

- support_governance_final_v1
  - `Action accuracy = 8/12 = 0.667`
  - `Answered = 6/12 = 0.500`
  - `Must-include pass = 0/10 = 0.000`
  - `No forbidden claims = 10/10 = 1.000`
  - `Primary-source pass = 5/10 = 0.500`

- refusal_boundary_final_v1
  - `Refusal accuracy = 16/20 = 0.800`
  - `Actual refuses = 16/20`

## Retrieval Experiments

第一轮 retrieval 优化已经做过三组：

1. `hybrid_retrieval_v1`
- 结果目录：
  - [eval/artifacts/experiments/hybrid_retrieval_v1](/Users/jessechen/project-大模型应用/course_rag_mvp - ai/eval/artifacts/experiments/hybrid_retrieval_v1)
- 结果结论：
  - 没有超过 dense baseline
  - reranked 后约为：
  - `Page Hit@3 = 0.911`
  - `Page MRR@3 = 0.869`
  - `Rec Hit@3 = 0.821`
  - `Rec MRR@3 = 0.774`
- 解释：
  - 当前 guideline recommendation chunk 已经非常适合 dense retrieval，hybrid 带来的 lexical 信号更多是扰动，而不是有效补召回

2. `metadata_filtering_v1`
- 结果目录：
  - [eval/artifacts/experiments/metadata_filtering_v1](/Users/jessechen/project-大模型应用/course_rag_mvp - ai/eval/artifacts/experiments/metadata_filtering_v1)
- 方法：
  - 在统一 retrieval 层中加入基于 `question_type + rec_id section` 的 soft metadata filtering
- 结果结论：
  - 在 dense + rerank 上有正收益
  - reranked 后为：
  - `Page Hit@3 = 0.964`
  - `Page MRR@3 = 0.943`
  - `Rec Hit@3 = 0.911`
  - `Rec MRR@3 = 0.872`
- 解释：
  - 对当前项目来说，减少错误召回比增加额外 lexical 召回更有效

3. `query_routing_v1`
- 结果目录：
  - [eval/artifacts/experiments/query_routing_v1](/Users/jessechen/project-大模型应用/course_rag_mvp - ai/eval/artifacts/experiments/query_routing_v1)
- 方法：
  - 基于规则的 query routing
  - 第一轮按 `question_type` 自动启用 metadata filtering
  - 第二轮按 query 类型决定继续 `main-only` 还是 `main + support`
- retrieval 结果：
  - `Page Hit@3 = 0.964`
  - `Page MRR@3 = 0.943`
  - `Rec Hit@3 = 0.911`
  - `Rec MRR@3 = 0.872`
- grounded QA 结果：
  - `Action accuracy = 23/24 = 0.958`
  - `Must-include pass = 10/24 = 0.417`
  - `No forbidden claims = 24/24 = 1.000`
  - `Primary-source pass = 23/24 = 0.958`
- support governance 结果：
  - `Action accuracy = 8/12 = 0.667`
  - `Answered = 8/12 = 0.667`
  - `Must-include pass = 0/10 = 0.000`
  - `No forbidden claims = 10/10 = 1.000`
  - `Primary-source pass = 7/10 = 0.700`
- 结果结论：
  - 在 retrieval 指标上与 `metadata_filtering_v1` 基本一致
  - 但在 grounded QA 上明显优于当前 `final_v1 baseline`
  - 对 support governance 有部分帮助，尤其是 `Answered` 和 `Primary-source pass`
  - `Must-include pass` 仍然偏弱，说明回答完整性和 evidence policy 还需要继续强化

## Recommended Next Step

后续建议严格按这个顺序推进：

1. 把 `query_routing_v1` 的结果正式沉淀成统一 metrics / 实验说明
2. 在同一套 benchmark 上继续做下一轮 retrieval 优化
3. 优先考虑更严格的 `support governance`

当前最推荐的下一个优化：
- `support governance` 强化

原因：
- `hybrid retrieval` 已验证收益不佳
- `metadata filtering` 已验证有效
- `query routing v1` 已验证对 grounded QA 有明显帮助
- 当前最明显的短板转移到了回答完整性与 support evidence boundary

## Handoff For Next Codex

如果你是下一个接手这个仓库的 Codex，请直接假设以下事实成立：

- 仓库已经做过结构整理，`src/config.py` 是路径真源
- 当前主系统是 guideline-first clinical RAG，不是泛化聊天机器人
- `eval/datasets/final_v1/` 是后续正式实验应使用的统一 benchmark
- `src/retrieval.py` 已经是统一 retrieval 入口
- `src/retrieval_profiles.py` 已经是 retrieval experiment config 的统一入口
- `eval/artifacts/baseline/metrics.json` 是旧版小样本 baseline，不是最终 benchmark baseline
- `eval/artifacts/final_v1_baseline/` 才是当前正式 baseline
- `hybrid_retrieval_v1` 已完成，结论是对当前项目没有提升
- `metadata_filtering_v1` 已完成，结论是对 dense + rerank 有正收益
- `query_routing_v1` 已完成，结论是 retrieval 指标与 metadata filtering 持平，但 grounded QA 明显更好
- 目前最值得继续做的是：`support governance` 强化
- 除非明确需要升级 benchmark，否则不要覆盖 `final_v1`

## Notes

- 当前仓库保留了部分历史实验和历史索引，用于回溯，不代表它们仍是正式实验入口
- 简历和最终项目描述应优先采用统一 benchmark 下的最终指标，不要混用不同题集口径
