# Acne Guideline RAG Agent

一个基于 NICE NG198 acne guideline 的课程型 RAG 项目。系统优先检索主 guideline，在证据不足时再补充 evidence review 文档，并通过一个简单的 LangGraph agent 决定是直接回答、改写查询后二次检索，还是拒答。

## Current Architecture

主链路分成两部分：

1. 离线构建
   - `src/ingest.py`：读取 PDF，清洗页面文本，生成统一 `chunks.jsonl`
   - `src/split_chunk.py`：把统一 chunk 拆成 `main` 和 `support`
   - `src/build_index.py`：为 `main` / `support` / `all` 构建 embedding 与 FAISS 索引

2. 在线问答
   - `src/rag_core.py`：加载索引、向量检索、过滤低质量 chunk、构造上下文、调用 LLM
   - `src/agent_graph.py`：两轮 agent 流程
   - `src/rag_answer_siliconflow.py`：命令行入口

## Retrieval Flow

```text
User Query
  -> retrieve main guideline
  -> rerank
  -> judge evidence
      -> sufficient: answer
      -> insufficient: rewrite query
  -> retrieve main + support
  -> rerank
  -> judge evidence
      -> sufficient: answer
      -> insufficient: refuse
```

## Project Layout

```text
course_rag_mvp/
├── src/                    # 主代码
├── data/
│   ├── raw_docs/           # 原始 PDF 与 manifest
│   └── processed/          # 处理后的 chunks
├── artifacts/
│   ├── index_main/         # main corpus FAISS 与 embedding cache
│   ├── index_support/      # support corpus FAISS 与 embedding cache
│   └── index/              # unified corpus 索引（可选历史产物）
├── eval/                   # 离线评测脚本
├── scripts/                # 调试脚本
├── archive/
│   └── experiments/        # 历史实验记录
├── schemas/                # 预留的 schema 文件，当前主链路未使用
├── requirements.txt
└── README.md
```

## Environment

推荐使用 Python 3.11。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

需要的环境变量：

```bash
export SILICONFLOW_API_KEY="your_key"
```

可选环境变量：

```bash
export OPENAI_API_KEY="your_key"           # 只有重建 embedding/index 时才需要
export LOCAL_EMBED_MODEL="all-MiniLM-L6-v2"
```

## Run

直接提问：

```bash
python -m src.rag_answer_siliconflow "What does NICE recommend for acne-related scarring?"
```

调试 agent 中间状态：

```bash
python scripts/debug_agent.py
```

## Rebuild Data Artifacts

如果需要从原始文档重新构建：

```bash
python -m src.ingest
python -m src.split_chunk
python -m src.build_index --target main
python -m src.build_index --target support
```

`artifacts/` 下的索引和 `eval/artifacts/` 下的评测输出都属于可再生成产物，不是核心源码。

## Evaluation

生成 baseline retrieval：

```bash
python eval/dump_retrieval_results.py
```

做 cross-encoder rerank：

```bash
python eval/rerank_cross_encoder.py
```

计算 Hit@k / MRR@k：

```bash
python eval/eval_retrieval.py
```

评测输出会写到 `eval/artifacts/`。

## Cleanup Decisions

本次整理遵循这些规则：

- 根目录只保留源码、数据、评测、脚本和说明文档
- 向量索引统一收进 `artifacts/`
- 历史实验统一收进 `archive/experiments/`
- 本地虚拟环境和缓存不进仓库
- 调试脚本收进 `scripts/`，不与正式入口混放

## Known Gaps

- `schemas/` 当前未接入主运行链路，后续可以删除或接到 API 层
- `artifacts/index/` 是 unified corpus 的历史索引，若确定不用可删除
- `src/agent_graph.py` 仍在 import 时加载索引，后续可以改为懒加载或封装成类
- 项目还没有正式测试目录，当前只有调试脚本
