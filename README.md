```md
# Acne Guideline RAG Agent

基于 **NICE NG198 Acne Guideline** 的 RAG + Agent 医疗问答系统。

系统目标：

- 优先基于 **官方 guideline** 回答问题  
- 当 guideline 证据不足时引入 **evidence review 文档**  
- 通过 **LangGraph Agent** 判断证据是否充分  
- 支持 **query rewrite + 二次检索**

该项目主要用于 **RAG 检索策略与 Agent 决策逻辑实验**。

---

# 系统架构

系统采用 **Guideline-first Retrieval Strategy**：

```

Guideline (main corpus)
↓
Evidence Review (support corpus)

```

优先检索 guideline 的原因：

- guideline 是 **正式 recommendation**
- evidence review 包含大量讨论性内容
- 若先检索 evidence review 会污染 top-k 结果

---

# Agent Pipeline

系统完整流程：

```

User Query
↓
Retrieve (main guideline)
↓
Rerank
↓
Evidence Judge
├─ sufficient → Answer
└─ insufficient
↓
Rewrite Query
↓
Retrieve (main + support)
↓
Rerank
↓
Judge
├─ sufficient → Answer
└─ insufficient → Refuse

```

---

# 主要入口

系统主入口：

```

src/rag_answer_siliconflow.py

````

运行方式：

```bash
python -m src.rag_answer_siliconflow "your question"
````

调用流程：

```
rag_answer_siliconflow
    ↓
agent_graph
    ↓
rag_core.retrieve
    ↓
rerank
    ↓
judge evidence
    ↓
generate answer
```

---

# 项目结构

```
course_rag_mvp/

src/                     # 核心代码
 ├ agent_graph.py        # Agent 决策逻辑
 ├ rag_core.py           # RAG 检索核心
 ├ ingest.py             # 文档解析
 ├ split_chunk.py        # 文档切块
 ├ build_index.py        # 向量索引构建
 └ rag_answer_siliconflow.py  # CLI入口

data/
 ├ raw_docs/             # 原始 PDF
 └ processed/            # chunk 数据

artifacts/
 ├ index_main/           # guideline 索引
 ├ index_support/        # evidence review 索引
 └ index/                # unified corpus 历史索引

eval/                    # 检索评测脚本

scripts/                 # 调试脚本

archive/
 └ experiments/          # 历史实验记录
```

---

# 核心模块说明

### rag_core.py

负责：

* embedding
* FAISS 检索
* context 构建
* LLM 调用

---

### agent_graph.py

Agent 控制逻辑：

* retrieve
* rerank
* evidence judge
* query rewrite
* answer / refuse

使用 **LangGraph** 构建。

---

### build_index.py

用于：

* 生成 embedding
* 构建 FAISS index

输出：

```
artifacts/index_main
artifacts/index_support
```

---

# 环境配置

推荐：

```
Python 3.11
```

创建环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

需要环境变量：

```bash
export SILICONFLOW_API_KEY="your_key"
```

可选：

```bash
export OPENAI_API_KEY="your_key"
export LOCAL_EMBED_MODEL="all-MiniLM-L6-v2"
```

---

# 数据构建流程

如果需要从原始文档重新生成数据：

```bash
python -m src.ingest
python -m src.split_chunk
python -m src.build_index --target main
python -m src.build_index --target support
```

生成：

```
artifacts/index_main
artifacts/index_support
```

---

# 检索评测

生成检索结果：

```bash
python eval/dump_retrieval_results.py
```

rerank：

```bash
python eval/rerank_cross_encoder.py
```

计算指标：

```bash
python eval/eval_retrieval.py
```

当前指标：

```
Hit@k
MRR@k
```

---

# 当前系统限制

当前版本仍存在一些问题：

1. evidence judge 依赖 LLM，稳定性有限
2. query rewrite 策略较简单
3. chunk 仍可能切断 guideline recommendation
4. evaluation dataset 较小

---

# 当前优化方向

未来主要优化方向：

### Retrieval

* 优化 chunk 策略
* 提高 guideline 命中率

### Rerank

* 尝试更强的 cross-encoder

### Agent

* 改进 evidence judge prompt
* 优化 query rewrite

### Evaluation

* 扩展 question dataset
* 增加 answer quality 评测

---

# 项目定位

该项目用于实验 **Agentic RAG Architecture**：

* guideline-first retrieval
* dual corpus retrieval
* evidence-aware reasoning
* query rewrite
* retrieval evaluation

适合场景：

* 医疗 guideline QA
* 法规问答
* 企业知识库

```
```
