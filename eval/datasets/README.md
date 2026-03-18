# Evaluation Datasets

这个目录保存面向后续 RAG 优化的分层评测集。

## Dataset Layers

- `retrieval_core_v2.jsonl`
  - 用于检索评测
  - 保留现有 guideline 命中问题，并补充了题型、难度、推荐 action、primary source 约束

- `qa_grounded_v1.jsonl`
  - 用于 grounded QA 评测
  - 每题包含 `reference_answer`、`must_include_points`、`must_not_claim`

- `support_governance_v1.jsonl`
  - 用于检验 support corpus 的使用边界
  - 关注何时只能 guideline 回答、何时 support 可以补充、何时 support 不能替代 recommendation

- `refusal_boundary_v1.jsonl`
  - 用于拒答评测
  - 覆盖越界问题、个体化用药、文档外问题和缺乏证据问题

## Usage Suggestion

- 检索优化：先跑 `retrieval_core_v2.jsonl`
- RAG 回答与 citation：跑 `qa_grounded_v1.jsonl`
- evidence policy / source priority：跑 `support_governance_v1.jsonl`
- judge / refusal：跑 `refusal_boundary_v1.jsonl`

## Example Commands

检索评测：

```bash
python eval/dump_retrieval_results.py \
  --questions-path eval/datasets/retrieval_core_v2.jsonl \
  --out-path eval/artifacts/retrieval_core_v2.results.jsonl

python eval/rerank_cross_encoder.py \
  --in-path eval/artifacts/retrieval_core_v2.results.jsonl \
  --out-path eval/artifacts/retrieval_core_v2.reranked.jsonl

python eval/eval_retrieval.py \
  --questions-path eval/datasets/retrieval_core_v2.jsonl \
  --baseline-path eval/artifacts/retrieval_core_v2.results.jsonl \
  --results-path eval/artifacts/retrieval_core_v2.reranked.jsonl
```

grounded QA 评测：

```bash
python eval/eval_grounded_qa.py \
  --dataset-path eval/datasets/qa_grounded_v1.jsonl \
  --out-path eval/artifacts/qa_grounded_report.jsonl
```

support 使用边界评测：

```bash
python eval/eval_grounded_qa.py \
  --dataset-path eval/datasets/support_governance_v1.jsonl \
  --out-path eval/artifacts/support_governance_report.jsonl
```

拒答评测：

```bash
python eval/eval_refusal.py \
  --dataset-path eval/datasets/refusal_boundary_v1.jsonl \
  --out-path eval/artifacts/refusal_report.jsonl
```

## Notes

- 这些数据集的目标是服务后续优化实验，不会自动替换现有 `eval/questions.jsonl`
- 当前 gold 仍以 guideline page 和 rec_id 为主，方便和现有索引结构对齐
- `eval_grounded_qa.py` 和 `eval_refusal.py` 需要可用的 `SILICONFLOW_API_KEY`
