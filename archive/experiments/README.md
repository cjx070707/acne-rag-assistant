# Experiment Summary

这些实验目录保存的是历史检索优化记录，不参与当前在线主链路。

## Experiment Timeline

- `exp_001_bad_chunk`
  - 早期 chunk 策略对照实验
  - `metrics.jsonl` 中记录的是小规模问题集结果

- `exp_002_guideline_chunk`
  - 引入 guideline recommendation-level chunking
  - 指标记录为 `Hit@3 = 0.833`，`MRR@3 = 0.678`

- `exp_003_top30`
  - 扩大 retrieval candidate 范围
  - 指标记录为 `Hit@20 = 0.967`，`MRR@20 = 0.693`

- `exp_004_rerank`
  - 在 recommendation chunking 基础上加入 rerank
  - 当前保留结果里表现最好
  - `metrics.jsonl` 记录为 `Hit@k = 0.933`，`MRR = 0.822`
  - 同文件同时保留 rerank 前对照：`Hit@3 = 0.833`，`MRR@3 = 0.678`

## How To Read These Folders

- `questions.jsonl`：该实验所用问题集
- `retrieval_results.jsonl`：该实验保存的检索结果
- `metrics.jsonl`：该实验记录的核心指标

这些目录更适合作为研究轨迹和答辩材料，而不是运行时依赖。
