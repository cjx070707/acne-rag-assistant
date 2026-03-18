# Final Benchmark v1

这是面向后续正式实验的稳定 benchmark 版本。

## Design Goals

- retrieval 不再只覆盖一小部分问题，而是尽量覆盖 guideline recommendation 主干
- grounded QA 覆盖 recommendation、referral、pregnancy、maintenance、isotretinoin、PCOS、scarring 等关键场景
- support governance 专门测试 support 文档何时只能补充、不能越权
- refusal boundary 专门测试越界、个体化建议和缺乏证据场景

## Dataset Sizes

- `retrieval_core_final_v1`: 56 questions
- `qa_grounded_final_v1`: 24 questions
- `support_governance_final_v1`: 12 questions
- `refusal_boundary_final_v1`: 20 questions

## Intended Usage

后续所有新优化实验，优先与这套 benchmark 对比。
如果未来要继续扩题，建议新增 `final_v2`，不要覆盖 `final_v1`。
