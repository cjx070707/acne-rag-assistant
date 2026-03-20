import argparse
import os
import sys
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from eval.common import dump_jsonl, load_jsonl, safe_ratio
from src.agent_graph import run_agent_query
from src.retrieval_profiles import DEFAULT_RUNTIME_RETRIEVAL_PROFILE


DEFAULT_DATASET = str(Path("eval/datasets/refusal_boundary_v1.jsonl"))
DEFAULT_OUT = str(Path("eval/artifacts/refusal_report.jsonl"))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate refusal behavior on boundary questions.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--out-path", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--prefilter-k", type=int, default=120)
    parser.add_argument("--retrieval-profile", default=DEFAULT_RUNTIME_RETRIEVAL_PROFILE)
    parser.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default=None)
    parser.add_argument("--query-routing", action="store_true", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_jsonl(args.dataset_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    reports = []
    expected_refuse = 0
    actual_refuse = 0
    correct_refuse = 0

    for row in rows:
        result = run_agent_query(
            row["question"],
            topk=args.topk,
            prefilter_k=args.prefilter_k,
            model_name=args.model,
            retrieval_profile=args.retrieval_profile,
            retrieval_mode=args.retrieval_mode,
            query_routing=args.query_routing,
        )
        exp_refuse = row.get("expected_action") == "refuse"
        act_refuse = result.get("action") == "refuse"

        if exp_refuse:
            expected_refuse += 1
        if act_refuse:
            actual_refuse += 1
        if exp_refuse == act_refuse:
            correct_refuse += 1

        reports.append(
            {
                "id": row["id"],
                "question": row["question"],
                "expected_action": row["expected_action"],
                "actual_action": result.get("action"),
                "refusal_reason_type": row.get("refusal_reason_type"),
                "answer": result.get("answer"),
                "sources": result.get("sources", []),
            }
        )

    dump_jsonl(args.out_path, reports)
    total = len(rows)
    print(f"Expected refuses: {expected_refuse}/{total}")
    print(f"Actual refuses: {actual_refuse}/{total}")
    print(f"Refusal accuracy: {correct_refuse}/{total} = {safe_ratio(correct_refuse, total):.3f}")
    print(f"Wrote: {args.out_path}")


if __name__ == "__main__":
    main()
