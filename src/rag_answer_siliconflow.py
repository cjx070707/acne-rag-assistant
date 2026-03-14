import json
import os
import argparse

from .agent_graph import run_agent_query
from .rag_core import DEFAULT_TOP_K, DEFAULT_PREFILTER_K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", type=str)
    ap.add_argument("--topk", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--prefilter-k", type=int, default=DEFAULT_PREFILTER_K)
    ap.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3.2")
    args = ap.parse_args()

    out = run_agent_query(
        args.question,
        topk=args.topk,
        prefilter_k=args.prefilter_k,
        model_name=args.model,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()