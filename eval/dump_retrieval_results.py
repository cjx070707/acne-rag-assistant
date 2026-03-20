import os
import sys
import json
import argparse
from pathlib import Path

# 先把仓库根目录加到 sys.path，再 import src.*
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from src.config import QUESTIONS_PATH, RETRIEVAL_RESULTS_PATH
from src.retrieval_profiles import DEFAULT_EVAL_RETRIEVAL_PROFILE, resolve_retrieval_profile
from src.retrieval import retrieve_main
from eval.common import ensure_parent_dir, load_jsonl

TOPK = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Dump retrieval results for a question set.")
    parser.add_argument("--questions-path", default=QUESTIONS_PATH)
    parser.add_argument("--out-path", default=RETRIEVAL_RESULTS_PATH)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--retrieval-profile", default=DEFAULT_EVAL_RETRIEVAL_PROFILE)
    parser.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default=None)
    parser.add_argument("--metadata-filtering", action="store_true", default=None)
    parser.add_argument("--query-routing", action="store_true", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    questions = load_jsonl(args.questions_path)

    print(f"[INFO] questions = {len(questions)}  topk = {args.topk}")
    print(f"[INFO] questions_path = {args.questions_path}")
    print(f"[INFO] retrieval_mode = {args.retrieval_mode}")
    resolved = resolve_retrieval_profile(
        args.retrieval_profile,
        {
            "retrieval_mode": args.retrieval_mode,
            "metadata_filtering": args.metadata_filtering,
            "query_routing": args.query_routing,
            "apply_filtering": False if args.retrieval_mode == "dense" else None,
        },
    )
    print(f"[INFO] retrieval_profile = {resolved['retrieval_profile']}")

    ensure_parent_dir(args.out_path)
    with open(args.out_path, "w", encoding="utf-8") as out:
        for q in questions:
            top = retrieve_main(
                q["question"],
                {
                    "retrieval_profile": args.retrieval_profile,
                    "topk": args.topk,
                    "prefilter_k": max(args.topk, 120),
                    "retrieval_mode": args.retrieval_mode,
                    "apply_filtering": False if args.retrieval_mode == "dense" else None,
                    "metadata_filtering": args.metadata_filtering,
                    "question_type": q.get("question_type"),
                    "query_routing": args.query_routing,
                },
            )

            out.write(
                json.dumps(
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "question_type": q.get("question_type"),
                        "retrieval_profile": resolved["retrieval_profile"],
                        "retrieval_mode": resolved["retrieval_mode"],
                        "metadata_filtering": resolved["metadata_filtering"],
                        "query_routing": resolved["query_routing"],
                        "topk": top,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[OK] wrote: {args.out_path}")


if __name__ == "__main__":
    main()
