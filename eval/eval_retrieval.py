import os
import argparse

import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from src.config import QUESTIONS_PATH, RETRIEVAL_RESULTS_PATH, RERANKED_RESULTS_PATH
from eval.common import load_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval results against gold doc/page/rec ids.")
    parser.add_argument("--questions-path", default=QUESTIONS_PATH)
    parser.add_argument("--results-path", default=RERANKED_RESULTS_PATH)
    parser.add_argument("--baseline-path", default=RETRIEVAL_RESULTS_PATH)
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    questions = {q["id"]: q for q in load_jsonl(args.questions_path)}
    results = load_jsonl(args.results_path)

    if not results:
        raise RuntimeError("retrieval results are empty (run dump_retrieval_results.py first)")

    page_hit = 0
    page_mrr = 0.0
    rec_hit = 0
    rec_mrr = 0.0
    rec_total = 0

    for r in results:
        qid = r["id"]
        gold = questions[qid]
        gold_doc = gold["gold_doc_id"]
        gold_pages = set(gold["gold_pages"])
        gold_rec_ids = set(gold.get("gold_rec_ids", []))

        page_rr = 0.0
        rec_rr = 0.0
        for rank, cand in enumerate(r.get("topk", [])[: args.k], start=1):
            if cand.get("doc_id") == gold_doc and cand.get("page") in gold_pages:
                page_hit += 1
                page_rr = 1.0 / rank
                break
        page_mrr += page_rr

        if gold_rec_ids:
            rec_total += 1
            for rank, cand in enumerate(r.get("topk", [])[: args.k], start=1):
                if cand.get("doc_id") == gold_doc and cand.get("rec_id") in gold_rec_ids:
                    rec_hit += 1
                    rec_rr = 1.0 / rank
                    break
            rec_mrr += rec_rr

    n = len(results)
    print(f"Page Hit@{args.k}: {page_hit/n:.3f}")
    print(f"Page MRR@{args.k}: {page_mrr/n:.3f}")
    if rec_total:
        print(f"Rec Hit@{args.k}: {rec_hit/rec_total:.3f}")
        print(f"Rec MRR@{args.k}: {rec_mrr/rec_total:.3f}")
    print("QUESTIONS_PATH:", args.questions_path)
    print("RESULTS_PATH:", args.results_path)
    print("RESULTS exists:", os.path.exists(args.results_path), "size:", os.path.getsize(args.results_path))
    print("first topk len:", len(results[0].get("topk", [])))
    baseline = load_jsonl(args.baseline_path)
    reranked = load_jsonl(args.results_path)

    for i in range(min(3, len(results), len(baseline), len(reranked))):
        b = [c["chunk_id"] for c in baseline[i]["topk"][: args.k]]
        r = [c["chunk_id"] for c in reranked[i]["topk"][: args.k]]
        print(baseline[i]["id"], "baseline:", b, "reranked:", r)


if __name__ == "__main__":
    main()
