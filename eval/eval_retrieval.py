# save as: eval/eval_retrieval.py
import os
import json
from typing import List, Dict

K = 3

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_PATH = os.path.join(THIS_DIR, "questions.jsonl")
RESULTS_PATH = os.path.join(THIS_DIR, "retrieval_results.jsonl")


def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {e}") from e
    return items


def main():
    questions = {q["id"]: q for q in load_jsonl(QUESTIONS_PATH)}
    results = load_jsonl(RESULTS_PATH)

    if not results:
        raise RuntimeError("retrieval_results.jsonl is empty (run dump_retrieval_results.py first)")

    hit = 0
    mrr = 0.0

    for r in results:
        qid = r["id"]
        gold = questions[qid]
        gold_doc = gold["gold_doc_id"]
        gold_pages = set(gold["gold_pages"])

        rr = 0.0
        for rank, cand in enumerate(r.get("topk", [])[:K], start=1):
            if cand.get("doc_id") == gold_doc and cand.get("page") in gold_pages:
                hit += 1
                rr = 1.0 / rank
                break
        mrr += rr

    n = len(results)
    print(f"Hit@{K}: {hit/n:.3f}")
    print(f"MRR@{K}: {mrr/n:.3f}")


if __name__ == "__main__":
    main()