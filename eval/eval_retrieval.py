# save as: eval/eval_retrieval.py
import os
import json
from typing import List, Dict

K = 3

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_PATH = os.path.join(THIS_DIR, "questions.jsonl")
RESULTS_PATH = os.path.join(THIS_DIR, "artifacts", "reranked_results_top20.jsonl")


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
        raise RuntimeError("retrieval results are empty (run dump_retrieval_results.py first)")

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
    print("QUESTIONS_PATH:", QUESTIONS_PATH)
    print("RESULTS_PATH:", RESULTS_PATH)
    print("RESULTS exists:", os.path.exists(RESULTS_PATH), "size:", os.path.getsize(RESULTS_PATH))
    print("first topk len:", len(results[0].get("topk", [])))
    BASELINE_PATH = os.path.join(THIS_DIR, "artifacts", "retrieval_results.jsonl")

    baseline = load_jsonl(BASELINE_PATH)
    reranked = load_jsonl(RESULTS_PATH)

    for i in range(3):
        b = [c["chunk_id"] for c in baseline[i]["topk"][:3]]
        r = [c["chunk_id"] for c in reranked[i]["topk"][:3]]
        print(baseline[i]["id"], "baseline:", b, "reranked:", r)
if __name__ == "__main__":
    main()
