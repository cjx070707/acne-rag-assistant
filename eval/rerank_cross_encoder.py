import argparse
from typing import List, Dict, Any

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from src.config import RETRIEVAL_RESULTS_PATH, RERANKED_RESULTS_PATH
from eval.common import dump_jsonl, load_jsonl
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 32

def parse_args():
    parser = argparse.ArgumentParser(description="Rerank retrieval candidates with a cross-encoder.")
    parser.add_argument("--in-path", default=RETRIEVAL_RESULTS_PATH)
    parser.add_argument("--out-path", default=RERANKED_RESULTS_PATH)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()

def main():
    args = parse_args()
    data = load_jsonl(args.in_path)
    ce = CrossEncoder(args.model_name)

    out = []
    for ex in data:
        qid = ex["id"]
        qtext = ex.get("question")  # 如果你的 retrieval_results 没存 question，就从 questions.jsonl 补
        if not qtext:
            raise ValueError(f"{qid} missing 'question' field in retrieval results. Store it when dumping retrieval.")

        cands = ex["topk"]  # list of chunks
        pairs = [(qtext, c.get("text", "")) for c in cands]
        scores = ce.predict(pairs, batch_size=args.batch_size)

        # attach rerank score
        for c, s in zip(cands, scores):
            c["rerank_score"] = float(s)

        cands_sorted = sorted(cands, key=lambda x: x["rerank_score"], reverse=True)

        out.append({
            "id": qid,
            "question": qtext,
            "topk": cands_sorted
        })

    dump_jsonl(args.out_path, out)
    print("Wrote:", args.out_path)

if __name__ == "__main__":
    main()
