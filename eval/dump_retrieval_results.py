import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

# 先把仓库根目录加到 sys.path，再 import src.*
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

import numpy as np
import faiss

from src.config import CHUNKS_MAIN_PATH, INDEX_MAIN_DIR, QUESTIONS_PATH, RETRIEVAL_RESULTS_PATH
from src.build_index import get_embedder, l2_normalize
from eval.common import ensure_parent_dir, load_jsonl
from src.rag_core import build_lexical_stats, retrieve_topk_hybrid

TOPK = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Dump dense retrieval results for a question set.")
    parser.add_argument("--questions-path", default=QUESTIONS_PATH)
    parser.add_argument("--out-path", default=RETRIEVAL_RESULTS_PATH)
    parser.add_argument("--chunks-path", default=CHUNKS_MAIN_PATH)
    parser.add_argument("--index-dir", default=INDEX_MAIN_DIR)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default="dense")
    return parser.parse_args()


def load_chunks_lookup(path: str) -> Dict[str, Dict]:
    lookup = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid = c.get("chunk_id") or c.get("id")
            if not cid:
                raise ValueError(f"Missing chunk_id at chunks.jsonl:{line_no}")
            lookup[cid] = c
    if not lookup:
        raise RuntimeError("chunks.jsonl is empty")
    return lookup


def load_id_map(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list) or not obj:
        raise ValueError("id_map.json must be a non-empty list of chunk_id")
    return obj


def main():
    args = parse_args()
    faiss_index_path = os.path.join(args.index_dir, "faiss.index")
    id_map_path = os.path.join(args.index_dir, "id_map.json")

    # load resources
    index = faiss.read_index(faiss_index_path)
    id_map = load_id_map(id_map_path)
    chunks = load_chunks_lookup(args.chunks_path)
    lexical_stats = build_lexical_stats(chunks)
    questions = load_jsonl(args.questions_path)

    backend, embed_texts = get_embedder()
    print(f"[INFO] embedding backend = {backend}")
    print(f"[INFO] questions = {len(questions)}  topk = {args.topk}")
    print(f"[INFO] questions_path = {args.questions_path}")
    print(f"[INFO] retrieval_mode = {args.retrieval_mode}")

    # dump retrieval results
    ensure_parent_dir(args.out_path)
    with open(args.out_path, "w", encoding="utf-8") as out:
        for q in questions:
            qid = q["id"]
            query = q["question"]

            top = []
            if args.retrieval_mode == "hybrid":
                fused = retrieve_topk_hybrid(
                    question=query,
                    index=index,
                    id_map_list=id_map,
                    chunk_lookup=chunks,
                    lexical_stats=lexical_stats,
                    top_k=args.topk,
                    prefilter_k=max(args.topk, 120),
                    doc_filter=None,
                    allow_junk_fallback=True,
                )
                for chunk, score in fused:
                    top.append(
                        {
                            "chunk_id": chunk.get("chunk_id"),
                            "doc_id": chunk.get("doc_id"),
                            "page": chunk.get("page"),
                            "rec_id": chunk.get("rec_id"),
                            "score": float(score),
                            "text": chunk.get("text", ""),
                        }
                    )
            else:
                qvec = embed_texts([query]).astype(np.float32)
                qvec = l2_normalize(qvec).astype(np.float32)

                D, I = index.search(qvec, args.topk)
                for score, pos in zip(D[0].tolist(), I[0].tolist()):
                    if pos < 0:
                        continue
                    if pos >= len(id_map):
                        continue
                    chunk_id = id_map[pos]
                    c = chunks.get(chunk_id, {})
                    top.append(
                        {
                            "chunk_id": chunk_id,
                            "doc_id": c.get("doc_id"),
                            "page": c.get("page"),
                            "rec_id": c.get("rec_id"),
                            "score": float(score),
                            "text": c["text"]
                        }
                    )


            out.write(
                json.dumps(
                    {
                        "id": qid,
                        "question": query,
                        "question_type": q.get("question_type"),
                        "retrieval_mode": args.retrieval_mode,
                        "topk": top
                    },
                    ensure_ascii=False
                )
                + "\n"
            )
    print(f"[OK] wrote: {args.out_path}")


if __name__ == "__main__":
    main()
