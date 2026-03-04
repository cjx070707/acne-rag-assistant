import os
import sys
import json
from typing import List, Dict

# 先把仓库根目录加到 sys.path，再 import src.*
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

import numpy as np
import faiss

from src.build_index import get_embedder, l2_normalize

CHUNKS_PATH = os.path.join(REPO_ROOT, "data", "processed", "chunks_main.jsonl")
INDEX_DIR = os.path.join(REPO_ROOT, "index_main")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
ID_MAP_PATH = os.path.join(INDEX_DIR, "id_map.json")

QUESTIONS_PATH = os.path.join(THIS_DIR, "questions.jsonl")
OUT_PATH = os.path.join(THIS_DIR, "retrieval_results.jsonl")

TOPK = 3


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
    # load resources
    index = faiss.read_index(FAISS_INDEX_PATH)
    id_map = load_id_map(ID_MAP_PATH)
    chunks = load_chunks_lookup(CHUNKS_PATH)
    questions = load_jsonl(QUESTIONS_PATH)

    backend, embed_texts = get_embedder()
    print(f"[INFO] embedding backend = {backend}")
    print(f"[INFO] questions = {len(questions)}  topk = {TOPK}")

    # dump retrieval results
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for q in questions:
            qid = q["id"]
            query = q["question"]

            qvec = embed_texts([query]).astype(np.float32)   # (1, dim)
            qvec = l2_normalize(qvec).astype(np.float32)     # match IndexFlatIP cosine setup

            D, I = index.search(qvec, TOPK)                  # (1, TOPK)

            top = []
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
                        "score": float(score),
                    }
                )

            out.write(json.dumps({"id": qid, "topk": top}, ensure_ascii=False) + "\n")

    print(f"[OK] wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()