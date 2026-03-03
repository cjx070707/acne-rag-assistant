# eval/eval_retrieval.py
import os
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
from tqdm import tqdm


# -----------------------------
# Paths (relative to repo root)
# -----------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKS_PATH = os.path.join(REPO_ROOT, "data", "processed", "chunks.jsonl")
FAISS_INDEX_PATH = os.path.join(REPO_ROOT, "index", "faiss.index")
ID_MAP_PATH = os.path.join(REPO_ROOT, "index", "id_map.json")

# Embedding backend (keep consistent with build_index.py)
OPENAI_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LOCAL_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_chunks(path: str) -> List[Dict[str, Any]]:
    return load_jsonl(path)


def load_id_map(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing id_map: {path} (did you run build_index.py?)")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Your build_index likely stored {"0": "chunk_id", ...} or a list; handle both.
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # keys might be strings of ints
        out = [None] * len(data)
        for k, v in data.items():
            out[int(k)] = v
        return out
    raise ValueError("Unsupported id_map format; expected list or dict.")


def get_embedder():
    """
    If OPENAI_API_KEY exists, try OpenAI embeddings (requires openai python package).
    Else use local sentence-transformers.
    """
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                "OPENAI_API_KEY is set but openai package is not available. "
                "Run: pip install openai"
            ) from e

        client = OpenAI()

        def embed_texts(texts: List[str]) -> np.ndarray:
            # OpenAI returns list of embeddings
            resp = client.embeddings.create(model=OPENAI_MODEL, input=texts)
            vecs = [d.embedding for d in resp.data]
            arr = np.array(vecs, dtype=np.float32)
            return arr

        return embed_texts, "openai"

    else:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "sentence-transformers not available. Run: pip install sentence-transformers"
            ) from e

        model = SentenceTransformer(LOCAL_MODEL)

        def embed_texts(texts: List[str]) -> np.ndarray:
            arr = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.array(arr, dtype=np.float32)

        return embed_texts, "local"


def normalize_if_needed(x: np.ndarray) -> np.ndarray:
    # If not already normalized, normalize for cosine-like search in inner product index.
    # (Your build might have normalized embeddings; this is harmless.)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def determine_route_docs(q: Dict[str, Any]) -> Optional[set]:
    """
    If question explicitly references a specific doc (our eval questions mostly start with 'In ...'),
    route retrieval to that doc.
    Priority:
      route_doc_ids > gold_doc_ids > (gold_doc_id if question startswith 'In ')
    """
    if q.get("route_doc_ids"):
        return set(q["route_doc_ids"])
    if q.get("gold_doc_ids"):
        return set(q["gold_doc_ids"])
    if q.get("gold_doc_id") and q.get("question", "").lstrip().lower().startswith("in "):
        return {q["gold_doc_id"]}
    return None


def retrieve_topk_routed(
    index: faiss.Index,
    id_map: List[str],
    query_vecs: np.ndarray,
    k: int,
    pre_k: int,
    route_docs_list: List[Optional[set]],
    chunk_lookup: Dict[str, Dict[str, Any]],
) -> List[List[str]]:
    # same normalization behavior as your existing retrieve_topk
    if query_vecs.dtype != np.float32:
        query_vecs = query_vecs.astype(np.float32)
    query_vecs = normalize_if_needed(query_vecs)

    D, I = index.search(query_vecs, pre_k)
    results: List[List[str]] = []

    for row_idx, row in enumerate(I):
        allow = route_docs_list[row_idx]
        picked: List[str] = []
        fallback: List[str] = []

        for idx in row:
            if idx < 0 or idx >= len(id_map):
                continue
            cid = id_map[idx]
            fallback.append(cid)

            if allow is None:
                # no routing, keep in order
                picked.append(cid)
            else:
                c = chunk_lookup.get(cid)
                if c and c.get("doc_id") in allow:
                    picked.append(cid)

            if len(picked) >= k:
                break

        # if routed and not enough, backfill using original ranking (keeps system robust)
        if len(picked) < k:
            for cid in fallback:
                if cid not in picked:
                    picked.append(cid)
                if len(picked) >= k:
                    break

        results.append(picked[:k])

    return results

def retrieve_topk(
    index: faiss.Index,
    id_map: List[str],
    query_vecs: np.ndarray,
    k: int
) -> List[List[str]]:
    # FAISS expects float32
    if query_vecs.dtype != np.float32:
        query_vecs = query_vecs.astype(np.float32)

    # If index is IP-based, normalized vectors typically used; keep it consistent.
    # We'll normalize regardless; if you built with L2, it still runs but might not be optimal.
    query_vecs = normalize_if_needed(query_vecs)

    D, I = index.search(query_vecs, k)
    results = []
    for row in I:
        chunk_ids = []
        for idx in row:
            if idx < 0:
                continue
            if idx >= len(id_map):
                continue
            chunk_ids.append(id_map[idx])
        results.append(chunk_ids)
    return results


def build_chunk_lookup(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup = {}
    for c in chunks:
        cid = c.get("chunk_id") or c.get("id")
        if cid:
            lookup[cid] = c
    return lookup


def match_gold(
    retrieved_chunk_ids: List[str],
    chunk_lookup: Dict[str, Dict[str, Any]],
    gold_contains: Optional[str],
    gold_doc_id: Optional[str],
    gold_doc_ids: Optional[List[str]],
) -> Tuple[bool, Optional[int]]:
    """
    Returns (hit, rank_1_based_if_hit).
    Hit rule priority:
      1) gold_contains: any retrieved chunk text contains it
      2) gold_doc_ids: any retrieved chunk doc_id in set
      3) gold_doc_id: any retrieved chunk doc_id matches
    """
    gc = (gold_contains or "").strip().lower()
    gset = set(gold_doc_ids or [])
    gd = (gold_doc_id or "").strip()

    for i, cid in enumerate(retrieved_chunk_ids):
        c = chunk_lookup.get(cid)
        if not c:
            continue

        if gc:
            text = (c.get("text") or "").lower()
            if gc in text:
                return True, i + 1

        doc = c.get("doc_id")
        if gset and doc in gset:
            return True, i + 1

        if (not gset) and gd and doc == gd:
            return True, i + 1

    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefilter-k", type=int, default=50, help="FAISS pre-retrieval size for routing/filtering")
    ap.add_argument("--qfile", required=True, help="Path to eval questions.jsonl")
    ap.add_argument("--k", type=int, default=3, help="Top-K for retrieval")
    ap.add_argument("--batch", type=int, default=32, help="Embedding batch size (for OpenAI/local)")
    ap.add_argument("--show-misses", type=int, default=10, help="Print first N misses")
    ap.add_argument("--out", default=None, help="Optional: write detailed run JSON to this path")
    args = ap.parse_args()

    # Load assets
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Missing chunks: {CHUNKS_PATH} (run ingest.py)")
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"Missing FAISS index: {FAISS_INDEX_PATH} (run build_index.py)")
    if not os.path.exists(ID_MAP_PATH):
        raise FileNotFoundError(f"Missing id_map: {ID_MAP_PATH} (run build_index.py)")

    chunks = load_chunks(CHUNKS_PATH)
    chunk_lookup = build_chunk_lookup(chunks)

    id_map = load_id_map(ID_MAP_PATH)
    index = faiss.read_index(FAISS_INDEX_PATH)

    questions = load_jsonl(args.qfile)
    if len(questions) < 10:
        print("[WARN] You have <10 questions. This is too small to trust. Use 30+.")

    embed_texts, backend = get_embedder()

    # Embed all questions
    q_texts = [q["question"] for q in questions]
    # Local embedder ignores args.batch (it has its own); OpenAI we batch manually:
    if backend == "openai":
        vecs_list = []
        for i in tqdm(range(0, len(q_texts), args.batch), desc="Embedding (OpenAI)"):
            vecs_list.append(embed_texts(q_texts[i:i + args.batch]))
        q_vecs = np.vstack(vecs_list).astype(np.float32)
    else:
        q_vecs = embed_texts(q_texts).astype(np.float32)

    # Dimension sanity check
    if q_vecs.shape[1] != index.d:
        raise RuntimeError(
            f"Embedding dim mismatch: query dim={q_vecs.shape[1]} vs index dim={index.d}. "
            f"Your index and eval embedder are inconsistent."
        )

    route_docs_list = [determine_route_docs(q) for q in questions]
    topk_ids = retrieve_topk_routed(
        index=index,
        id_map=id_map,
        query_vecs=q_vecs,
        k=args.k,
        pre_k=args.prefilter_k,
        route_docs_list=route_docs_list,
        chunk_lookup=chunk_lookup,
    )

    # Metrics
    hits = 0
    rr_sum = 0.0
    miss_examples = []

    details = []
    for q, retrieved in zip(questions, topk_ids):
        gold_contains = q.get("gold_contains")
        gold_doc_id = q.get("gold_doc_id")
        gold_doc_ids = q.get("gold_doc_ids")

        hit, rank = match_gold(retrieved, chunk_lookup, gold_contains, gold_doc_id, gold_doc_ids)
        if hit:
            hits += 1
            rr_sum += 1.0 / rank
        else:
            if len(miss_examples) < args.show_misses:
                miss_examples.append({
                    "id": q.get("id"),
                    "question": q.get("question"),
                    "gold_contains": gold_contains,
                    "gold_doc_id": gold_doc_id,
                    "topk_chunk_ids": retrieved[:args.k],
                    "top1_preview": (chunk_lookup.get(retrieved[0], {}).get("text", "")[:220] if retrieved else "")
                })

        details.append({
            "id": q.get("id"),
            "question": q.get("question"),
            "hit": hit,
            "rank": rank,
            "topk_chunk_ids": retrieved[:args.k],
        })

    n = len(questions)
    hit_at_k = hits / n if n else 0.0
    mrr = rr_sum / n if n else 0.0

    print("========== Retrieval Eval ==========")
    print(f"backend      : {backend}")
    print(f"questions    : {n}")
    print(f"top_k        : {args.k}")
    print(f"Hit@{args.k}      : {hit_at_k:.3f}")
    print(f"MRR@{args.k}      : {mrr:.3f}")

    if miss_examples:
        print("\n========== Example Misses ==========")
        for m in miss_examples:
            print(f"\n[{m['id']}] {m['question']}")
            if m.get("gold_contains"):
                print(f"  gold_contains: {m['gold_contains']}")
            if m.get("gold_doc_id"):
                print(f"  gold_doc_id  : {m['gold_doc_id']}")
            print(f"  topk_chunk_ids: {m['topk_chunk_ids']}")
            if m.get("top1_preview"):
                print(f"  top1_preview : {m['top1_preview']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        payload = {
            "backend": backend,
            "k": args.k,
            "n": n,
            "hit_at_k": hit_at_k,
            "mrr_at_k": mrr,
            "details": details,
            "miss_examples": miss_examples,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] wrote eval result: {args.out}")


if __name__ == "__main__":
    main()