# src/query_retrieve.py
"""
Query-time retrieval probe (NO LLM).
Input: a query string
Output: top-k hits with score + doc/page + snippet (evidence excerpt)

Usage (from repo root):
  python src/query_retrieve.py --q "what is acne vulgaris" --k 5
  python src/query_retrieve.py --q "first-line treatment for mild acne" --k 5 --show-evidence

Notes:
- Embedding backend must match build_index.py (OpenAI vs local).
- If you built FAISS with normalized vectors + IndexFlatIP, then score ~= cosine similarity.
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import faiss


THIS_FILE = os.path.abspath(__file__)
THIS_DIR = os.path.dirname(THIS_FILE)

def detect_repo_root() -> str:
    """
    Works whether this script is in repo root or in src/.
    Rule:
      - if current file is under .../src, repo root = parent of src
      - else repo root = directory containing this file
    Also validates by checking required folders/files exist.
    """
    # candidate 1: if script is in src/
    if os.path.basename(THIS_DIR).lower() == "src":
        cand = os.path.dirname(THIS_DIR)
    else:
        cand = THIS_DIR·

    # minimal validation (avoid silent wrong paths)
    need_paths = [
        os.path.join(cand, "data"),
        os.path.join(cand, "index"),
    ]
    for p in need_paths:
        if not os.path.exists(p):
            # fallback: use current working directory (some people run from repo root)
            cwd = os.getcwd()
            if all(os.path.exists(os.path.join(cwd, x)) for x in ["data", "index"]):
                return cwd
            raise FileNotFoundError(
                f"Cannot detect repo root.\n"
                f"Script dir: {THIS_DIR}\n"
                f"Candidate: {cand}\n"
                f"CWD: {cwd}\n"
                f"Expected folders: data/ and index/ under repo root."
            )
    return cand

REPO_ROOT = detect_repo_root()

CHUNKS_PATH = os.path.join(REPO_ROOT, "data", "processed", "chunks.jsonl")
FAISS_INDEX_PATH = os.path.join(REPO_ROOT, "index", "faiss.index")
ID_MAP_PATH = os.path.join(REPO_ROOT, "index", "id_map.json")

print("[DEBUG] REPO_ROOT =", REPO_ROOT)
print("[DEBUG] FAISS_INDEX_PATH =", FAISS_INDEX_PATH)

# ---------- Embedding backend config (keep consistent with build_index.py) ----------
OPENAI_MODEL = "text-embedding-3-small"
LOCAL_MODEL = "all-MiniLM-L6-v2"


def _simple_tokenize_for_print(text: str, max_chars: int = 220) -> str:
    """Compact a long chunk into a one-line snippet."""
    t = " ".join(text.strip().split())
    return (t[:max_chars] + "…") if len(t) > max_chars else t


def load_chunks(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load chunks.jsonl into a dict keyed by chunk_id for fast lookup.
    Expect each line is a JSON dict with at least: chunk_id, text, doc_id, page.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path} (run ingest.py first)")

    chunks_by_id: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSONL at line {line_no}: {e}") from e

            chunk_id = obj.get("chunk_id")
            if not chunk_id:
                # If your schema uses another key, fix it here.
                raise ValueError(f"Chunk missing chunk_id at line {line_no}")

            chunks_by_id[chunk_id] = obj

    if not chunks_by_id:
        raise ValueError(f"No chunks loaded from {path}")
    return chunks_by_id


def load_id_map(path: str) -> List[str]:
    """
    Load id_map.json produced by build_index.py.

    Supported shapes:
      1) list: ["chunk_id0", "chunk_id1", ...]  (index position -> chunk_id)
      2) dict: {"0":"chunk_id0", "1":"chunk_id1", ...} (string/int keys)
    Returns a list where position i maps to chunk_id.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path} (run build_index.py first)")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # keys might be str "0" or int 0; normalize and rebuild a dense list
        items: List[Tuple[int, str]] = []
        for k, v in data.items():
            try:
                idx = int(k)
            except Exception:
                # if it’s not an int-keyed dict, it’s not an id_map we can use
                raise ValueError("id_map.json dict keys must be int-like (e.g., '0','1',...)")
            items.append((idx, v))

        items.sort(key=lambda x: x[0])
        max_idx = items[-1][0] if items else -1
        out = [None] * (max_idx + 1)  # type: ignore
        for idx, chunk_id in items:
            out[idx] = chunk_id  # type: ignore
        # quick sanity check
        if any(x is None for x in out):
            raise ValueError("id_map.json has missing indices (non-dense). Fix build_index output.")
        return out  # type: ignore

    raise ValueError("Unsupported id_map.json format (must be list or int-keyed dict).")


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Create embeddings for a list of texts, returning shape (n, dim) float32.
    Backend selection:
      - If OPENAI_API_KEY exists, try OpenAI embeddings
      - Else, use local sentence-transformers
    """
    use_openai = bool(os.getenv("OPENAI_API_KEY"))

    if use_openai:
        # OpenAI python SDK has multiple major versions; handle both patterns.
        try:
            from openai import OpenAI  # new style
            client = OpenAI()
            resp = client.embeddings.create(model=OPENAI_MODEL, input=texts)
            vecs = [item.embedding for item in resp.data]
            return np.array(vecs, dtype=np.float32)
        except Exception:
            # fallback to local if OpenAI call/import fails
            print("[WARN] OpenAI embedding failed/import error. Falling back to local embeddings.")

    # Local embedding
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers not installed. Install it OR set OPENAI_API_KEY for OpenAI embeddings."
        ) from e

    model = SentenceTransformer(LOCAL_MODEL)
    vecs = model.encode(texts, normalize_embeddings=False)  # we normalize ourselves for consistency
    return np.array(vecs, dtype=np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", "--query", dest="query", required=True, help="Query text")
    ap.add_argument("--k", type=int, default=5, help="top-k results")
    ap.add_argument("--show-evidence", action="store_true", help="Print evidence excerpt block (top1-2)")
    args = ap.parse_args()

    # 1) Load artifacts
    chunks_by_id = load_chunks(CHUNKS_PATH)
    id_map = load_id_map(ID_MAP_PATH)

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"Missing: {FAISS_INDEX_PATH} (run build_index.py first)")
    index = faiss.read_index(FAISS_INDEX_PATH)

    # 2) Embed query (must match build_index embedding space)
    q_vec = embed_texts([args.query])
    q_vec = l2_normalize(q_vec).astype(np.float32)

    # 3) Search
    k = max(1, args.k)
    D, I = index.search(q_vec, k)  # shapes: (1,k)
    scores = D[0].tolist()
    ids = I[0].tolist()

    # 4) Print results
    print(f"[Q] {args.query}")
    print(f"--- topk={k} backend=faiss ---")

    hits: List[Dict[str, Any]] = []
    for rank, (score, internal_id) in enumerate(zip(scores, ids), start=1):
        if internal_id < 0 or internal_id >= len(id_map):
            # -1 can appear if index returns empty; or id_map mismatch
            continue

        chunk_id = id_map[internal_id]
        chunk = chunks_by_id.get(chunk_id)
        if not chunk:
            continue

        doc_id = chunk.get("doc_id", "UNKNOWN_DOC")
        page = chunk.get("page", chunk.get("page_start", "UNKNOWN_PAGE"))
        text = chunk.get("text", "")

        snippet = _simple_tokenize_for_print(text, max_chars=220)

        print(f"{rank:>2} score={score:.4f} doc={doc_id} page={page}  \"{snippet}\"")

        hits.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "page": page,
                "text": text,
            }
        )

    # 5) Evidence-only “answer” (no LLM)
    if args.show_evidence and hits:
        print("\n=== EVIDENCE EXCERPT (no LLM) ===")
        take = hits[:2]
        for h in take:
            doc_id = h["doc_id"]
            page = h["page"]
            excerpt = h["text"].strip()
            # Keep excerpt reasonably short for terminal readability
            excerpt = excerpt[:1200] + ("…" if len(excerpt) > 1200 else "")
            print(f"\n[doc={doc_id} page={page} score={h['score']:.4f}]")
            print(excerpt)


if __name__ == "__main__":
    main()