import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss

# ---------------- Paths (robust to CWD / F5) ----------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR) if os.path.basename(THIS_DIR).lower() == "src" else THIS_DIR

CHUNKS_PATH = os.path.join(REPO_ROOT, "data", "processed", "chunks.jsonl")
INDEX_DIR = os.path.join(REPO_ROOT, "index")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
ID_MAP_PATH = os.path.join(INDEX_DIR, "id_map.json")

DEFAULT_TOP_K = 4
MAX_OUTPUT_TOKENS = 500

LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"

REFUSAL_TEXT = "资料不足：未在提供的文档片段中找到可靠依据。"
SCORE_THRESH = 0.30  # 先用0.30，后面你再调


def load_chunks(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing chunks: {path} (run ingest.py first)")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if not items:
        raise RuntimeError("chunks.jsonl is empty")
    return items


def load_id_map_list(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing id_map: {path} (run build_index.py first)")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("id_map.json must be a list (faiss_id -> chunk_id)")
    return data


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def get_local_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(LOCAL_EMBED_MODEL)

    def embed(texts: List[str]) -> np.ndarray:
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        vecs = np.array(vecs, dtype="float32")
        return l2_normalize(vecs).astype("float32")  # FlatIP + cosine: normalize
    return embed


def build_chunk_lookup(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup = {}
    for c in chunks:
        cid = c.get("chunk_id")
        if cid:
            lookup[cid] = c
    return lookup


def retrieve(
    question: str,
    index: faiss.Index,
    id_map_list: List[str],          # faiss_id -> chunk_id
    chunk_lookup: Dict[str, Dict[str, Any]],
    top_k: int,
) -> List[Tuple[Dict[str, Any], float]]:
    embed = get_local_embedder()
    qvec = embed([question])  # (1, d) already normalized float32

    D, I = index.search(qvec, top_k)

    results = []
    for score, fid in zip(D[0].tolist(), I[0].tolist()):
        if fid == -1:
            continue
        if fid < 0 or fid >= len(id_map_list):
            continue
        chunk_id = id_map_list[fid]
        item = chunk_lookup.get(chunk_id)
        if not item:
            continue
        results.append((item, float(score)))
    return results


def build_context_and_sources(retrieved: List[Tuple[Dict[str, Any], float]]) -> Tuple[str, List[Dict[str, Any]]]:
    blocks = []
    sources = []
    for rank, (c, score) in enumerate(retrieved, start=1):
        blocks.append(
            f"[{rank}] doc_id={c.get('doc_id')} page={c.get('page')} score={score:.4f} title={c.get('title')}\n"
            f"{c.get('text','')}"
        )
        sources.append(
            {
                "rank": rank,
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "title": c.get("title"),
                "page": c.get("page"),
                "file_path": c.get("file_path"),
            }
        )
    return "\n\n".join(blocks), sources


def call_llm_siliconflow(question: str, context: str) -> Dict[str, Any]:
    api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY not set.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

    system = (
        "You are a RAG assistant. Use ONLY the provided context.\n"
        "If the context is insufficient, say you don't know.\n"
        "Return STRICT JSON with keys:\n"
        '  - answer: string\n'
        '  - citations: array of integers referencing the bracket ids like [1],[2]\n'
        'Example: {"answer":"...","citations":[1,3]}'
    )
    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nReturn JSON only."

    resp = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3.2",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    txt = resp.choices[0].message.content.strip()
    try:
        return json.loads(txt)
    except Exception:
        return {"answer": txt, "citations": []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(ID_MAP_PATH):
        raise FileNotFoundError("Missing index files. Run build_index.py to create index/faiss.index and index/id_map.json")

    chunks = load_chunks(CHUNKS_PATH)
    chunk_lookup = build_chunk_lookup(chunks)
    id_map_list = load_id_map_list(ID_MAP_PATH)
    index = faiss.read_index(FAISS_INDEX_PATH)

    retrieved = retrieve(args.question, index, id_map_list, chunk_lookup, args.topk)

    if not retrieved:
        print(json.dumps({"answer": REFUSAL_TEXT, "sources": []}, ensure_ascii=False, indent=2))
        return

    # refusal gate
    top1_score = retrieved[0][1]
    if top1_score < SCORE_THRESH:
        print(json.dumps({"answer": REFUSAL_TEXT, "sources": []}, ensure_ascii=False, indent=2))
        return

    context, sources_meta = build_context_and_sources(retrieved)

    if args.debug:
        print("=== RETRIEVED ===")
        for i, (c, s) in enumerate(retrieved, 1):
            print(i, c.get("doc_id"), c.get("page"), f"{s:.4f}", c.get("chunk_id"))
        print("\n=== CONTEXT (first 800 chars) ===")
        print(context[:800])
        print("\n=== LLM ===")

    llm_out = call_llm_siliconflow(args.question, context)

    cited = llm_out.get("citations", [])
    resolved_sources = []
    for cid in cited:
        if isinstance(cid, int) and 1 <= cid <= len(sources_meta):
            resolved_sources.append(sources_meta[cid - 1])

    final = {"answer": llm_out.get("answer", ""), "sources": resolved_sources}
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()