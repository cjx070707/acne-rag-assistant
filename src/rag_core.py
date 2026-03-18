import os
import json
import math
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss

LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")

DEFAULT_TOP_K = 3
DEFAULT_PREFILTER_K = 120
MAX_OUTPUT_TOKENS = 700

REFUSAL_TEXT = "资料不足：未在提供的文档片段中找到可靠依据。"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if not items:
        raise RuntimeError(f"{path} is empty")
    return items


def load_id_map_list(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing id_map: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("id_map.json must be a list (faiss_id -> chunk_id)")
    return data


def build_chunk_lookup(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        cid = c.get("chunk_id")
        if cid:
            lookup[cid] = c
    return lookup


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


_embedder = None
TOKEN_RE = re.compile(r"[a-z0-9]+")


def get_local_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(LOCAL_EMBED_MODEL)

    def embed(texts: List[str]) -> np.ndarray:
        vecs = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        vecs = np.array(vecs, dtype="float32")
        return l2_normalize(vecs).astype("float32")

    _embedder = embed
    return _embedder


def is_junk_chunk(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    low = t.lower()

    hard_bad = [
        "field content",
        "current review status",
        "completed but not published",
        "analysis of sub-groups",
    ]
    if any(p in low for p in hard_bad):
        return True

    if "evidence review" in low and "management options" in low and "page" in low:
        return True

    return False


def tokenize_for_lexical(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def build_lexical_stats(chunk_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    inverted: Dict[str, List[Tuple[str, int]]] = {}
    doc_lens: Dict[str, int] = {}

    for chunk_id, chunk in chunk_lookup.items():
        tokens = tokenize_for_lexical(chunk.get("text") or "")
        doc_lens[chunk_id] = len(tokens)
        if not tokens:
            continue

        tf_map: Dict[str, int] = {}
        for tok in tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1

        for tok, tf in tf_map.items():
            inverted.setdefault(tok, []).append((chunk_id, tf))

    total_docs = max(len(chunk_lookup), 1)
    avg_doc_len = sum(doc_lens.values()) / max(len(doc_lens), 1)
    return {
        "inverted": inverted,
        "doc_lens": doc_lens,
        "total_docs": total_docs,
        "avg_doc_len": avg_doc_len or 1.0,
    }


def lexical_search(
    question: str,
    lexical_stats: Dict[str, Any],
    chunk_lookup: Dict[str, Dict[str, Any]],
    top_k: int,
    doc_filter: Optional[str] = None,
    allow_junk_fallback: bool = True,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[Dict[str, Any], float]]:
    query_tokens = tokenize_for_lexical(question)
    if not query_tokens:
        return []

    inverted = lexical_stats["inverted"]
    doc_lens = lexical_stats["doc_lens"]
    total_docs = lexical_stats["total_docs"]
    avg_doc_len = lexical_stats["avg_doc_len"]

    scores: Dict[str, float] = {}
    for tok in query_tokens:
        postings = inverted.get(tok)
        if not postings:
            continue

        df = len(postings)
        idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
        for chunk_id, tf in postings:
            doc_len = doc_lens.get(chunk_id, 0) or 1
            denom = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
            score = idf * ((tf * (k1 + 1.0)) / max(denom, 1e-12))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    filtered: List[Tuple[Dict[str, Any], float]] = []
    unfiltered: List[Tuple[Dict[str, Any], float]] = []

    for chunk_id, score in ranked:
        item = chunk_lookup.get(chunk_id)
        if not item:
            continue

        pair = (item, float(score))
        unfiltered.append(pair)

        if doc_filter is not None and item.get("doc_id") != doc_filter:
            continue
        if is_junk_chunk(item.get("text") or ""):
            continue

        filtered.append(pair)
        if len(filtered) >= top_k:
            break

    if filtered or not allow_junk_fallback:
        return filtered[:top_k]
    return unfiltered[:top_k]


def reciprocal_rank_fuse(
    ranked_lists: List[List[Tuple[Dict[str, Any], float]]],
    *,
    top_k: int,
    rrf_k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    fused_scores: Dict[str, float] = {}
    fused_chunks: Dict[str, Dict[str, Any]] = {}

    for weight, ranked in zip(weights, ranked_lists):
        for rank, (chunk, _) in enumerate(ranked, start=1):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            fused_chunks[chunk_id] = chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (weight / (rrf_k + rank))

    ranked_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(fused_chunks[chunk_id], float(score)) for chunk_id, score in ranked_ids]


def build_context_and_sources(
    retrieved: List[Tuple[Dict[str, Any], float]],
    max_chars_per_chunk: int = 900,
) -> Tuple[str, List[Dict[str, Any]]]:
    blocks: List[str] = []
    sources: List[Dict[str, Any]] = []

    for rank, (c, score) in enumerate(retrieved, start=1):
        text = (c.get("text") or "").strip()
        text = " ".join(text.split())
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."

        blocks.append(
            f"[{rank}] doc_id={c.get('doc_id')} page={c.get('page')} score={score:.4f} title={c.get('title')}\n"
            f"{text}"
        )
        sources.append(
            {
                "rank": rank,
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "title": c.get("title"),
                "page": c.get("page"),
                "file_path": c.get("file_path"),
                "score": score,
            }
        )

    return "\n\n".join(blocks), sources


def extract_json(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s

    if s.startswith("```"):
        first_nl = s.find("\n")
        s2 = s[first_nl + 1:] if first_nl != -1 else ""
        end = s2.rfind("```")
        if end != -1:
            s2 = s2[:end]
        s = s2.strip()

    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        return s[l : r + 1].strip()
    return s


def _get_siliconflow_client():
    api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY not set.")

    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")


def call_llm_siliconflow_raw(
    *,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> str:
    client = _get_siliconflow_client()

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return (resp.choices[0].message.content or "").strip()


def call_llm_siliconflow(question: str, context: str, model_name: str) -> Dict[str, Any]:
    system = (
        "You are a medical RAG assistant.\n"
        "Answer strictly based on the provided context.\n"
        "Combine information from multiple context blocks if needed.\n"
        "Cite sources using bracket numbers like [1], [2].\n"
        'Return STRICT JSON: {"answer": "...", "citations":[1,2]}'
    )
    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nReturn JSON only."

    txt = call_llm_siliconflow_raw(
        system_prompt=system,
        user_prompt=user,
        model_name=model_name,
        temperature=0.0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    try:
        obj = json.loads(extract_json(txt))
        if not isinstance(obj, dict):
            raise ValueError("LLM JSON is not an object")
        obj.setdefault("answer", "")
        obj.setdefault("citations", [])
        return obj
    except Exception:
        return {"answer": txt or REFUSAL_TEXT, "citations": []}


def retrieve_topk_filtered(
    question: str,
    index: faiss.Index,
    id_map_list: List[str],
    chunk_lookup: Dict[str, Dict[str, Any]],
    top_k: int,
    prefilter_k: int,
    doc_filter: Optional[str] = None,
    allow_junk_fallback: bool = True,
) -> List[Tuple[Dict[str, Any], float]]:
    embed = get_local_embedder()
    qvec = embed([question])  # (1, d)

    if qvec.shape[1] != index.d:
        raise RuntimeError(
            f"Embedding dim mismatch: query dim={qvec.shape[1]} vs index dim={index.d}. "
            f"Index and embedding model must match."
        )

    pre_k = max(prefilter_k, top_k)
    D, I = index.search(qvec, pre_k)

    filtered: List[Tuple[Dict[str, Any], float]] = []
    unfiltered: List[Tuple[Dict[str, Any], float]] = []

    for score, fid in zip(D[0].tolist(), I[0].tolist()):
        if fid == -1:
            continue
        if fid < 0 or fid >= len(id_map_list):
            continue

        chunk_id = id_map_list[fid]
        item = chunk_lookup.get(chunk_id)
        if not item:
            continue

        unfiltered.append((item, float(score)))

        if doc_filter is not None and item.get("doc_id") != doc_filter:
            continue

        if is_junk_chunk(item.get("text") or ""):
            continue

        filtered.append((item, float(score)))
        if len(filtered) >= top_k:
            break

    if len(filtered) >= top_k:
        return filtered[:top_k]

    if not allow_junk_fallback:
        return filtered

    seen = {c.get("chunk_id") for c, _ in filtered}
    for item, score in unfiltered:
        if doc_filter is not None and item.get("doc_id") != doc_filter:
            continue
        if item.get("chunk_id") in seen:
            continue
        filtered.append((item, score))
        if len(filtered) >= top_k:
            break

    return filtered[:top_k]


def retrieve_topk_hybrid(
    question: str,
    index: faiss.Index,
    id_map_list: List[str],
    chunk_lookup: Dict[str, Dict[str, Any]],
    lexical_stats: Dict[str, Any],
    top_k: int,
    prefilter_k: int,
    doc_filter: Optional[str] = None,
    allow_junk_fallback: bool = True,
) -> List[Tuple[Dict[str, Any], float]]:
    candidate_k = max(prefilter_k, top_k)
    lexical_k = min(max(top_k * 3, top_k), 40)

    dense_items = retrieve_topk_filtered(
        question=question,
        index=index,
        id_map_list=id_map_list,
        chunk_lookup=chunk_lookup,
        top_k=candidate_k,
        prefilter_k=candidate_k,
        doc_filter=doc_filter,
        allow_junk_fallback=allow_junk_fallback,
    )
    lexical_items = lexical_search(
        question=question,
        lexical_stats=lexical_stats,
        chunk_lookup=chunk_lookup,
        top_k=lexical_k,
        doc_filter=doc_filter,
        allow_junk_fallback=allow_junk_fallback,
    )

    fused = reciprocal_rank_fuse(
        [dense_items, lexical_items],
        top_k=top_k,
        weights=[1.0, 0.3],
    )
    if fused:
        return fused
    return dense_items[:top_k]
