import os
import json
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