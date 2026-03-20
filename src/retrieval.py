from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import faiss

from .config import CHUNKS_MAIN_PATH, CHUNKS_SUPPORT_PATH, INDEX_MAIN_DIR, INDEX_SUPPORT_DIR
from .retrieval_profiles import DEFAULT_RUNTIME_RETRIEVAL_PROFILE, resolve_retrieval_profile
from .rag_core import (
    build_chunk_lookup,
    build_context_and_sources,
    build_lexical_stats,
    get_local_embedder,
    load_id_map_list,
    load_jsonl,
    retrieve_topk_filtered,
    retrieve_topk_hybrid,
)


DEFAULT_RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "dense").strip().lower() or "dense"


class RetrievalConfig(TypedDict, total=False):
    retrieval_profile: str
    retrieval_mode: str
    topk: int
    prefilter_k: int
    doc_filter: Optional[str]
    allow_junk_fallback: bool
    apply_filtering: bool
    metadata_filtering: bool
    question_type: Optional[str]
    query_routing: bool


class Candidate(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    page: Optional[int]
    rec_id: Optional[str]
    title: Optional[str]
    file_path: Optional[str]
    text: str
    score: float
    source_type: str
    rank: int
    retrieval_mode: str
    retrieval_profile: str


class CorpusResources(TypedDict):
    kind: str
    index: faiss.Index
    id_map_list: List[str]
    chunk_lookup: Dict[str, Dict[str, Any]]
    lexical_stats: Dict[str, Any]


RESOURCE_CACHE: Dict[str, CorpusResources] = {}


class RoutingDecision(TypedDict):
    question_type: str
    metadata_filtering: bool
    use_support: bool


def load_resources(kind: str) -> CorpusResources:
    if kind in RESOURCE_CACHE:
        return RESOURCE_CACHE[kind]

    if kind == "main":
        index_dir = INDEX_MAIN_DIR
        chunks_path = CHUNKS_MAIN_PATH
    elif kind == "support":
        index_dir = INDEX_SUPPORT_DIR
        chunks_path = CHUNKS_SUPPORT_PATH
    else:
        raise ValueError(f"Unknown corpus kind: {kind}")

    faiss_path = os.path.join(index_dir, "faiss.index")
    idmap_path = os.path.join(index_dir, "id_map.json")

    if not (os.path.exists(faiss_path) and os.path.exists(idmap_path)):
        raise FileNotFoundError(f"Missing index files in: {index_dir} (faiss.index / id_map.json)")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    chunks = load_jsonl(chunks_path)
    chunk_lookup = build_chunk_lookup(chunks)
    lexical_stats = build_lexical_stats(chunk_lookup)

    resource: CorpusResources = {
        "kind": kind,
        "index": faiss.read_index(faiss_path),
        "id_map_list": load_id_map_list(idmap_path),
        "chunk_lookup": chunk_lookup,
        "lexical_stats": lexical_stats,
    }
    RESOURCE_CACHE[kind] = resource
    return resource


def _normalized_config(config: Optional[RetrievalConfig] = None) -> RetrievalConfig:
    raw_cfg = dict(config or {})
    profile_name = raw_cfg.get("retrieval_profile")
    cfg = resolve_retrieval_profile(profile_name, raw_cfg) if profile_name else raw_cfg
    return {
        "retrieval_profile": str(cfg.get("retrieval_profile", DEFAULT_RUNTIME_RETRIEVAL_PROFILE)),
        "retrieval_mode": str(cfg.get("retrieval_mode", DEFAULT_RETRIEVAL_MODE)).lower(),
        "topk": int(cfg.get("topk", 6)),
        "prefilter_k": int(cfg.get("prefilter_k", 120)),
        "doc_filter": cfg.get("doc_filter"),
        "allow_junk_fallback": bool(cfg.get("allow_junk_fallback", True)),
        "apply_filtering": bool(cfg.get("apply_filtering", True)),
        "metadata_filtering": bool(cfg.get("metadata_filtering", False)),
        "question_type": cfg.get("question_type"),
        "query_routing": bool(cfg.get("query_routing", False)),
    }


def _rec_section_prefix(rec_id: Optional[str]) -> str:
    if not rec_id:
        return ""
    parts = str(rec_id).split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return str(rec_id)


def _infer_question_type(query: str) -> str:
    text = (query or "").lower()
    if any(k in text for k in ["refer", "referral", "dermatologist", "gpwer", "same-day"]):
        return "referral"
    if any(k in text for k in ["maintenance", "relapse", "review after", "review maintenance"]):
        return "maintenance"
    if any(k in text for k in ["pregnan", "isotretinoin", "mental health", "monitor", "contraindication", "precaution"]):
        return "precautions"
    if any(k in text for k in ["skin care", "cleanser", "sunscreen", "make-up", "makeup", "pick", "scratch", "diet"]):
        return "lifestyle"
    if any(k in text for k in ["first-line", "first line", "oral antibiotic", "escalat", "severe acne", "moderate to severe"]):
        return "treatment_escalation"
    return "recommendation"


def _preferred_sections(question_type: str, query: str) -> List[str]:
    mapping = {
        "recommendation": ["1.1", "1.5", "1.8"],
        "lifestyle": ["1.2", "1.3"],
        "referral": ["1.4"],
        "treatment_escalation": ["1.5", "1.6"],
        "maintenance": ["1.7", "1.6"],
        "precautions": ["1.5"],
    }
    prefs = list(mapping.get(question_type, []))
    text = (query or "").lower()
    if "scar" in text and "1.8" not in prefs:
        prefs.insert(0, "1.8")
    if "diet" in text and "1.3" not in prefs:
        prefs.insert(0, "1.3")
    if any(k in text for k in ["cleanser", "sunscreen", "make-up", "makeup", "pick", "scratch"]) and "1.2" not in prefs:
        prefs.insert(0, "1.2")
    if "refer" in text and "1.4" not in prefs:
        prefs.insert(0, "1.4")
    return prefs


def route_query(query: str, *, question_type: Optional[str] = None, attempt: int = 1) -> RoutingDecision:
    qtype = (question_type or "").strip().lower() or _infer_question_type(query)

    use_support = False
    if attempt >= 2:
        if qtype in {"treatment_escalation", "maintenance", "precautions"}:
            use_support = True

    return {
        "question_type": qtype,
        "metadata_filtering": True,
        "use_support": use_support,
    }


def _apply_metadata_filtering(
    candidates: List[Candidate],
    *,
    query: str,
    question_type: Optional[str],
    topk: int,
) -> List[Candidate]:
    qtype = (question_type or "").strip().lower() or _infer_question_type(query)
    preferred = _preferred_sections(qtype, query)
    if not preferred:
        return candidates[:topk]

    scored: List[Tuple[float, Candidate]] = []
    for idx, candidate in enumerate(candidates):
        base = float(candidate.get("score", 0.0))
        rec_prefix = _rec_section_prefix(candidate.get("rec_id"))
        bonus = 0.0

        if candidate.get("source_type") == "main" and rec_prefix in preferred:
            bonus += 0.10
        elif candidate.get("source_type") == "main" and rec_prefix:
            bonus -= 0.015

        text = (query or "").lower()
        cand_text = (candidate.get("text") or "").lower()
        if "scar" in text and "scar" in cand_text:
            bonus += 0.05
        if any(k in text for k in ["refer", "referral"]) and "refer" in cand_text:
            bonus += 0.04
        if "diet" in text and "diet" in cand_text:
            bonus += 0.04
        if "isotretinoin" in text and "isotretinoin" in cand_text:
            bonus += 0.04

        scored.append((base + bonus - (idx * 1e-6), candidate))

    reranked = [candidate for _, candidate in sorted(scored, key=lambda x: x[0], reverse=True)[:topk]]
    for rank, candidate in enumerate(reranked, start=1):
        candidate["rank"] = rank
    return reranked


def _to_candidates(
    items: List[Tuple[Dict[str, Any], float]],
    *,
    source_type: str,
    retrieval_mode: str,
    retrieval_profile: str,
) -> List[Candidate]:
    out: List[Candidate] = []
    for rank, (chunk, score) in enumerate(items, start=1):
        out.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "page": chunk.get("page"),
                "rec_id": chunk.get("rec_id"),
                "title": chunk.get("title"),
                "file_path": chunk.get("file_path"),
                "text": chunk.get("text", ""),
                "score": float(score),
                "source_type": source_type,
                "rank": rank,
                "retrieval_mode": retrieval_mode,
                "retrieval_profile": retrieval_profile,
            }
        )
    return out


def retrieve(
    query: str,
    resources: CorpusResources,
    config: Optional[RetrievalConfig] = None,
) -> List[Candidate]:
    cfg = _normalized_config(config)
    retrieval_profile = str(cfg["retrieval_profile"])
    retrieval_mode = str(cfg["retrieval_mode"])
    top_k = int(cfg["topk"])
    metadata_filtering = bool(cfg["metadata_filtering"])
    question_type = cfg.get("question_type")
    if bool(cfg["query_routing"]):
        decision = route_query(query, question_type=question_type, attempt=1)
        metadata_filtering = bool(decision["metadata_filtering"])
        question_type = decision["question_type"]
    requested_top_k = max(top_k, min(int(cfg["prefilter_k"]), 40)) if metadata_filtering else top_k

    if retrieval_mode == "hybrid":
        items = retrieve_topk_hybrid(
            question=query,
            index=resources["index"],
            id_map_list=resources["id_map_list"],
            chunk_lookup=resources["chunk_lookup"],
            lexical_stats=resources["lexical_stats"],
            top_k=requested_top_k,
            prefilter_k=int(cfg["prefilter_k"]),
            doc_filter=cfg.get("doc_filter"),
            allow_junk_fallback=bool(cfg["allow_junk_fallback"]),
        )
    elif not bool(cfg["apply_filtering"]):
        embed = get_local_embedder()
        qvec = embed([query])
        distances, indices = resources["index"].search(qvec, requested_top_k)
        raw_items: List[Tuple[Dict[str, Any], float]] = []
        for score, fid in zip(distances[0].tolist(), indices[0].tolist()):
            if fid == -1:
                continue
            if fid < 0 or fid >= len(resources["id_map_list"]):
                continue
            chunk_id = resources["id_map_list"][fid]
            item = resources["chunk_lookup"].get(chunk_id)
            if not item:
                continue
            raw_items.append((item, float(score)))
        items = raw_items
    else:
        items = retrieve_topk_filtered(
            question=query,
            index=resources["index"],
            id_map_list=resources["id_map_list"],
            chunk_lookup=resources["chunk_lookup"],
            top_k=requested_top_k,
            prefilter_k=int(cfg["prefilter_k"]),
            doc_filter=cfg.get("doc_filter"),
            allow_junk_fallback=bool(cfg["allow_junk_fallback"]),
        )

    candidates = _to_candidates(
        items,
        source_type=resources["kind"],
        retrieval_mode=retrieval_mode,
        retrieval_profile=retrieval_profile,
    )
    if metadata_filtering:
        return _apply_metadata_filtering(
            candidates,
            query=query,
            question_type=question_type,
            topk=top_k,
        )
    return candidates[:top_k]


def retrieve_main(query: str, config: Optional[RetrievalConfig] = None) -> List[Candidate]:
    return retrieve(query, load_resources("main"), config)


def retrieve_dual(query: str, config: Optional[RetrievalConfig] = None) -> List[Candidate]:
    cfg = _normalized_config(config)
    return retrieve(query, load_resources("main"), cfg) + retrieve(query, load_resources("support"), cfg)


def rerank_candidates(query: str, candidates: List[Candidate], topk: int) -> List[Candidate]:
    def adjusted_score(candidate: Candidate) -> float:
        bonus = 0.0
        source_type = (candidate.get("source_type") or "").lower()
        doc_id = (candidate.get("doc_id") or "").lower()
        text = (candidate.get("text") or "").lower()

        if source_type == "main":
            bonus += 0.12
        elif source_type == "support":
            bonus -= 0.03

        if "nice_ng198_guideline" in doc_id:
            bonus += 0.05

        bad_patterns = [
            "study selection",
            "research recommendations",
            "forest plots",
            "review question",
            "appendix",
        ]
        if any(p in text for p in bad_patterns):
            bonus -= 0.15

        return float(candidate.get("score", 0.0)) + bonus

    reranked = sorted(candidates, key=adjusted_score, reverse=True)[:topk]
    for rank, candidate in enumerate(reranked, start=1):
        candidate["rank"] = rank
    return reranked


def build_context(candidates: List[Candidate]) -> Tuple[str, List[Dict[str, Any]]]:
    if not candidates:
        return "", []

    pairs = [(dict(candidate), float(candidate.get("score", 0.0))) for candidate in candidates]
    context, sources = build_context_and_sources(pairs, max_chars_per_chunk=900)

    repaired_sources: List[Dict[str, Any]] = []
    for i, src in enumerate(sources):
        row = dict(src)
        if i < len(candidates):
            candidate = candidates[i]
            row["source_type"] = candidate.get("source_type")
            row["retrieval_mode"] = candidate.get("retrieval_mode")
            row["retrieval_profile"] = candidate.get("retrieval_profile")
        repaired_sources.append(row)

    return context, repaired_sources
