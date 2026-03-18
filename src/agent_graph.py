from __future__ import annotations

import json
import os
import re
from typing import TypedDict, Optional, List, Dict, Any, Tuple

import faiss
from langgraph.graph import StateGraph, END

from .config import CHUNKS_MAIN_PATH, CHUNKS_SUPPORT_PATH, INDEX_MAIN_DIR, INDEX_SUPPORT_DIR
from .rag_core import (
    load_jsonl,
    load_id_map_list,
    build_chunk_lookup,
    build_lexical_stats,
    retrieve_topk_filtered,
    retrieve_topk_hybrid,
    build_context_and_sources,
    call_llm_siliconflow,
    call_llm_siliconflow_raw,
    REFUSAL_TEXT,
)

DEFAULT_RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "dense").strip().lower() or "dense"
RESOURCE_CACHE: Dict[str, Tuple[faiss.Index, List[str], Dict[str, Dict[str, Any]], Dict[str, Any]]] = {}


class AgentState(TypedDict, total=False):
    query: str
    final_query_used: str
    rewritten_query: Optional[str]

    topk: int
    prefilter_k: int
    model_name: str
    retrieval_mode: str
    attempt: int

    candidates: List[Tuple[Dict[str, Any], float]]
    reranked: List[Tuple[Dict[str, Any], float]]

    context: str
    sources: List[Dict[str, Any]]

    action: str
    judge_reason: str

    answer: str


def _load_single_resource(index_dir: str, chunks_path: str):
    faiss_path = os.path.join(index_dir, "faiss.index")
    idmap_path = os.path.join(index_dir, "id_map.json")

    if not (os.path.exists(faiss_path) and os.path.exists(idmap_path)):
        raise FileNotFoundError(f"Missing index files in: {index_dir} (faiss.index / id_map.json)")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    chunks = load_jsonl(chunks_path)
    chunk_lookup = build_chunk_lookup(chunks)
    lexical_stats = build_lexical_stats(chunk_lookup)

    index = faiss.read_index(faiss_path)
    id_map_list = load_id_map_list(idmap_path)
    return index, id_map_list, chunk_lookup, lexical_stats


def _get_resource(kind: str):
    if kind in RESOURCE_CACHE:
        return RESOURCE_CACHE[kind]

    if kind == "main":
        RESOURCE_CACHE[kind] = _load_single_resource(INDEX_MAIN_DIR, CHUNKS_MAIN_PATH)
        return RESOURCE_CACHE[kind]
    if kind == "support":
        RESOURCE_CACHE[kind] = _load_single_resource(INDEX_SUPPORT_DIR, CHUNKS_SUPPORT_PATH)
        return RESOURCE_CACHE[kind]
    raise ValueError(f"Unknown resource kind: {kind}")


def _tag_candidates(
    items: List[Tuple[Dict[str, Any], float]],
    source_type: str,
) -> List[Tuple[Dict[str, Any], float]]:
    out: List[Tuple[Dict[str, Any], float]] = []
    for chunk, score in items:
        row = dict(chunk)
        row["source_type"] = source_type
        out.append((row, float(score)))
    return out

def _retrieve_single_corpus(
    *,
    kind: str,
    query: str,
    topk: int,
    prefilter_k: int,
    retrieval_mode: str,
) -> List[Tuple[Dict[str, Any], float]]:
    index, id_map_list, chunk_lookup, lexical_stats = _get_resource(kind)

    if retrieval_mode == "hybrid":
        items = retrieve_topk_hybrid(
            question=query,
            index=index,
            id_map_list=id_map_list,
            chunk_lookup=chunk_lookup,
            lexical_stats=lexical_stats,
            top_k=topk,
            prefilter_k=prefilter_k,
            doc_filter=None,
            allow_junk_fallback=True,
        )
    else:
        items = retrieve_topk_filtered(
            question=query,
            index=index,
            id_map_list=id_map_list,
            chunk_lookup=chunk_lookup,
            top_k=topk,
            prefilter_k=prefilter_k,
            doc_filter=None,
            allow_junk_fallback=True,
        )

    return _tag_candidates(items or [], kind)


def _retrieve_dual(query: str, topk: int, prefilter_k: int, retrieval_mode: str) -> List[Tuple[Dict[str, Any], float]]:
    main_items = _retrieve_single_corpus(
        kind="main",
        query=query,
        topk=topk,
        prefilter_k=prefilter_k,
        retrieval_mode=retrieval_mode,
    )
    support_items = _retrieve_single_corpus(
        kind="support",
        query=query,
        topk=topk,
        prefilter_k=prefilter_k,
        retrieval_mode=retrieval_mode,
    )
    return main_items + support_items


def _safe_score(x: Tuple[Dict[str, Any], float]) -> float:
    _, score = x
    try:
        return float(score)
    except Exception:
        return 0.0


def _fallback_rerank(
    query: str,
    candidates: List[Tuple[Dict[str, Any], float]],
    topk: int,
) -> List[Tuple[Dict[str, Any], float]]:
    def adjusted_score(x: Tuple[Dict[str, Any], float]) -> float:
        chunk, score = x
        bonus = 0.0

        source_type = (chunk.get("source_type") or "").lower()
        doc_id = (chunk.get("doc_id") or "").lower()
        text = (chunk.get("text") or "").lower()

        # main 永远优先于 support
        if source_type == "main":
            bonus += 0.12
        elif source_type == "support":
            bonus -= 0.03

        # guideline 文档再加一点
        if "nice_ng198_guideline" in doc_id:
            bonus += 0.05

        # support 中明显属于附录/研究建议/流程图/森林图的垃圾页降权
        bad_patterns = [
            "study selection",
            "research recommendations",
            "forest plots",
            "review question",
            "appendix",
        ]
        if any(p in text for p in bad_patterns):
            bonus -= 0.15

        return float(score) + bonus

    return sorted(candidates, key=adjusted_score, reverse=True)[:topk]


def _build_context_from_candidates(
    candidates: List[Tuple[Dict[str, Any], float]]
) -> Tuple[str, List[Dict[str, Any]]]:
    if not candidates:
        return "", []

    context, sources = build_context_and_sources(candidates, max_chars_per_chunk=900)

    repaired_sources: List[Dict[str, Any]] = []
    for i, src in enumerate(sources):
        row = dict(src)
        if i < len(candidates):
            chunk, _ = candidates[i]
            if "source_type" in chunk:
                row["source_type"] = chunk["source_type"]
        repaired_sources.append(row)

    return context, repaired_sources


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    return {}


def _judge_action_with_llm(
    *,
    query: str,
    context: str,
    sources: List[Dict[str, Any]],
    attempt: int,
    model_name: str,
) -> Dict[str, str]:
    source_preview_lines = []
    for i, s in enumerate(sources[:8], start=1):
        doc_id = s.get("doc_id", "")
        page = s.get("page", "")
        source_type = s.get("source_type", "")
        source_preview_lines.append(f"[{i}] doc_id={doc_id}, page={page}, source_type={source_type}")
    source_preview = "\n".join(source_preview_lines) if source_preview_lines else "(no sources)"

    system_prompt = """
You are an evidence sufficiency judge for an acne clinical guideline RAG system.

You must decide the next action based on the retrieved evidence.

Allowed actions:
- answer
- rewrite_query
- refuse

Return ONLY strict JSON with this schema:
{"action":"answer|rewrite_query|refuse","reason":"short reason"}

Do not answer the user's medical question.
Do not cite sources.
Do not output markdown.
""".strip()

    user_prompt = f"""
Attempt number: {attempt}

User question:
{query}

Retrieved source summary:
{source_preview}

Retrieved context:
{context}

Decide whether the current evidence is sufficient to answer, whether the query should be rewritten for one more retrieval attempt, or whether the system should refuse.
If attempt >= 2 and evidence is still insufficient, prefer refuse.
Return JSON only.
""".strip()

    text = call_llm_siliconflow_raw(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
        temperature=0.0,
        max_tokens=300,
    )

    obj = _extract_json_object(text)

    action = str(obj.get("action", "")).strip().lower()
    reason = str(obj.get("reason", "")).strip()

    if action not in {"answer", "rewrite_query", "refuse"}:
        if attempt >= 2:
            action = "refuse"
            reason = reason or "invalid judge output after retry"
        else:
            action = "rewrite_query" if sources else "refuse"
            reason = reason or "invalid judge output"

    return {"action": action, "reason": reason or "no reason provided"}

def _rewrite_query_with_llm(
    *,
    query: str,
    context: str,
    judge_reason: str,
    model_name: str,
) -> str:
    system_prompt = """
You are a retrieval query rewriter for an acne clinical guideline RAG system.

Your job is to rewrite the user's question into a short, retrieval-friendly search query.

Rules:
- Preserve the original intent
- Make it more precise for guideline retrieval
- Prefer concise clinical wording
- Do not answer the question
- Do not explain your reasoning
- Do not output JSON
- Output only one short rewritten query
""".strip()

    user_prompt = f"""
Original question:
{query}

Why rewrite is needed:
{judge_reason}

Current retrieved context:
{context}

Return only one short rewritten query.
""".strip()

    rewritten = call_llm_siliconflow_raw(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
        temperature=0.0,
        max_tokens=80,
    ).strip()

    rewritten = rewritten.strip().strip('"').strip("'")
    rewritten = re.sub(r"\s+", " ", rewritten)

    if not rewritten:
        return query

    if len(rewritten) > 200:
        return query

    return rewritten

def retrieve_node(state: AgentState) -> AgentState:
    query = state["query"]
    topk = int(state.get("topk", 6))
    prefilter_k = int(state.get("prefilter_k", 120))
    retrieval_mode = str(state.get("retrieval_mode", DEFAULT_RETRIEVAL_MODE)).lower()

    candidates = _retrieve_single_corpus(
        kind="main",
        query=query,
        topk=topk,
        prefilter_k=prefilter_k,
        retrieval_mode=retrieval_mode,
    )
    state["candidates"] = candidates
    state["final_query_used"] = query
    return state


def rerank_node(state: AgentState) -> AgentState:
    candidates = state.get("candidates", []) or []
    topk = int(state.get("topk", 6))

    reranked = _fallback_rerank(state["query"], candidates, topk=topk)
    context, sources = _build_context_from_candidates(reranked)

    state["reranked"] = reranked
    state["context"] = context
    state["sources"] = sources
    return state


def judge_action_node(state: AgentState) -> AgentState:
    query = state["query"]
    model_name = state.get("model_name", "deepseek-ai/DeepSeek-V3.2")
    attempt = int(state.get("attempt", 1))
    context = state.get("context", "")
    sources = state.get("sources", []) or []

    if not context or not sources:
        state["action"] = "refuse" if attempt >= 2 else "rewrite_query"
        state["judge_reason"] = "no usable evidence retrieved"
        return state

    decision = _judge_action_with_llm(
        query=query,
        context=context,
        sources=sources,
        attempt=attempt,
        model_name=model_name,
    )
    state["action"] = decision["action"]
    state["judge_reason"] = decision["reason"]
    return state


def route_after_judge(state: AgentState) -> str:
    action = (state.get("action") or "").strip().lower()
    if action == "answer":
        return "answer"
    if action == "rewrite_query":
        return "rewrite_query"
    return "refuse"


def rewrite_query_node(state: AgentState) -> AgentState:
    model_name = state.get("model_name", "deepseek-ai/DeepSeek-V3.2")
    query = state["query"]
    context = state.get("context", "")
    judge_reason = state.get("judge_reason", "")

    rewritten = _rewrite_query_with_llm(
        query=query,
        context=context,
        judge_reason=judge_reason,
        model_name=model_name,
    )

    state["rewritten_query"] = rewritten
    state["attempt"] = 2
    state["final_query_used"] = rewritten
    return state


def second_retrieve_node(state: AgentState) -> AgentState:
    rewritten_query = state.get("rewritten_query") or state["query"]
    topk = int(state.get("topk", 6))
    prefilter_k = int(state.get("prefilter_k", 120))
    retrieval_mode = str(state.get("retrieval_mode", DEFAULT_RETRIEVAL_MODE)).lower()

    candidates = _retrieve_dual(
        query=rewritten_query,
        topk=topk,
        prefilter_k=prefilter_k,
        retrieval_mode=retrieval_mode,
    )
    state["candidates"] = candidates
    return state


def second_rerank_node(state: AgentState) -> AgentState:
    rewritten_query = state.get("rewritten_query") or state["query"]
    candidates = state.get("candidates", []) or []
    topk = int(state.get("topk", 6))

    reranked = _fallback_rerank(rewritten_query, candidates, topk=topk)
    context, sources = _build_context_from_candidates(reranked)

    state["reranked"] = reranked
    state["context"] = context
    state["sources"] = sources
    return state


def second_judge_node(state: AgentState) -> AgentState:
    rewritten_query = state.get("rewritten_query") or state["query"]
    model_name = state.get("model_name", "deepseek-ai/DeepSeek-V3.2")
    context = state.get("context", "")
    sources = state.get("sources", []) or []

    if not context or not sources:
        state["action"] = "refuse"
        state["judge_reason"] = "no usable evidence after rewrite retry"
        return state

    decision = _judge_action_with_llm(
        query=rewritten_query,
        context=context,
        sources=sources,
        attempt=2,
        model_name=model_name,
    )
    state["action"] = "answer" if decision["action"] == "answer" else "refuse"
    state["judge_reason"] = decision["reason"]
    return state


def route_after_second_judge(state: AgentState) -> str:
    return "answer" if state.get("action") == "answer" else "refuse"


def answer_node(state: AgentState) -> AgentState:
    model_name = state.get("model_name", "deepseek-ai/DeepSeek-V3.2")
    final_query = state.get("final_query_used") or state["query"]
    context = state.get("context", "")
    sources_meta = state.get("sources", []) or []

    if not context or not sources_meta:
        state["answer"] = REFUSAL_TEXT
        state["sources"] = []
        return state

    llm_out = call_llm_siliconflow(final_query, context, model_name)

    cited = llm_out.get("citations", [])
    resolved_sources: List[Dict[str, Any]] = []
    for cid in cited:
        if isinstance(cid, int) and 1 <= cid <= len(sources_meta):
            resolved_sources.append(sources_meta[cid - 1])

    if not resolved_sources:
        resolved_sources = sources_meta

    state["answer"] = llm_out.get("answer", "") or REFUSAL_TEXT
    state["sources"] = resolved_sources
    return state


def refuse_node(state: AgentState) -> AgentState:
    state["answer"] = REFUSAL_TEXT
    state["sources"] = []
    return state


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("judge_action", judge_action_node)

    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("second_retrieve", second_retrieve_node)
    g.add_node("second_rerank", second_rerank_node)
    g.add_node("second_judge", second_judge_node)

    g.add_node("answer", answer_node)
    g.add_node("refuse", refuse_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "judge_action")

    g.add_conditional_edges(
        "judge_action",
        route_after_judge,
        {
            "answer": "answer",
            "rewrite_query": "rewrite_query",
            "refuse": "refuse",
        },
    )

    g.add_edge("rewrite_query", "second_retrieve")
    g.add_edge("second_retrieve", "second_rerank")
    g.add_edge("second_rerank", "second_judge")

    g.add_conditional_edges(
        "second_judge",
        route_after_second_judge,
        {
            "answer": "answer",
            "refuse": "refuse",
        },
    )

    g.add_edge("answer", END)
    g.add_edge("refuse", END)
    return g.compile()


GRAPH = build_graph()


def run_agent_query(
    question: str,
    *,
    topk: int = 6,
    prefilter_k: int = 120,
    model_name: str = "deepseek-ai/DeepSeek-V3.2",
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
) -> Dict[str, Any]:
    out = GRAPH.invoke(
        {
            "query": question,
            "topk": int(topk),
            "prefilter_k": int(prefilter_k),
            "model_name": model_name,
            "retrieval_mode": str(retrieval_mode).lower(),
            "attempt": 1,
        }
    )
    return {
        "answer": out.get("answer", REFUSAL_TEXT),
        "sources": out.get("sources", []),
        "action": out.get("action", "refuse"),
        "judge_reason": out.get("judge_reason", ""),
        "rewritten_query": out.get("rewritten_query"),
        "final_query_used": out.get("final_query_used", question),
    }

