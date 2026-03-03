from __future__ import annotations

import os
from typing import TypedDict, Optional, List, Dict, Any

import faiss
from langgraph.graph import StateGraph, END

from .rag_core import (
    load_jsonl,
    load_id_map_list,
    build_chunk_lookup,
    retrieve_topk_filtered,
    build_context_and_sources,
    call_llm_siliconflow,
    REFUSAL_TEXT,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR) if os.path.basename(THIS_DIR).lower() == "src" else THIS_DIR

DEFAULT_INDEX_DIR = "index"
DEFAULT_CHUNKS_PATH = os.path.join("data", "processed", "chunks.jsonl")


class AgentState(TypedDict, total=False):
    query: str
    topk: int
    prefilter_k: int
    model_name: str

    doc_filter: Optional[str]
    context: str
    sources: List[Dict[str, Any]]

    answer: str


def _load_resources(index_dir: str = DEFAULT_INDEX_DIR, chunks_path: str = DEFAULT_CHUNKS_PATH):
    index_dir_abs = os.path.join(REPO_ROOT, index_dir)
    faiss_path = os.path.join(index_dir_abs, "faiss.index")
    idmap_path = os.path.join(index_dir_abs, "id_map.json")
    chunks_path_abs = os.path.join(REPO_ROOT, chunks_path)

    if not (os.path.exists(faiss_path) and os.path.exists(idmap_path)):
        raise FileNotFoundError(f"Missing index files in: {index_dir_abs} (faiss.index / id_map.json)")
    if not os.path.exists(chunks_path_abs):
        raise FileNotFoundError(f"Missing chunks file: {chunks_path_abs}")

    chunks = load_jsonl(chunks_path_abs)
    chunk_lookup = build_chunk_lookup(chunks)

    index = faiss.read_index(faiss_path)
    id_map_list = load_id_map_list(idmap_path)
    return index, id_map_list, chunk_lookup


_INDEX, _ID_MAP_LIST, _CHUNK_LOOKUP = _load_resources()


def _route_doc_filter(query: str) -> Optional[str]:
    q = (query or "").lower()

    if "ng198" in q:
        return "nice_ng198_guideline"

    if "evidence review a" in q:
        return "nice_evidence_a_information"
    if "evidence review b" in q:
        return "nice_evidence_b_skin_care"
    if "evidence review c" in q:
        return "nice_evidence_c_diet"
    if "evidence review d" in q:
        return "nice_evidence_d_referral"

    if " e1" in f" {q}" or q.startswith("e1"):
        return "e1_mild_moderate_network_meta"
    if " e2" in f" {q}" or q.startswith("e2"):
        return "e2_mild_moderate_pairwise"
    if " f1" in f" {q}" or q.startswith("f1"):
        return "f1_moderate_severe_network_meta"
    if " f2" in f" {q}" or q.startswith("f2"):
        return "f2_moderate_severe_pairwise"

    return None


def route_node(state: AgentState) -> AgentState:
    state["doc_filter"] = _route_doc_filter(state["query"])
    return state


def search_node(state: AgentState) -> AgentState:
    query = state["query"]
    topk = int(state.get("topk", 3))
    prefilter_k = int(state.get("prefilter_k", 120))
    doc_filter = state.get("doc_filter")

    retrieved = retrieve_topk_filtered(
        question=query,
        index=_INDEX,
        id_map_list=_ID_MAP_LIST,
        chunk_lookup=_CHUNK_LOOKUP,
        top_k=topk,
        prefilter_k=prefilter_k,
        doc_filter=doc_filter,
        allow_junk_fallback=True,
    )

    if not retrieved:
        state["context"] = ""
        state["sources"] = []
        return state

    context, sources = build_context_and_sources(retrieved, max_chars_per_chunk=900)
    state["context"] = context
    state["sources"] = sources
    return state


def judge_node(state: AgentState) -> str:
    topk = int(state.get("topk", 3))
    sources = state.get("sources", [])
    # agreed: count-only judge
    if len(sources) < topk:
        return "fallback"
    return "answer"


def fallback_search_node(state: AgentState) -> AgentState:
    # agreed: fallback by removing doc_filter and searching whole corpus
    state["doc_filter"] = None
    return search_node(state)


def answer_node(state: AgentState) -> AgentState:
    query = state["query"]
    model_name = state.get("model_name", "deepseek-ai/DeepSeek-V3.2")
    context = state.get("context", "")
    sources_meta = state.get("sources", [])

    if not context or not sources_meta:
        state["answer"] = REFUSAL_TEXT
        state["sources"] = []
        return state

    llm_out = call_llm_siliconflow(query, context, model_name)

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


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("search", search_node)
    g.add_node("fallback_search", fallback_search_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("route")
    g.add_edge("route", "search")
    g.add_conditional_edges("search", judge_node, {"fallback": "fallback_search", "answer": "answer"})
    g.add_edge("fallback_search", "answer")
    g.add_edge("answer", END)
    return g.compile()


GRAPH = build_graph()


def run_agent_query(
    question: str,
    *,
    topk: int = 3,
    prefilter_k: int = 120,
    model_name: str = "deepseek-ai/DeepSeek-V3.2",
) -> Dict[str, Any]:
    out = GRAPH.invoke(
        {
            "query": question,
            "topk": int(topk),
            "prefilter_k": int(prefilter_k),
            "model_name": model_name,
        }
    )
    return {"answer": out.get("answer", REFUSAL_TEXT), "sources": out.get("sources", [])}