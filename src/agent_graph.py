from __future__ import annotations

import json
import os
import re
from typing import TypedDict, Optional, List, Dict, Any, Tuple

from langgraph.graph import StateGraph, END

from .retrieval_profiles import DEFAULT_RUNTIME_RETRIEVAL_PROFILE, resolve_retrieval_profile
from .retrieval import (
    DEFAULT_RETRIEVAL_MODE,
    build_context,
    rerank_candidates,
    route_query,
    retrieve_dual,
    retrieve_main,
)
from .rag_core import call_llm_siliconflow, call_llm_siliconflow_raw, REFUSAL_TEXT


class AgentState(TypedDict, total=False):
    query: str
    final_query_used: str
    rewritten_query: Optional[str]

    topk: int
    prefilter_k: int
    model_name: str
    retrieval_profile: str
    retrieval_mode: Optional[str]
    metadata_filtering: Optional[bool]
    query_routing: Optional[bool]
    question_type: Optional[str]
    attempt: int

    candidates: List[Tuple[Dict[str, Any], float]]
    reranked: List[Tuple[Dict[str, Any], float]]

    context: str
    sources: List[Dict[str, Any]]

    action: str
    judge_reason: str

    answer: str


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
    retrieval_profile = str(state.get("retrieval_profile", DEFAULT_RUNTIME_RETRIEVAL_PROFILE))
    resolved = resolve_retrieval_profile(
        retrieval_profile,
        {
            "retrieval_mode": state.get("retrieval_mode"),
            "metadata_filtering": state.get("metadata_filtering"),
            "query_routing": state.get("query_routing"),
            "apply_filtering": state.get("apply_filtering"),
        },
    )
    retrieval_mode = str(resolved.get("retrieval_mode", DEFAULT_RETRIEVAL_MODE)).lower()
    metadata_filtering = bool(resolved.get("metadata_filtering", False))
    query_routing = bool(resolved.get("query_routing", False))
    question_type = state.get("question_type")

    if query_routing:
        decision = route_query(query, question_type=question_type, attempt=1)
        metadata_filtering = bool(decision["metadata_filtering"])
        question_type = decision["question_type"]
        state["question_type"] = question_type

    candidates = retrieve_main(
        query,
        {
            "retrieval_profile": retrieval_profile,
            "topk": topk,
            "prefilter_k": prefilter_k,
            "retrieval_mode": retrieval_mode,
            "metadata_filtering": metadata_filtering,
            "apply_filtering": resolved.get("apply_filtering"),
            "question_type": question_type,
            "query_routing": query_routing,
        },
    )
    state["candidates"] = candidates
    state["final_query_used"] = query
    return state


def rerank_node(state: AgentState) -> AgentState:
    candidates = state.get("candidates", []) or []
    topk = int(state.get("topk", 6))

    reranked = rerank_candidates(state["query"], candidates, topk=topk)
    context, sources = build_context(reranked)

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
    retrieval_profile = str(state.get("retrieval_profile", DEFAULT_RUNTIME_RETRIEVAL_PROFILE))
    resolved = resolve_retrieval_profile(
        retrieval_profile,
        {
            "retrieval_mode": state.get("retrieval_mode"),
            "metadata_filtering": state.get("metadata_filtering"),
            "query_routing": state.get("query_routing"),
            "apply_filtering": state.get("apply_filtering"),
        },
    )
    retrieval_mode = str(resolved.get("retrieval_mode", DEFAULT_RETRIEVAL_MODE)).lower()
    metadata_filtering = bool(resolved.get("metadata_filtering", False))
    query_routing = bool(resolved.get("query_routing", False))
    question_type = state.get("question_type")

    use_support = True
    if query_routing:
        decision = route_query(rewritten_query, question_type=question_type, attempt=2)
        metadata_filtering = bool(decision["metadata_filtering"])
        question_type = decision["question_type"]
        use_support = bool(decision["use_support"])
        state["question_type"] = question_type

    if use_support:
        candidates = retrieve_dual(
            rewritten_query,
            {
                "retrieval_profile": retrieval_profile,
                "topk": topk,
                "prefilter_k": prefilter_k,
                "retrieval_mode": retrieval_mode,
                "metadata_filtering": metadata_filtering,
                "apply_filtering": resolved.get("apply_filtering"),
                "question_type": question_type,
                "query_routing": query_routing,
            },
        )
    else:
        candidates = retrieve_main(
            rewritten_query,
            {
                "retrieval_profile": retrieval_profile,
                "topk": topk,
                "prefilter_k": prefilter_k,
                "retrieval_mode": retrieval_mode,
                "metadata_filtering": metadata_filtering,
                "apply_filtering": resolved.get("apply_filtering"),
                "question_type": question_type,
                "query_routing": query_routing,
            },
        )
    state["candidates"] = candidates
    return state


def second_rerank_node(state: AgentState) -> AgentState:
    rewritten_query = state.get("rewritten_query") or state["query"]
    candidates = state.get("candidates", []) or []
    topk = int(state.get("topk", 6))

    reranked = rerank_candidates(rewritten_query, candidates, topk=topk)
    context, sources = build_context(reranked)

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
    retrieval_profile: str = DEFAULT_RUNTIME_RETRIEVAL_PROFILE,
    retrieval_mode: Optional[str] = None,
    metadata_filtering: Optional[bool] = None,
    query_routing: Optional[bool] = None,
) -> Dict[str, Any]:
    out = GRAPH.invoke(
        {
            "query": question,
            "topk": int(topk),
            "prefilter_k": int(prefilter_k),
            "model_name": model_name,
            "retrieval_profile": retrieval_profile,
            "retrieval_mode": str(retrieval_mode).lower() if retrieval_mode else None,
            "metadata_filtering": metadata_filtering,
            "query_routing": query_routing,
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

