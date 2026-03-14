import os
import sys
from pprint import pprint

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from src.agent_graph import GRAPH

query = "What does NICE recommend for acne-related scarring?"

out = GRAPH.invoke(
    {
        "query": query,
        "topk": 6,
        "prefilter_k": 120,
        "model_name": "deepseek-ai/DeepSeek-V3.2",
        "attempt": 1,
    }
)

print("\n=== FINAL OUTPUT ===")
pprint(
    {
        "answer": out.get("answer"),
        "action": out.get("action"),
        "judge_reason": out.get("judge_reason"),
        "rewritten_query": out.get("rewritten_query"),
        "final_query_used": out.get("final_query_used"),
    },
    width=120,
)

print("\n=== FINAL SOURCES ===")
pprint(out.get("sources", []), width=140)

print("\n=== RERANKED CANDIDATES ===")
reranked = out.get("reranked", [])
for i, item in enumerate(reranked, start=1):
    chunk, score = item
    print(f"\n--- Rank {i} ---")
    print("score:", score)
    print("source_type:", chunk.get("source_type"))
    print("chunk_id:", chunk.get("chunk_id"))
    print("doc_id:", chunk.get("doc_id"))
    print("page:", chunk.get("page"))
    print("title:", chunk.get("title"))
    text = (chunk.get("text") or "").replace("\n", " ")
    print("text:", text[:500])

print("\n=== FINAL CONTEXT PREVIEW ===")
context = out.get("context", "")
print(context[:3000])
