import os
import sys
from pprint import pprint

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from src.agent_graph import run_agent_query

query = "What does NICE recommend for acne-related scarring?"

out = run_agent_query(
    query,
    topk=6,
    prefilter_k=120,
    model_name="deepseek-ai/DeepSeek-V3.2",
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

