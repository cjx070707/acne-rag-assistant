import argparse
import json
import os
import sys
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from eval.common import dump_jsonl, load_jsonl, normalize_text, safe_ratio
from src.agent_graph import run_agent_query


DEFAULT_DATASET = str(Path("eval/datasets/qa_grounded_v1.jsonl"))
DEFAULT_OUT = str(Path("eval/artifacts/qa_grounded_report.jsonl"))


def parse_args():
    parser = argparse.ArgumentParser(description="Run grounded QA evaluation on a labeled dataset.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--out-path", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--prefilter-k", type=int, default=120)
    parser.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default="dense")
    return parser.parse_args()


def contains_all(answer: str, phrases: list[str]) -> tuple[bool, list[str]]:
    text = normalize_text(answer)
    missing = [p for p in phrases if normalize_text(p) not in text]
    return not missing, missing


def contains_any(answer: str, phrases: list[str]) -> tuple[bool, list[str]]:
    text = normalize_text(answer)
    present = [p for p in phrases if normalize_text(p) in text]
    return bool(present), present


def main():
    args = parse_args()
    rows = load_jsonl(args.dataset_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    reports = []
    action_correct = 0
    answered = 0
    scored_answer_rows = 0
    all_points_ok = 0
    no_bad_claims = 0
    primary_ok = 0

    for row in rows:
        result = run_agent_query(
            row["question"],
            topk=args.topk,
            prefilter_k=args.prefilter_k,
            model_name=args.model,
            retrieval_mode=args.retrieval_mode,
        )
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        actual_action = result.get("action")
        expected_action = row["expected_action"]

        must_ok, missing = contains_all(answer, row.get("must_include_points", []))
        bad_found, present_bad = contains_any(answer, row.get("must_not_claim", []))
        has_primary = any(s.get("doc_id") == row.get("gold_doc_id") for s in sources)

        if actual_action == expected_action:
            action_correct += 1
        if actual_action == "answer":
            answered += 1
        if expected_action == "answer":
            scored_answer_rows += 1
            if must_ok:
                all_points_ok += 1
            if not bad_found:
                no_bad_claims += 1
            if (not row.get("primary_source_required")) or has_primary:
                primary_ok += 1

        reports.append(
            {
                "id": row["id"],
                "question": row["question"],
                "expected_action": expected_action,
                "actual_action": actual_action,
                "action_correct": actual_action == expected_action,
                "must_include_ok": must_ok if expected_action == "answer" else None,
                "missing_points": missing,
                "must_not_claim_ok": (not bad_found) if expected_action == "answer" else None,
                "bad_claims_present": present_bad,
                "primary_source_ok": has_primary if expected_action == "answer" else None,
                "sources": sources,
                "answer": answer,
            }
        )

    dump_jsonl(args.out_path, reports)
    total = len(rows)
    print(f"Action accuracy: {action_correct}/{total} = {safe_ratio(action_correct, total):.3f}")
    print(f"Answered: {answered}/{total} = {safe_ratio(answered, total):.3f}")
    if scored_answer_rows:
        print(f"Must-include pass: {all_points_ok}/{scored_answer_rows} = {safe_ratio(all_points_ok, scored_answer_rows):.3f}")
        print(f"No forbidden claims: {no_bad_claims}/{scored_answer_rows} = {safe_ratio(no_bad_claims, scored_answer_rows):.3f}")
        print(f"Primary-source pass: {primary_ok}/{scored_answer_rows} = {safe_ratio(primary_ok, scored_answer_rows):.3f}")
    print(f"Wrote: {args.out_path}")


if __name__ == "__main__":
    main()
