import json
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

IN_PATH  = Path("eval/artifacts/retrieval_results.jsonl")
OUT_PATH = Path("eval/artifacts/reranked_results_top20.jsonl")

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 32

def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def dump_jsonl(p: Path, items: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    data = load_jsonl(IN_PATH)
    ce = CrossEncoder(MODEL_NAME)

    out = []
    for ex in data:
        qid = ex["id"]
        qtext = ex.get("question")  # 如果你的 retrieval_results 没存 question，就从 questions.jsonl 补
        if not qtext:
            raise ValueError(f"{qid} missing 'question' field in retrieval results. Store it when dumping retrieval.")

        cands = ex["topk"]  # list of chunks
        pairs = [(qtext, c.get("text", "")) for c in cands]
        scores = ce.predict(pairs, batch_size=BATCH_SIZE)

        # attach rerank score
        for c, s in zip(cands, scores):
            c["rerank_score"] = float(s)

        cands_sorted = sorted(cands, key=lambda x: x["rerank_score"], reverse=True)

        out.append({
            "id": qid,
            "question": qtext,
            "topk": cands_sorted
        })

    dump_jsonl(OUT_PATH, out)
    print("Wrote:", OUT_PATH)

if __name__ == "__main__":
    main()
