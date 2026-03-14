from __future__ import annotations

import os
import json
from typing import Dict, Any, Tuple


def _safe_makedirs(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def split_chunks(
    src_path: str,
    main_doc_id: str,
    out_main_path: str,
    out_support_path: str,
    *,
    ensure_unique_chunk_id: bool = True,
) -> Tuple[int, int, int]:
    """
    Split a unified chunks.jsonl into main/support jsonl files by doc_id.

    - main: records with doc_id == main_doc_id
    - support: everything else

    Also handles duplicate chunk_id:
    - If ensure_unique_chunk_id=True, duplicates will be renamed with suffix _dup2/_dup3...
    - If False, duplicates will be skipped (first one wins).

    Returns: (n_main, n_support, n_skipped_invalid)
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"src_path not found: {src_path}")

    _safe_makedirs(out_main_path)
    _safe_makedirs(out_support_path)

    seen_chunk_ids: Dict[str, int] = {}
    n_main = 0
    n_sup = 0
    n_bad = 0

    with open(src_path, "r", encoding="utf-8") as f_in, \
         open(out_main_path, "w", encoding="utf-8") as f_main, \
         open(out_support_path, "w", encoding="utf-8") as f_sup:

        for lineno, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec: Dict[str, Any] = json.loads(line)
            except Exception:
                n_bad += 1
                continue

            doc_id = rec.get("doc_id")
            chunk_id = rec.get("chunk_id")

            # Basic validation
            if not doc_id or not chunk_id:
                n_bad += 1
                continue

            # Handle duplicate chunk_id
            if chunk_id in seen_chunk_ids:
                seen_chunk_ids[chunk_id] += 1
                dup_idx = seen_chunk_ids[chunk_id]

                if ensure_unique_chunk_id:
                    # Rename chunk_id to keep every record (but avoid collisions)
                    new_chunk_id = f"{chunk_id}_dup{dup_idx}"
                    # Also store original
                    rec["chunk_id_original"] = chunk_id
                    rec["chunk_id"] = new_chunk_id
                    chunk_id = new_chunk_id
                else:
                    # Skip duplicates entirely
                    continue
            else:
                seen_chunk_ids[chunk_id] = 1

            out_line = json.dumps(rec, ensure_ascii=False)

            if doc_id == main_doc_id:
                f_main.write(out_line + "\n")
                n_main += 1
            else:
                f_sup.write(out_line + "\n")
                n_sup += 1

    return n_main, n_sup, n_bad


if __name__ == "__main__":
    # Repo-root friendly defaults (run from repo root)
    SRC = os.path.join("data", "processed", "chunks.jsonl")
    OUT_MAIN = os.path.join("data", "processed", "chunks_main.jsonl")
    OUT_SUP = os.path.join("data", "processed", "chunks_support.jsonl")

    MAIN_DOC_ID = "nice_ng198_guideline"  # change if your main doc_id differs

    n_main, n_sup, n_bad = split_chunks(
        src_path=SRC,
        main_doc_id=MAIN_DOC_ID,
        out_main_path=OUT_MAIN,
        out_support_path=OUT_SUP,
        ensure_unique_chunk_id=True,
    )

    print("Split done.")
    print(f"  main_doc_id: {MAIN_DOC_ID}")
    print(f"  main chunks: {n_main}")
    print(f"  support chunks: {n_sup}")
    print(f"  skipped invalid lines: {n_bad}")
    print(f"  wrote: {OUT_MAIN}")
    print(f"  wrote: {OUT_SUP}")