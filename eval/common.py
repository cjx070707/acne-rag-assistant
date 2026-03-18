import json
import os
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {e}") from e
    return items


def dump_jsonl(path: str | Path, items: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def safe_ratio(hit: int, total: int) -> float:
    return (hit / total) if total else 0.0

