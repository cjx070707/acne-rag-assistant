import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets"
INDEX_PATH = DATASET_DIR / "index.json"

COMMON_REQUIRED = {"id", "dataset", "question", "question_type", "difficulty", "expected_action"}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate benchmark dataset files against an index manifest.")
    parser.add_argument("--index-path", default=str(INDEX_PATH))
    return parser.parse_args()


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {e}") from e
    return rows


def validate_dataset(path: Path, expected_size: int):
    rows = load_jsonl(path)
    if len(rows) != expected_size:
        raise ValueError(f"{path} expected {expected_size} rows, got {len(rows)}")

    seen_ids = set()
    for row in rows:
        missing = COMMON_REQUIRED - set(row)
        if missing:
            raise ValueError(f"{path} row {row.get('id')} missing fields: {sorted(missing)}")
        if row["id"] in seen_ids:
            raise ValueError(f"{path} duplicate id: {row['id']}")
        seen_ids.add(row["id"])

    return len(rows)


def main():
    args = parse_args()
    index_path = Path(args.index_path)
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    total = 0
    for ds in index["datasets"]:
        path = Path(ds["path"])
        if not path.is_absolute():
            path = ROOT.parent / path
        count = validate_dataset(path, ds["size"])
        total += count
        print(f"[OK] {ds['name']}: {count} rows")

    print(f"[OK] validated datasets total={total}")


if __name__ == "__main__":
    main()
