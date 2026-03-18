import json
import tempfile
import unittest
from pathlib import Path

from src.split_chunk import split_chunks


class SplitChunksTest(unittest.TestCase):
    def test_duplicate_chunk_ids_are_renamed_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            src_path = base / "chunks.jsonl"
            out_main = base / "chunks_main.jsonl"
            out_support = base / "chunks_support.jsonl"

            rows = [
                {"chunk_id": "c1", "doc_id": "main_doc", "text": "alpha"},
                {"chunk_id": "c1", "doc_id": "main_doc", "text": "beta"},
                {"chunk_id": "c2", "doc_id": "support_doc", "text": "gamma"},
            ]

            with src_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            n_main, n_support, n_bad = split_chunks(
                src_path=str(src_path),
                main_doc_id="main_doc",
                out_main_path=str(out_main),
                out_support_path=str(out_support),
                ensure_unique_chunk_id=True,
            )

            self.assertEqual((n_main, n_support, n_bad), (2, 1, 0))

            with out_main.open("r", encoding="utf-8") as f:
                main_rows = [json.loads(line) for line in f if line.strip()]

            self.assertEqual(main_rows[0]["chunk_id"], "c1")
            self.assertEqual(main_rows[1]["chunk_id"], "c1_dup2")
            self.assertEqual(main_rows[1]["chunk_id_original"], "c1")

    def test_duplicate_chunk_ids_can_be_dropped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            src_path = base / "chunks.jsonl"
            out_main = base / "chunks_main.jsonl"
            out_support = base / "chunks_support.jsonl"

            rows = [
                {"chunk_id": "c1", "doc_id": "main_doc", "text": "alpha"},
                {"chunk_id": "c1", "doc_id": "main_doc", "text": "beta"},
                {"chunk_id": "c2", "doc_id": "support_doc", "text": "gamma"},
            ]

            with src_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            n_main, n_support, n_bad = split_chunks(
                src_path=str(src_path),
                main_doc_id="main_doc",
                out_main_path=str(out_main),
                out_support_path=str(out_support),
                ensure_unique_chunk_id=False,
            )

            self.assertEqual((n_main, n_support, n_bad), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
