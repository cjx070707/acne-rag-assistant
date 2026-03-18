import unittest

from src.rag_core import (
    build_lexical_stats,
    lexical_search,
    reciprocal_rank_fuse,
    tokenize_for_lexical,
)


class HybridRetrievalTest(unittest.TestCase):
    def test_tokenize_for_lexical_normalizes_ascii_terms(self):
        tokens = tokenize_for_lexical("Acne-related scarring in PCOS, age 16+")
        self.assertEqual(tokens, ["acne", "related", "scarring", "in", "pcos", "age", "16"])

    def test_lexical_search_prefers_keyword_overlap(self):
        chunks = {
            "c1": {"chunk_id": "c1", "doc_id": "main", "text": "Offer topical treatment for mild acne."},
            "c2": {"chunk_id": "c2", "doc_id": "main", "text": "Refer acne conglobata for specialist assessment."},
            "c3": {"chunk_id": "c3", "doc_id": "main", "text": "Maintenance review and follow up advice."},
        }
        lexical_stats = build_lexical_stats(chunks)

        results = lexical_search(
            question="When should acne conglobata be referred?",
            lexical_stats=lexical_stats,
            chunk_lookup=chunks,
            top_k=2,
        )

        self.assertEqual(results[0][0]["chunk_id"], "c2")

    def test_reciprocal_rank_fuse_merges_dense_and_lexical_hits(self):
        dense = [
            ({"chunk_id": "c1", "doc_id": "main", "text": "dense first"}, 0.9),
            ({"chunk_id": "c2", "doc_id": "main", "text": "dense second"}, 0.8),
        ]
        lexical = [
            ({"chunk_id": "c2", "doc_id": "main", "text": "lexical first"}, 2.0),
            ({"chunk_id": "c3", "doc_id": "main", "text": "lexical second"}, 1.0),
        ]

        fused = reciprocal_rank_fuse([dense, lexical], top_k=3, weights=[1.0, 1.0], rrf_k=10)
        self.assertEqual([item[0]["chunk_id"] for item in fused], ["c2", "c1", "c3"])


if __name__ == "__main__":
    unittest.main()
