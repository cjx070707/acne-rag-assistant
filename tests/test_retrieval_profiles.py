import unittest

from src.retrieval_profiles import (
    DEFAULT_EVAL_RETRIEVAL_PROFILE,
    DEFAULT_RUNTIME_RETRIEVAL_PROFILE,
    resolve_retrieval_profile,
)


class RetrievalProfilesTest(unittest.TestCase):
    def test_runtime_default_profile(self):
        cfg = resolve_retrieval_profile(DEFAULT_RUNTIME_RETRIEVAL_PROFILE)
        self.assertEqual(cfg["retrieval_profile"], DEFAULT_RUNTIME_RETRIEVAL_PROFILE)
        self.assertEqual(cfg["retrieval_mode"], "dense")
        self.assertTrue(cfg["apply_filtering"])

    def test_eval_default_profile(self):
        cfg = resolve_retrieval_profile(DEFAULT_EVAL_RETRIEVAL_PROFILE)
        self.assertEqual(cfg["retrieval_profile"], DEFAULT_EVAL_RETRIEVAL_PROFILE)
        self.assertEqual(cfg["retrieval_mode"], "dense")
        self.assertFalse(cfg["apply_filtering"])

    def test_overrides_take_precedence(self):
        cfg = resolve_retrieval_profile(
            "dense_routing_v1",
            {
                "retrieval_mode": "hybrid",
                "query_routing": False,
            },
        )
        self.assertEqual(cfg["retrieval_profile"], "dense_routing_v1")
        self.assertEqual(cfg["retrieval_mode"], "hybrid")
        self.assertFalse(cfg["query_routing"])


if __name__ == "__main__":
    unittest.main()
