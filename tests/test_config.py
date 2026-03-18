import os
import unittest

from src import config


class ConfigPathsTest(unittest.TestCase):
    def test_repo_layout_paths_stay_inside_repo(self):
        repo_root = os.path.abspath(config.REPO_ROOT)
        important_paths = [
            config.DATA_DIR,
            config.PROCESSED_DIR,
            config.ARTIFACTS_DIR,
            config.EVAL_DIR,
            config.EVAL_ARTIFACTS_DIR,
            config.CHUNKS_ALL_PATH,
            config.CHUNKS_MAIN_PATH,
            config.CHUNKS_SUPPORT_PATH,
            config.INDEX_ALL_DIR,
            config.INDEX_MAIN_DIR,
            config.INDEX_SUPPORT_DIR,
            config.QUESTIONS_PATH,
            config.RETRIEVAL_RESULTS_PATH,
            config.RERANKED_RESULTS_PATH,
        ]

        for path in important_paths:
            self.assertTrue(os.path.abspath(path).startswith(repo_root))

    def test_eval_artifacts_are_nested_under_eval(self):
        eval_dir = os.path.abspath(config.EVAL_DIR)
        self.assertTrue(os.path.abspath(config.EVAL_ARTIFACTS_DIR).startswith(eval_dir))
        self.assertTrue(os.path.abspath(config.RETRIEVAL_RESULTS_PATH).startswith(eval_dir))
        self.assertTrue(os.path.abspath(config.RERANKED_RESULTS_PATH).startswith(eval_dir))


if __name__ == "__main__":
    unittest.main()
