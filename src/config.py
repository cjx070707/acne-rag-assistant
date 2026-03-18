import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR) if os.path.basename(THIS_DIR).lower() == "src" else THIS_DIR

DATA_DIR = os.path.join(REPO_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")
EVAL_DIR = os.path.join(REPO_ROOT, "eval")
EVAL_ARTIFACTS_DIR = os.path.join(EVAL_DIR, "artifacts")

CHUNKS_ALL_PATH = os.path.join(PROCESSED_DIR, "chunks.jsonl")
CHUNKS_MAIN_PATH = os.path.join(PROCESSED_DIR, "chunks_main.jsonl")
CHUNKS_SUPPORT_PATH = os.path.join(PROCESSED_DIR, "chunks_support.jsonl")

INDEX_ALL_DIR = os.path.join(ARTIFACTS_DIR, "index")
INDEX_MAIN_DIR = os.path.join(ARTIFACTS_DIR, "index_main")
INDEX_SUPPORT_DIR = os.path.join(ARTIFACTS_DIR, "index_support")

QUESTIONS_PATH = os.path.join(EVAL_DIR, "questions.jsonl")
RETRIEVAL_RESULTS_PATH = os.path.join(EVAL_ARTIFACTS_DIR, "retrieval_results.jsonl")
RERANKED_RESULTS_PATH = os.path.join(EVAL_ARTIFACTS_DIR, "reranked_results_top20.jsonl")

