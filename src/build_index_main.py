import os
import json
import time
from typing import List, Dict, Tuple, Callable

import numpy as np
from tqdm import tqdm
import faiss

# ---------------- Paths (robust to CWD / F5) ----------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR) if os.path.basename(THIS_DIR).lower() == "src" else THIS_DIR

CHUNKS_PATH = os.path.join(REPO_ROOT, "data", "processed", "chunks_main.jsonl")
INDEX_DIR = os.path.join(REPO_ROOT, "index_main")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
ID_MAP_PATH = os.path.join(INDEX_DIR, "id_map.json")

# Cache: embeddings + signature (so we don't recompute every time)
EMB_CACHE_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
SIG_PATH = os.path.join(INDEX_DIR, "embeddings.sig.json")

# Embedding backend
OPENAI_MODEL = "text-embedding-3-small"
LOCAL_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_chunks(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing chunks file: {path} (run ingest.py first)")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at line {line_no}: {e}") from e
    if not items:
        raise RuntimeError("chunks.jsonl is empty")
    return items


def get_embedder() -> Tuple[str, Callable[[List[str]], np.ndarray]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        def embed_texts(texts: List[str]) -> np.ndarray:
            resp = client.embeddings.create(model=OPENAI_MODEL, input=texts)
            vecs = [d.embedding for d in resp.data]
            return np.array(vecs, dtype=np.float32)

        return "openai", embed_texts

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(LOCAL_MODEL)

    def embed_texts(texts: List[str]) -> np.ndarray:
        # model.encode 内部已做 batch，这里外面再分 batch 便于进度与可控
        arr = model.encode(
            texts,
            batch_size=min(BATCH_SIZE, 128),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return arr.astype(np.float32)

    return "local", embed_texts


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    # cosine similarity = normalize + inner product
    embeddings = l2_normalize(embeddings).astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
    os.replace(tmp, path)


def make_signature(chunks_path: str, backend: str) -> Dict:
    """
    签名用于判断缓存是否可用：
    - chunks 文件大小 + mtime
    - embedding backend/model
    够用且快（不做全文 hash，避免又变慢）
    """
    st = os.stat(chunks_path)
    return {
        "chunks_path": os.path.abspath(chunks_path),
        "chunks_mtime": int(st.st_mtime),
        "chunks_size": int(st.st_size),
        "backend": backend,
        "openai_model": OPENAI_MODEL if backend == "openai" else None,
        "local_model": LOCAL_MODEL if backend == "local" else None,
    }


def load_signature(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_signature(path: str, sig: Dict) -> None:
    atomic_write_text(path, json.dumps(sig, ensure_ascii=False, indent=2))


def main():
    ensure_dir(INDEX_DIR)

    t0 = time.perf_counter()
    chunks = load_chunks(CHUNKS_PATH)
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    t_load = time.perf_counter()

    backend, embed_texts = get_embedder()
    print(f"[INFO] embedding backend = {backend}")

    # -------- embeddings cache --------
    sig_now = make_signature(CHUNKS_PATH, backend)
    sig_old = load_signature(SIG_PATH)

    use_cache = (
        sig_old == sig_now
        and os.path.exists(EMB_CACHE_PATH)
        and os.path.getsize(EMB_CACHE_PATH) > 0
    )

    if use_cache:
        embs = np.load(EMB_CACHE_PATH).astype(np.float32)
        print(f"[OK] loaded cached embeddings: {EMB_CACHE_PATH} shape={embs.shape}")
        t_embed = time.perf_counter()
    else:
        all_vecs = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
            batch = texts[i:i + BATCH_SIZE]
            vecs = embed_texts(batch)
            if vecs.ndim != 2:
                raise RuntimeError("Embedding output must be 2D array")
            all_vecs.append(vecs)

        embs = np.vstack(all_vecs).astype(np.float32)
        if embs.shape[0] != len(chunk_ids):
            raise RuntimeError("Embedding count mismatch")

        # cache embeddings + signature
        np.save(EMB_CACHE_PATH, embs)
        save_signature(SIG_PATH, sig_now)
        print(f"[OK] cached embeddings: {EMB_CACHE_PATH} shape={embs.shape}")
        t_embed = time.perf_counter()

    # -------- build & write index (atomic) --------
    index = build_faiss_index(embs)
    t_index = time.perf_counter()

    # write faiss index via temp file then replace (avoid half-written file)
    tmp_index = FAISS_INDEX_PATH + ".tmp"
    faiss.write_index(index, tmp_index)
    os.replace(tmp_index, FAISS_INDEX_PATH)

    atomic_write_text(ID_MAP_PATH, json.dumps(chunk_ids, ensure_ascii=False, indent=2))
    t_write = time.perf_counter()

    print(f"[OK] index size = {index.ntotal}")
    print(f"[OK] wrote: {FAISS_INDEX_PATH}")
    print(f"[OK] wrote: {ID_MAP_PATH}")

    # -------- timing report --------
    print(
        "[TIME] load_chunks:  %.2fs\n"
        "[TIME] embeddings:   %.2fs\n"
        "[TIME] build_index:  %.2fs\n"
        "[TIME] write_files:  %.2fs\n"
        "[TIME] total:        %.2fs"
        % (
            (t_load - t0),
            (t_embed - t_load),
            (t_index - t_embed),
            (t_write - t_index),
            (t_write - t0),
        )
    )


if __name__ == "__main__":
    main()