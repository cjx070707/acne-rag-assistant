import os
import json
import time
import argparse
from typing import List, Dict, Tuple, Callable

import numpy as np
from tqdm import tqdm
import faiss

from .config import CHUNKS_ALL_PATH, CHUNKS_MAIN_PATH, CHUNKS_SUPPORT_PATH
from .config import INDEX_ALL_DIR, INDEX_MAIN_DIR, INDEX_SUPPORT_DIR

# Embedding backend
OPENAI_MODEL = "text-embedding-3-small"
LOCAL_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

# Target configs
TARGET_CONFIG = {
    "main": {
        "chunks_path": CHUNKS_MAIN_PATH,
        "index_dir": INDEX_MAIN_DIR,
    },
    "support": {
        "chunks_path": CHUNKS_SUPPORT_PATH,
        "index_dir": INDEX_SUPPORT_DIR,
    },
    "all": {
        "chunks_path": CHUNKS_ALL_PATH,
        "index_dir": INDEX_ALL_DIR,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index for different chunk sets.")
    parser.add_argument(
        "--target",
        choices=["main", "support", "all"],
        default="all",
        help="Which chunk set to build index for",
    )
    return parser.parse_args()


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
        raise RuntimeError(f"{path} is empty")
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
    embeddings = l2_normalize(embeddings).astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
    os.replace(tmp, path)


def make_signature(chunks_path: str, backend: str) -> Dict:
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
    args = parse_args()
    cfg = TARGET_CONFIG[args.target]

    chunks_path = cfg["chunks_path"]
    index_dir = cfg["index_dir"]
    faiss_index_path = os.path.join(index_dir, "faiss.index")
    id_map_path = os.path.join(index_dir, "id_map.json")
    emb_cache_path = os.path.join(index_dir, "embeddings.npy")
    sig_path = os.path.join(index_dir, "embeddings.sig.json")

    ensure_dir(index_dir)

    t0 = time.perf_counter()
    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    t_load = time.perf_counter()

    backend, embed_texts = get_embedder()
    print(f"[INFO] target = {args.target}")
    print(f"[INFO] chunks_path = {chunks_path}")
    print(f"[INFO] index_dir = {index_dir}")
    print(f"[INFO] embedding backend = {backend}")

    sig_now = make_signature(chunks_path, backend)
    sig_old = load_signature(sig_path)

    use_cache = (
        sig_old == sig_now
        and os.path.exists(emb_cache_path)
        and os.path.getsize(emb_cache_path) > 0
    )

    if use_cache:
        embs = np.load(emb_cache_path).astype(np.float32)
        print(f"[OK] loaded cached embeddings: {emb_cache_path} shape={embs.shape}")
        if embs.shape[0] != len(chunk_ids):
            raise RuntimeError(
                f"Cached embeddings count mismatch: {embs.shape[0]} != {len(chunk_ids)}"
            )
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

        np.save(emb_cache_path, embs)
        save_signature(sig_path, sig_now)
        print(f"[OK] cached embeddings: {emb_cache_path} shape={embs.shape}")
        t_embed = time.perf_counter()

    index = build_faiss_index(embs)
    t_index = time.perf_counter()

    tmp_index = faiss_index_path + ".tmp"
    faiss.write_index(index, tmp_index)
    os.replace(tmp_index, faiss_index_path)

    atomic_write_text(id_map_path, json.dumps(chunk_ids, ensure_ascii=False, indent=2))
    t_write = time.perf_counter()

    print(f"[OK] index size = {index.ntotal}")
    print(f"[OK] wrote: {faiss_index_path}")
    print(f"[OK] wrote: {id_map_path}")

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
