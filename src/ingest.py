import os
import json
import re
from typing import Dict, List, Tuple

from pypdf import PdfReader
from tqdm import tqdm


MANIFEST_PATH = "data/raw_docs/acne/manifest.jsonl"
OUTPUT_CHUNKS_PATH = "data/processed/chunks.jsonl"

# 建议：段落优先切块，超长段落再滑窗
CHUNK_CHAR_SIZE = 1200
CHUNK_CHAR_OVERLAP = 200
MIN_CHUNK_CHARS = 200

# 段落层面：太短的段落直接丢（避免标题/页眉残片）
MIN_PARA_CHARS = 80


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def read_manifest(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def fix_hyphen_linebreaks(s: str) -> str:
    # exam-\nple -> example
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)


def clean_page_text(raw: str) -> str:
    """
    关键：先按行清洗，再合并段落。
    目标：去掉页眉页脚/版权/目录点线/纯符号行，避免污染 embedding。
    """
    if not raw:
        return ""

    raw = fix_hyphen_linebreaks(raw)

    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            lines.append("")  # 保留空行作为段落分隔
            continue

        # 1) NICE 常见版权/链接/页码
        if "© NICE" in s:
            continue
        if "nice.org.uk/terms-andconditions" in s or "notice-of-rights" in s:
            continue
        if re.search(r"\bPage\s+\d+\s+of\s+\d+\b", s, re.IGNORECASE):
            continue

        # 2) 目录点线 / 引导符： ".... 58"
        if re.fullmatch(r"[.\s]{8,}\d{1,4}", s):
            continue

        # 3) 几乎全是标点/点/空格
        if re.fullmatch(r"[.\s]{20,}", s):
            continue

        # 4) 过短且像页眉（可选：保守一点）
        # if len(s) < 10:
        #     continue

        lines.append(s)

    # 合并：把多空行压成一个空行，作为段落分隔
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def split_paragraphs(text: str) -> List[str]:
    """
    按空行切段，过滤太短段落。
    """
    paras = []
    for p in re.split(r"\n\s*\n", text):
        p = " ".join(p.strip().split())  # 段内压空格
        if len(p) >= MIN_PARA_CHARS:
            paras.append(p)
    return paras


def sliding_window_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    对超长文本做滑窗切块（兜底用）。
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if len(text) >= MIN_CHUNK_CHARS else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_page_text(page_text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    段落优先：
    - 先取段落
    - 段落本身过长再滑窗切
    - 最后把相邻短段落拼接到接近 chunk_size（可选，这里做一个简单拼接）
    """
    paras = split_paragraphs(page_text)
    if not paras:
        return []

    # 简单拼接：把多个段落拼成接近 chunk_size 的块
    chunks: List[str] = []
    buf = ""

    def flush_buf():
        nonlocal buf
        b = buf.strip()
        if len(b) >= MIN_CHUNK_CHARS:
            chunks.append(b)
        buf = ""

    for p in paras:
        # 段落超长：先把当前 buf flush，再对该段落滑窗切
        if len(p) > chunk_size * 1.2:
            flush_buf()
            chunks.extend(sliding_window_chunks(p, chunk_size, overlap))
            continue

        # 正常段落：拼接到 buf
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= chunk_size:
            buf = buf + "\n\n" + p
        else:
            flush_buf()
            buf = p

    flush_buf()
    return chunks


def extract_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        cleaned = clean_page_text(raw)
        pages_text.append(cleaned)
    return pages_text


def make_chunk_id(doc_id: str, page: int, idx: int) -> str:
    return f"{doc_id}_p{page:04d}_c{idx:03d}"


def ingest(
    manifest_path: str = MANIFEST_PATH,
    out_path: str = OUTPUT_CHUNKS_PATH,
    chunk_size: int = CHUNK_CHAR_SIZE,
    overlap: int = CHUNK_CHAR_OVERLAP,
) -> Tuple[int, int]:
    ensure_parent_dir(out_path)

    manifest = read_manifest(manifest_path)
    if not manifest:
        raise RuntimeError(f"manifest is empty: {manifest_path}")

    total_docs = 0
    total_chunks = 0

    with open(out_path, "w", encoding="utf-8") as w:
        for doc in tqdm(manifest, desc="Ingest docs"):
            doc_id = doc["doc_id"]
            pdf_path = doc["file_path"]

            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Missing PDF: {pdf_path} (doc_id={doc_id})")

            pages = extract_pages(pdf_path)
            total_docs += 1

            for i, page_text in enumerate(pages):
                page_num = i + 1
                if not page_text or len(page_text) < MIN_CHUNK_CHARS:
                    continue

                chunks = chunk_page_text(page_text, chunk_size=chunk_size, overlap=overlap)
                for j, ch in enumerate(chunks, start=1):
                    rec = {
                        "chunk_id": make_chunk_id(doc_id, page_num, j),
                        "doc_id": doc_id,
                        "page": page_num,
                        "text": ch,
                        "title": doc.get("title"),
                        "source": doc.get("source"),
                        "year": doc.get("year"),
                        "doc_type": doc.get("doc_type"),
                        "jurisdiction": doc.get("jurisdiction"),
                        "language": doc.get("language"),
                        "file_path": pdf_path,
                    }
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_chunks += 1

    return total_docs, total_chunks


if __name__ == "__main__":
    docs, chunks = ingest()
    print(f"[OK] docs={docs}, chunks={chunks}")
    print(f"[OK] wrote: {OUTPUT_CHUNKS_PATH}")