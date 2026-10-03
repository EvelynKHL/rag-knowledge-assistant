import json
import re
import hashlib
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def _extract_tabular_text(path):
    suffix = path.suffix.lower()
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to process CSV/Excel files.") from exc

    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        frame = pd.read_csv(path, sep=sep)
    elif suffix in {".xls", ".xlsx"}:
        frame = pd.read_excel(path)
    else:
        return []

    rows = []
    for _, row in frame.iterrows():
        items = [
            f"{column}: {value}"
            for column, value in row.items()
            if pd.notna(value) and str(value).strip()
        ]
        if items:
            rows.append(" | ".join(items))

    if not rows:
        return [{"page": 1, "text": ""}]

    return [{"page": 1, "text": "\n".join(rows)}]


def _extract_json_text(path):
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="ignore")

    if suffix in {".jsonl", ".ndjson"}:
        lines = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            lines.append(json.loads(line))
        payload = lines
    else:
        payload = json.loads(raw)

    if isinstance(payload, dict):
        return [{"page": 1, "text": json.dumps(payload, ensure_ascii=False, indent=2)}]

    if isinstance(payload, list):
        text = "\n\n".join(
            json.dumps(item, ensure_ascii=False, indent=2) for item in payload
        )
        return [{"page": 1, "text": text}]

    return [{"page": 1, "text": str(payload)}]


def extract_file(path):
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        import fitz
        doc = fitz.open(path)
        return [
            {"page": i + 1, "text": page.get_text("text")}
            for i, page in enumerate(doc)
        ]

    if suffix == ".docx":
        from docx import Document
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"page": 1, "text": text}]

    if suffix == ".txt":
        return [{"page": 1, "text": path.read_text(encoding="utf-8", errors="ignore")}]

    if suffix in {".csv", ".tsv", ".xls", ".xlsx"}:
        return _extract_tabular_text(path)

    if suffix in {".json", ".jsonl", ".ndjson"}:
        return _extract_json_text(path)

    raise ValueError(f"Unsupported file type: {suffix}")

def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text, chunk_size=800, overlap=120):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()

        if len(chunk) >= 40:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks

def ingest_documents(paths, index_dir):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    records = []
    seen_hashes = set()

    for path in paths:
        pages = extract_file(path)

        for page_record in pages:
            text = clean_text(page_record["text"])
            if not text:
                continue

            chunks = chunk_text(text)

            for chunk_number, chunk in enumerate(chunks, 1):
                content_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()

                if content_hash in seen_hashes:
                    continue

                seen_hashes.add(content_hash)
                records.append({
                    "id": len(records),
                    "source": path.name,
                    "page": page_record["page"],
                    "chunk": chunk_number,
                    "text": chunk,
                    "hash": content_hash,
                })

    if not records:
        raise ValueError("No usable text was extracted from the documents.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        [r["text"] for r in records],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "metadata.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "documents": len(set(r["source"] for r in records)),
        "pages": len(set((r["source"], r["page"]) for r in records)),
        "chunks": len(records),
    }
