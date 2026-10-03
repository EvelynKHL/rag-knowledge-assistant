import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_dir):
        self.index_dir = Path(index_dir)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.metadata = []

        index_path = self.index_dir / "index.faiss"
        metadata_path = self.index_dir / "metadata.json"

        if index_path.exists() and metadata_path.exists():
            self.index = faiss.read_index(str(index_path))
            self.metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )

    def search(self, query, k=5):
        if self.index is None or not self.metadata:
            return []

        vector = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")

        k = min(k, len(self.metadata))
        scores, indices = self.index.search(vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[int(idx)])
            item["score"] = float(score)
            results.append(item)

        return results
