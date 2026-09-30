"""Exact cosine retrieval on a small corpus; no vector server needed."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from modelforge.data import passages


def normalize_text(text: str) -> str:
    """Normalize whitespace without deleting technical symbols or case information."""
    return re.sub(r"\s+", " ", text).strip()


def cache_key(rows: list[dict], model_name: str) -> str:
    payload = json.dumps(
        {"model": model_name, "texts": [(r["id"], r["text"]) for r in rows]},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


class Retriever:
    def __init__(
        self,
        data_dir: str | Path,
        method: str = "lexical",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str | Path = "outputs/cache",
    ) -> None:
        if method not in {"lexical", "dense"}:
            raise ValueError("method must be lexical or dense")
        self.rows = passages(data_dir)
        ids = [row["id"] for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate passage IDs")
        self.method = method
        self.embedding_model = embedding_model
        self.texts = [normalize_text(row["title"] + ". " + row["text"]) for row in self.rows]
        if method == "lexical":
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self.matrix = self.vectorizer.fit_transform(self.texts)
        else:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(embedding_model)
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"passages-{cache_key(self.rows, embedding_model)}.npz"
            if path.exists():
                with np.load(path) as archive:
                    self.matrix = archive["embeddings"]
                if self.matrix.shape[0] != len(self.rows):
                    raise ValueError("embedding cache row count mismatch; delete the cache")
            else:
                self.matrix = np.asarray(
                    self.encoder.encode(self.texts, normalize_embeddings=True), dtype=np.float32
                )
                np.savez_compressed(path, embeddings=self.matrix)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip() or top_k < 1:
            raise ValueError("nonempty query and positive top_k required")
        if self.method == "lexical":
            query_vector = self.vectorizer.transform([normalize_text(query)])
            scores = (self.matrix @ query_vector.T).toarray().ravel()
        else:
            vector = np.asarray(
                self.encoder.encode([normalize_text(query)], normalize_embeddings=True),
                dtype=np.float32,
            )[0]
            scores = self.matrix @ vector
        indices = sorted(range(len(self.rows)), key=lambda index: (-float(scores[index]), self.rows[index]["id"]))
        return [
            {"id": self.rows[i]["id"], "title": self.rows[i]["title"],
             "text": self.rows[i]["text"], "score": float(scores[i])}
            for i in indices[:top_k]
        ]
