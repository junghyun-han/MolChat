"""A tiny FAISS-backed vector store with parallel metadata.

Vectors are assumed L2-normalized (the embedding backends guarantee this), so an
inner-product index gives cosine similarity directly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import faiss
import numpy as np


@dataclass
class SearchHit:
    score: float
    metadata: Dict[str, Any]


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.metadatas)

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected vectors of shape (n, {self.dim}), got {vectors.shape}"
            )
        if vectors.shape[0] != len(metadatas):
            raise ValueError("vectors and metadatas must have the same length")
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def search(self, vector: np.ndarray, k: int = 5) -> List[SearchHit]:
        if len(self) == 0:
            return []
        query = np.ascontiguousarray(vector, dtype="float32").reshape(1, self.dim)
        k = min(k, len(self))
        scores, idxs = self.index.search(query, k)
        hits: List[SearchHit] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            hits.append(SearchHit(score=float(score), metadata=self.metadatas[int(idx)]))
        return hits

    # --- persistence -----------------------------------------------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.json"), "w") as fh:
            json.dump({"dim": self.dim, "metadatas": self.metadatas}, fh)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        with open(os.path.join(directory, "metadata.json")) as fh:
            payload = json.load(fh)
        store = cls(dim=int(payload["dim"]))
        store.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        store.metadatas = list(payload["metadatas"])
        return store
