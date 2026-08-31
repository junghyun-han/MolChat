"""Text-document retrieval over a small chemistry/ML knowledge base.

This is the classic "document RAG" half of the system: retrieve relevant *text
passages* for a query and hand them to a language model as grounding context
(see rag_generate.py). It complements the molecular-embedding retrieval in
rag.py (structured similarity) with unstructured text retrieval.

The default backend is TF-IDF over the passages (offline, no model download),
which is a standard sparse retrieval method. A dense sentence-embedding backend
can replace it behind the same interface.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_KB = os.path.join(_DATA_DIR, "knowledge.jsonl")


@dataclass
class Passage:
    id: str
    title: str
    text: str
    score: float = 0.0


def load_passages(path: str = DEFAULT_KB) -> List[Passage]:
    passages: List[Passage] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            passages.append(Passage(id=row["id"], title=row["title"], text=row["text"]))
    return passages


class TextKnowledgeBase:
    """TF-IDF retrieval over the text passages."""

    def __init__(self, passages: Optional[List[Passage]] = None):
        self.passages = passages if passages is not None else load_passages()
        self._vectorizer = None
        self._matrix = None

    def _ensure_fitted(self) -> None:
        if self._matrix is not None:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [f"{p.title}. {p.text}" for p in self.passages]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, k: int = 3) -> List[Passage]:
        self._ensure_fitted()
        from sklearn.metrics.pairwise import linear_kernel

        q = self._vectorizer.transform([query])
        scores = linear_kernel(q, self._matrix)[0]
        order = scores.argsort()[::-1][:k]
        hits: List[Passage] = []
        for idx in order:
            p = self.passages[int(idx)]
            hits.append(
                Passage(id=p.id, title=p.title, text=p.text, score=round(float(scores[idx]), 3))
            )
        return hits

    def format_context(self, query: str, k: int = 3) -> str:
        hits = self.search(query, k=k)
        lines = ["Retrieved knowledge passages:"]
        for i, p in enumerate(hits, 1):
            lines.append(f"[{i}] {p.title}: {p.text}")
        return "\n".join(lines)
