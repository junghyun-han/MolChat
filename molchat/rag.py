"""Retrieval-augmented layer: molecular-similarity search over the corpus.

``RagIndex`` embeds every corpus molecule with the chosen embedding backend,
stores the vectors in a FAISS ``VectorStore``, and retrieves the nearest
molecules for a query SMILES. This is the ``rag_search`` tool's engine and the
"R" in the agent's RAG loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .corpus import load_corpus
from .embeddings import EmbeddingBackend, get_embedder
from .vectorstore import SearchHit, VectorStore


class RagIndex:
    def __init__(self, embedder: Optional[EmbeddingBackend] = None):
        self.embedder = embedder or get_embedder("fingerprint")
        self.store: Optional[VectorStore] = None
        self.corpus: List[Dict] = []

    def build(self, corpus: Optional[List[Dict]] = None) -> "RagIndex":
        self.corpus = corpus if corpus is not None else load_corpus()
        smiles = [row["smiles"] for row in self.corpus]
        vectors = self.embedder.embed_many(smiles)
        self.store = VectorStore(dim=self.embedder.dim)
        self.store.add(vectors, self.corpus)
        return self

    def _ensure_built(self) -> None:
        if self.store is None:
            self.build()

    def retrieve(self, smiles: str, k: int = 5) -> List[SearchHit]:
        self._ensure_built()
        query_vec = self.embedder.embed(smiles)
        assert self.store is not None
        return self.store.search(query_vec, k=k)

    def format_context(self, smiles: str, k: int = 5) -> str:
        """Human/LLM-readable retrieval block for the query molecule."""
        hits = self.retrieve(smiles, k=k)
        lines = [f"Top {len(hits)} similar molecules (embedder: {self.embedder.name}):"]
        for rank, hit in enumerate(hits, 1):
            m = hit.metadata
            d = m["descriptors"]
            lines.append(
                f"  {rank}. {m['name']} [{m['category']}] "
                f"sim={hit.score:.3f} | {m['canonical_smiles']} | "
                f"MW={d['molecular_weight']}, logP={d['logp']}, TPSA={d['tpsa']}"
            )
        return "\n".join(lines)
