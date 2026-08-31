"""Retrieval-quality metrics and an embedder A/B.

No ground-truth labels are needed. The signal is *chemical clustering*: a good
molecular embedding should retrieve neighbors from the same chemical class more
often than a random baseline. We report:

- ``self_recall@1`` — sanity: does a molecule retrieve itself first?
- ``category_consistency@k`` — mean fraction of the top-k neighbors (excluding
  self) that share the query molecule's category tag.

The A/B compares the real ``fingerprint`` embedder against the ``random``
baseline; the fingerprint embedder should win decisively on category
consistency, which is the evidence that retrieval is chemically meaningful.
"""

from __future__ import annotations

from typing import Dict, List

from ..embeddings import get_embedder
from ..rag import RagIndex


def evaluate_embedder(embedder_name: str, k: int = 3) -> Dict[str, float]:
    rag = RagIndex(embedder=get_embedder(embedder_name)).build()
    corpus = rag.corpus

    self_hits = 0
    consistency_sum = 0.0
    for row in corpus:
        # k+1 because the top hit is the molecule itself.
        hits = rag.retrieve(row["smiles"], k=k + 1)
        if hits and hits[0].metadata["canonical_smiles"] == row["canonical_smiles"]:
            self_hits += 1
        neighbors = [
            h for h in hits
            if h.metadata["canonical_smiles"] != row["canonical_smiles"]
        ][:k]
        if neighbors:
            same = sum(1 for h in neighbors if h.metadata["category"] == row["category"])
            consistency_sum += same / len(neighbors)

    n = len(corpus)
    return {
        "embedder": rag.embedder.name,
        "n": n,
        "self_recall@1": round(self_hits / n, 3),
        f"category_consistency@{k}": round(consistency_sum / n, 3),
    }


def ab_embedders(
    names: List[str] = ("fingerprint", "random"), k: int = 3
) -> Dict:
    results = [evaluate_embedder(name, k=k) for name in names]
    metric = f"category_consistency@{k}"
    delta = round(results[0][metric] - results[-1][metric], 3)
    return {
        "metric": metric,
        "results": results,
        "winner": results[0]["embedder"] if delta > 0 else results[-1]["embedder"],
        "delta": delta,
    }
