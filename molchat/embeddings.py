"""Pluggable molecular embedding backends.

The vector store is populated by whatever backend ``get_embedder`` returns, so
the retrieval layer is agnostic to *how* a molecule is turned into a vector.

Backends
--------
- ``fingerprint`` (default): RDKit Morgan/ECFP fingerprint folded to a dense
  float vector. Deterministic, dependency-light, and the backend used in CI.
- ``molformer`` (optional): the MoLFormer-XL chemical language model from
  Hugging Face (``ibm/MoLFormer-XL-both-10pct``), mean-pooled. This is the same
  foundation model the Moleco project builds on. Requires ``transformers`` +
  ``torch``; CPU-capable.
- ``moleco`` (drop-in): loads a Moleco fine-tuned checkpoint. This is the
  personal research artifact — the checkpoint is not shipped in this repo; the
  class documents the exact interface it plugs into.

All backends implement the same contract: ``embed(smiles) -> np.ndarray`` of
shape ``(dim,)``, L2-normalized, so cosine similarity == inner product.
"""

from __future__ import annotations

import abc
from typing import List

import numpy as np

from .descriptors import parse_smiles


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.astype("float32")
    return (vec / norm).astype("float32")


class EmbeddingBackend(abc.ABC):
    """Common interface for turning a SMILES string into a unit vector."""

    name: str = "abstract"
    dim: int = 0

    @abc.abstractmethod
    def embed(self, smiles: str) -> np.ndarray:  # pragma: no cover - abstract
        ...

    def embed_many(self, smiles_list: List[str]) -> np.ndarray:
        if not smiles_list:
            return np.zeros((0, self.dim), dtype="float32")
        return np.vstack([self.embed(s) for s in smiles_list]).astype("float32")


class FingerprintEmbedder(EmbeddingBackend):
    """Morgan (ECFP-like) fingerprint folded into a dense, normalized vector."""

    def __init__(self, n_bits: int = 2048, radius: int = 2):
        # Import here so the module imports even on unusual RDKit builds.
        from rdkit.Chem import rdFingerprintGenerator

        self.name = f"fingerprint(morgan,r={radius},bits={n_bits})"
        self.dim = n_bits
        self.radius = radius
        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )

    def embed(self, smiles: str) -> np.ndarray:
        mol = parse_smiles(smiles)
        fp = self._gen.GetFingerprint(mol)
        arr = np.zeros((self.dim,), dtype="float32")
        for bit in fp.GetOnBits():
            arr[bit] = 1.0
        return _l2_normalize(arr)


class MolformerEmbedder(EmbeddingBackend):
    """MoLFormer-XL chemical LM embedding (optional, requires transformers+torch)."""

    def __init__(self, model_name: str = "ibm/MoLFormer-XL-both-10pct"):
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional path
            raise ImportError(
                "MolformerEmbedder needs `transformers` and `torch`. "
                "Install them or use the default `fingerprint` backend."
            ) from exc

        import torch

        self.name = f"molformer({model_name})"
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self._model.eval()
        self.dim = int(self._model.config.hidden_size)

    def embed(self, smiles: str) -> np.ndarray:  # pragma: no cover - optional path
        parse_smiles(smiles)  # validate
        torch = self._torch
        with torch.no_grad():
            enc = self._tok(smiles, return_tensors="pt", padding=True, truncation=True)
            out = self._model(**enc)
            hidden = out.last_hidden_state[0]  # (seq, dim)
            mask = enc["attention_mask"][0].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(0) / mask.sum().clamp(min=1.0)
        return _l2_normalize(pooled.cpu().numpy())


class MolecoEmbedder(MolformerEmbedder):
    """Drop-in for the Moleco fine-tuned checkpoint (personal research artifact).

    Moleco = MoLFormer + fingerprint-based contrastive learning + substructure
    prediction. The checkpoint is not distributed with this portfolio; point
    ``checkpoint_path`` at a local Moleco checkpoint to use the real embeddings.
    Until then this raises with a clear message rather than silently degrading.
    """

    def __init__(self, checkpoint_path: str | None = None):
        if not checkpoint_path:
            raise NotImplementedError(
                "MolecoEmbedder requires a Moleco checkpoint. Pass checkpoint_path "
                "to load it, or use the `fingerprint` (default) / `molformer` "
                "backend. See docs/ARCHITECTURE.md."
            )
        super().__init__(model_name=checkpoint_path)
        self.name = f"moleco({checkpoint_path})"


def get_embedder(name: str = "fingerprint", **kwargs) -> EmbeddingBackend:
    name = name.lower()
    if name == "fingerprint":
        return FingerprintEmbedder(**kwargs)
    if name == "molformer":
        return MolformerEmbedder(**kwargs)
    if name == "moleco":
        return MolecoEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder backend: {name!r}")
