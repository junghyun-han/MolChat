"""Function-calling tool registry.

Each tool has an OpenAI/Anthropic-style JSON schema (so it can be handed to a
hosted LLM's tool-use API, or wrapped as a LangChain tool) plus a Python
implementation. ``Toolbox.dispatch`` runs a tool by name with validated args.

Every tool accepts a ``molecule`` argument that may be either a SMILES string
or a known molecule name from the corpus; resolution is centralized so the LLM
never has to know SMILES.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .corpus import load_corpus, name_to_smiles
from .descriptors import compute_descriptors, parse_smiles
from .knowledge import TextKnowledgeBase
from .predictor import Predictor, get_predictor
from .rag import RagIndex


class ToolError(ValueError):
    """Raised when a tool is called with arguments it cannot use."""


class Toolbox:
    def __init__(
        self,
        rag: Optional[RagIndex] = None,
        predictor: Optional[Predictor] = None,
    ):
        self.corpus = load_corpus()
        self._names = name_to_smiles(self.corpus)
        self.rag = rag or RagIndex().build(self.corpus)
        self.predictor = predictor or get_predictor("heuristic")
        self.knowledge = TextKnowledgeBase()

    # --- molecule resolution --------------------------------------------
    def resolve(self, molecule: str) -> str:
        """Resolve a name or SMILES to a SMILES string."""
        if not molecule or not molecule.strip():
            raise ToolError("empty molecule argument")
        key = molecule.strip().lower()
        if key in self._names:
            return self._names[key]
        if parse_smiles_or_none(molecule) is not None:
            return molecule.strip()
        raise ToolError(
            f"could not resolve {molecule!r} as a known name or a valid SMILES"
        )

    # --- tool implementations -------------------------------------------
    def rdkit_descriptors(self, molecule: str) -> Dict[str, Any]:
        return compute_descriptors(self.resolve(molecule))

    def moleco_predict(self, molecule: str) -> Dict[str, Any]:
        return self.predictor.predict_bbb(self.resolve(molecule))

    def rag_search(self, molecule: str, k: int = 5) -> Dict[str, Any]:
        smiles = self.resolve(molecule)
        hits = self.rag.retrieve(smiles, k=int(k))
        return {
            "query_smiles": smiles,
            "embedder": self.rag.embedder.name,
            "results": [
                {
                    "name": h.metadata["name"],
                    "category": h.metadata["category"],
                    "smiles": h.metadata["canonical_smiles"],
                    "similarity": round(h.score, 3),
                }
                for h in hits
            ],
        }

    def knowledge_search(self, query: str, k: int = 3) -> Dict[str, Any]:
        hits = self.knowledge.search(query, k=int(k))
        return {
            "query": query,
            "passages": [
                {"id": p.id, "title": p.title, "text": p.text, "score": p.score}
                for p in hits
            ],
        }

    # --- dispatch + schemas ---------------------------------------------
    def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name == "rdkit_descriptors":
            return self.rdkit_descriptors(**arguments)
        if name == "moleco_predict":
            return self.moleco_predict(**arguments)
        if name == "rag_search":
            return self.rag_search(**arguments)
        if name == "knowledge_search":
            return self.knowledge_search(**arguments)
        raise ToolError(f"unknown tool: {name!r}")

    def specs(self) -> List[Dict[str, Any]]:
        return TOOL_SPECS


def parse_smiles_or_none(smiles: str):
    try:
        return parse_smiles(smiles)
    except Exception:
        return None


TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "rdkit_descriptors",
        "description": (
            "Compute exact physicochemical descriptors (MW, logP, TPSA, H-bond "
            "donors/acceptors, rings, formula) for a molecule."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "molecule": {
                    "type": "string",
                    "description": "A SMILES string or a known molecule name.",
                }
            },
            "required": ["molecule"],
        },
    },
    {
        "name": "moleco_predict",
        "description": (
            "Predict blood-brain-barrier (BBB) permeability for a molecule and "
            "return a label with rationale."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "molecule": {
                    "type": "string",
                    "description": "A SMILES string or a known molecule name.",
                }
            },
            "required": ["molecule"],
        },
    },
    {
        "name": "rag_search",
        "description": (
            "Retrieve the most similar molecules from the corpus using molecular "
            "embeddings (retrieval-augmented context)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "molecule": {
                    "type": "string",
                    "description": "A SMILES string or a known molecule name.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of neighbors to return (default 5).",
                },
            },
            "required": ["molecule"],
        },
    },
    {
        "name": "knowledge_search",
        "description": (
            "Retrieve relevant text passages from the chemistry/ML knowledge base "
            "(document retrieval for grounding a generated answer)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language query.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of passages to return (default 3).",
                },
            },
            "required": ["query"],
        },
    },
]
