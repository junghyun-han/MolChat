"""Real retrieval-augmented generation: retrieve, then let an LLM generate.

Unlike the deterministic rule-based composer used for CI, this path runs an
actual language model that reads the retrieved context (text passages from the
knowledge base plus molecular neighbors and descriptors) and *generates* the
answer grounded in it. This is textbook RAG.

Generator backends (auto-selected by env MOLCHAT_GEN, else availability):
- ``transformers`` (default when installed): a small local instruct model
  (Qwen2.5-0.5B-Instruct) on MPS/CPU, no API key.
- ``anthropic`` / ``openai``: hosted models when a key is present.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

from .llm import RuleBasedBackend
from .tools import Toolbox

_SYSTEM = (
    "You are MolChat, a molecular chemistry assistant. Answer the question using "
    "ONLY the retrieved context below. Be concise and do not invent numbers. If the "
    "context is insufficient, say so."
)


def _build_context(
    tb: Toolbox, question: str, molecule: Optional[str], k_doc: int, k_mol: int
) -> Dict:
    passages = tb.knowledge_search(question, k=k_doc)["passages"]
    ctx = {"passages": passages, "molecule": None}
    if molecule:
        ctx["molecule"] = {
            "descriptors": tb.rdkit_descriptors(molecule),
            "prediction": tb.moleco_predict(molecule),
            "neighbors": tb.rag_search(molecule, k=k_mol)["results"],
        }
    return ctx


def _render_context(ctx: Dict) -> str:
    lines = ["Retrieved knowledge:"]
    for i, p in enumerate(ctx["passages"], 1):
        lines.append(f"[{i}] {p['title']}: {p['text']}")
    m = ctx["molecule"]
    if m:
        d = m["descriptors"]
        lines.append(
            f"Molecule {d['canonical_smiles']} ({d['formula']}): MW {d['molecular_weight']}, "
            f"logP {d['logp']}, TPSA {d['tpsa']}, HBD {d['h_bond_donors']}."
        )
        pred = m["prediction"]
        lines.append(f"Property estimate: {pred['label']} ({pred['backend']}).")
        neigh = ", ".join(n["name"] for n in m["neighbors"])
        lines.append(f"Similar molecules: {neigh}.")
    return "\n".join(lines)


# --- generator backends ---------------------------------------------------
def _transformers_generator(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Callable:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    def generate(system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=200, do_sample=False, pad_token_id=tok.eos_token_id
            )
        return tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

    generate.backend = f"transformers({model_name},{device})"  # type: ignore[attr-defined]
    return generate


def get_generator() -> Callable:
    choice = os.getenv("MOLCHAT_GEN", "").lower()
    if choice in ("", "transformers"):
        try:
            return _transformers_generator()
        except Exception as exc:  # pragma: no cover
            if choice == "transformers":
                raise
            print(f"[rag_generate] transformers unavailable ({exc}); no generator.")
            raise
    raise ValueError(f"Unsupported generator backend: {choice!r}")


def answer_with_rag(
    question: str,
    molecule: Optional[str] = None,
    k_doc: int = 3,
    k_mol: int = 3,
    generator: Optional[Callable] = None,
) -> Dict:
    tb = Toolbox()
    if molecule is None:
        extractor = RuleBasedBackend(known_names=[r["name"] for r in tb.corpus])
        molecule = extractor._extract_molecule(question)
    ctx = _build_context(tb, question, molecule, k_doc, k_mol)
    gen = generator or get_generator()
    prompt = f"{_render_context(ctx)}\n\nQuestion: {question}"
    answer = gen(_SYSTEM, prompt)
    return {
        "question": question,
        "answer": answer,
        "backend": getattr(gen, "backend", "unknown"),
        "retrieved": ctx,
    }
