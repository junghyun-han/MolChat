"""Synthesize a molecular instruction-tuning dataset from the corpus.

The answers are *grounded* in MolChat's own tools (exact RDKit descriptors, the
BBB predictor, and embedding-based retrieval), so the fine-tuned model (P2) is
distilled from faithful, reproducible tool outputs rather than free-form text.

Output format: chat messages, one example per line as JSON:
    {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
which is the standard shape for SFT of instruct models (e.g. TRL SFTTrainer).

Splitting is done by *molecule* (a held-out set of molecules never appears in
train), so the eval split measures generalization to unseen molecules.
"""

from __future__ import annotations

from typing import Dict, List

from .tools import Toolbox


def _example(user: str, assistant: str) -> Dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def examples_for_molecule(tb: Toolbox, name: str) -> List[Dict]:
    """All single-molecule QA pairs for one molecule, grounded in tool outputs."""
    d = tb.rdkit_descriptors(name)
    p = tb.moleco_predict(name)
    r = tb.rag_search(name, k=3)
    out: List[Dict] = []

    # 1. Identity / SMILES
    out.append(
        _example(
            f"What is the SMILES and molecular formula of {name}?",
            f"{name.capitalize()} has SMILES {d['canonical_smiles']} and molecular "
            f"formula {d['formula']}.",
        )
    )
    # 2. Single descriptor
    out.append(
        _example(
            f"What is the molecular weight of {name}?",
            f"The molecular weight of {name} is {d['molecular_weight']} g/mol.",
        )
    )
    # 3. Full physicochemical profile
    out.append(
        _example(
            f"Describe the physicochemical properties of {name}.",
            f"{name.capitalize()} ({d['formula']}, {d['canonical_smiles']}) has "
            f"molecular weight {d['molecular_weight']} g/mol, logP {d['logp']}, "
            f"TPSA {d['tpsa']} A^2, {d['h_bond_donors']} H-bond donors, and "
            f"{d['h_bond_acceptors']} H-bond acceptors.",
        )
    )
    # 4. BBB permeability with rationale
    out.append(
        _example(
            f"Is {name} likely to cross the blood-brain barrier? Explain briefly.",
            f"{name.capitalize()} is {p['label']}. Reasoning: {p['rationale']}. "
            f"(Rule-of-thumb estimate from physicochemical descriptors.)",
        )
    )
    # 5. Similarity / retrieval
    neigh = ", ".join(f"{x['name']} (similarity {x['similarity']})" for x in r["results"])
    out.append(
        _example(
            f"Which molecules are most similar to {name}?",
            f"By molecular-fingerprint embedding, the closest molecules to {name} "
            f"are: {neigh}.",
        )
    )
    return out


def comparison_examples(tb: Toolbox, name_a: str, name_b: str) -> List[Dict]:
    da = tb.rdkit_descriptors(name_a)
    db = tb.rdkit_descriptors(name_b)
    more = name_a if da["logp"] > db["logp"] else name_b
    return [
        _example(
            f"Compare the lipophilicity (logP) of {name_a} and {name_b}.",
            f"{name_a.capitalize()} has logP {da['logp']} and {name_b} has logP "
            f"{db['logp']}, so {more} is the more lipophilic of the two.",
        )
    ]


def build_dataset(holdout: int = 6) -> Dict[str, List[Dict]]:
    """Return {'train': [...], 'eval': [...]} split by molecule."""
    tb = Toolbox()
    names = [row["name"] for row in tb.corpus]
    eval_names = set(names[-holdout:])
    train_names = names[:-holdout]

    train: List[Dict] = []
    eval_: List[Dict] = []
    for nm in names:
        bucket = eval_ if nm in eval_names else train
        bucket.extend(examples_for_molecule(tb, nm))

    # Comparison pairs only among train molecules (adjacent pairs, deterministic).
    for a, b in zip(train_names, train_names[1:]):
        train.extend(comparison_examples(tb, a, b))

    return {"train": train, "eval": eval_}
