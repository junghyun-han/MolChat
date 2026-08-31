"""Real RAG demo: retrieve context, then an actual LLM generates the answer.

    python scripts/rag_demo.py --out docs/RAG_DEMO.md

Downloads a small instruct model on first run and runs it on MPS/CPU (no API key).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molchat.rag_generate import answer_with_rag, get_generator  # noqa: E402

QUESTIONS = [
    ("Is aspirin likely to cross the blood-brain barrier, and why?", "aspirin"),
    ("What makes a molecule able to cross the blood-brain barrier?", None),
    ("In one sentence, what is retrieval-augmented generation?", None),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    gen = get_generator()
    lines = ["# MolChat — real RAG generation demo", "", f"Generator backend: `{gen.backend}`", ""]
    for q, mol in QUESTIONS:
        res = answer_with_rag(q, molecule=mol, generator=gen)
        retrieved_ids = [p["id"] for p in res["retrieved"]["passages"]]
        print(f"Q: {q}\nA: {res['answer']}\n(retrieved: {retrieved_ids})\n")
        lines += [
            f"### Q: {q}",
            f"- retrieved passages: {retrieved_ids}"
            + (f", molecule: {mol}" if mol else ""),
            "",
            f"**A:** {res['answer']}",
            "",
        ]

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"[saved to {args.out}]")


if __name__ == "__main__":
    main()
