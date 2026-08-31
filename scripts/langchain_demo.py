"""LangChain demo: MolChat tools + retriever + an LCEL RAG chain, real LLM.

    python scripts/langchain_demo.py --out docs/LANGCHAIN_DEMO.md

Shows the four MolChat tools as LangChain StructuredTools and runs a LangChain
LCEL RAG chain (retriever -> prompt -> local LLM -> parser). Downloads a small
model on first run (MPS/CPU, no API key).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molchat.integrations.langchain_tools import (  # noqa: E402
    build_langchain_tools,
    build_rag_chain,
)

QUESTIONS = [
    "What makes a molecule able to cross the blood-brain barrier?",
    "In one sentence, what is retrieval-augmented generation?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tools = build_langchain_tools()
    chain = build_rag_chain(k=3)

    lines = ["# MolChat — LangChain integration demo", ""]
    lines.append("LangChain StructuredTools: " + ", ".join(f"`{t.name}`" for t in tools))
    lines.append("")
    lines.append("LCEL chain: retriever -> prompt -> local LLM -> parser")
    lines.append("")
    for q in QUESTIONS:
        ans = chain.invoke(q)
        print(f"Q: {q}\nA: {ans}\n")
        lines += [f"### Q: {q}", "", f"**A:** {ans}", ""]

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"[saved to {args.out}]")


if __name__ == "__main__":
    main()
