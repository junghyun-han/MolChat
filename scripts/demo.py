"""End-to-end demo: run the agent on a few molecular questions and print traces.

    python scripts/demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molchat.agent import Agent  # noqa: E402

QUESTIONS = [
    "Is aspirin likely to cross the blood-brain barrier?",
    "Describe caffeine and show me similar molecules.",
    "What are the descriptors of CC(=O)Oc1ccccc1C(=O)O?",
    "Compare diazepam's BBB permeability.",
]


def main() -> None:
    agent = Agent()
    print(f"LLM backend: {agent.llm.name}\n")
    for q in QUESTIONS:
        result = agent.run(q)
        print(f"Q: {q}")
        print(f"A: {result.answer}")
        print(f"   (tools used: {[t.tool for t in result.trace]})\n")


if __name__ == "__main__":
    main()
