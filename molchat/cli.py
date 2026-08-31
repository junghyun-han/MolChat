"""Command-line entry point.

    python -m molchat.cli "Is aspirin likely to cross the blood-brain barrier?"
    python -m molchat.cli --trace "Describe caffeine"

Uses whatever LLM backend the environment selects (rule-based by default).
"""

from __future__ import annotations

import argparse
import json
import sys

from .agent import Agent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MolChat — molecular Q&A agent")
    parser.add_argument("question", nargs="+", help="natural-language question")
    parser.add_argument(
        "--trace", action="store_true", help="print the tool-call trace as JSON"
    )
    args = parser.parse_args(argv)

    agent = Agent()
    result = agent.run(" ".join(args.question))

    print(f"[backend: {agent.llm.name} | steps: {result.steps}]")
    print(result.answer)
    if args.trace:
        print("\n--- trace ---")
        for entry in result.trace:
            print(
                json.dumps(
                    {"tool": entry.tool, "arguments": entry.arguments, "result": entry.result},
                    ensure_ascii=False,
                    indent=2,
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
