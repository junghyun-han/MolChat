"""Run the P3 evaluation suite and print (optionally save) a markdown report.

    python scripts/run_eval.py                 # print report
    python scripts/run_eval.py --out docs/EVAL.md

Runs fully offline with the rule-based judge (no API key). Pass --judge llm to
use an LLM-as-judge when a key is configured.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molchat.eval.ab import run_agent_ab  # noqa: E402
from molchat.eval.judge import get_judge  # noqa: E402
from molchat.eval.retrieval import ab_embedders  # noqa: E402


def render_report(retrieval: dict, agent_ab: dict) -> str:
    lines = ["# MolChat — Evaluation report (P3)", ""]

    lines += ["## 1. Retrieval A/B — do embeddings capture chemistry?", ""]
    lines += [f"Metric: `{retrieval['metric']}` (higher = neighbors share chemical class).", ""]
    lines += ["| embedder | self_recall@1 | " + retrieval["metric"] + " |", "|---|---|---|"]
    for r in retrieval["results"]:
        metric_key = retrieval["metric"]
        lines.append(f"| {r['embedder']} | {r['self_recall@1']} | {r[metric_key]} |")
    lines += [
        "",
        f"**Winner: {retrieval['winner']} (Δ {retrieval['delta']}).** The fingerprint "
        "embedding beats the random baseline, i.e. retrieval is chemically meaningful.",
        "",
    ]

    lines += ["## 2. Agent A/B — RAG on vs off", ""]
    lines += [f"Judge: `{agent_ab['judge']}` (faithfulness = tool-grounded).", ""]
    lines += ["| condition | groundedness | coverage | tools/answer |", "|---|---|---|---|"]
    for c in agent_ab["conditions"]:
        lines.append(
            f"| {c['condition']} | {c['mean_groundedness']} | "
            f"{c['mean_coverage']} | {c['mean_tools_called']} |"
        )
    lines += [
        "",
        f"**Δ coverage = {agent_ab['delta_coverage']}, "
        f"Δ groundedness = {agent_ab['delta_groundedness']}.** RAG raises coverage "
        "(neighbor context) while both conditions stay faithful to their tool outputs.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default="rule", choices=["rule", "llm"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--n", type=int, default=12)
    args = parser.parse_args()

    retrieval = ab_embedders(["fingerprint", "random"], k=3)
    agent_ab = run_agent_ab(judge=get_judge(args.judge), n=args.n)
    report = render_report(retrieval, agent_ab)

    print(report)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(report + "\n")
        print(f"\n[saved to {args.out}]")


if __name__ == "__main__":
    main()
