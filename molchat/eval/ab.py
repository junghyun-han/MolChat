"""Agent A/B: does retrieval augmentation change answer quality?

Runs the same question set through the agent under two conditions — RAG on vs
RAG off (the retrieval tool withheld) — and scores every answer with a
faithfulness judge. Reports mean groundedness and coverage per condition plus
the deltas. This is the applied A/B-testing evidence: a controlled comparison of
two system configurations over a fixed evaluation set with a defined metric.
"""

from __future__ import annotations

from statistics import mean
from typing import Dict, List, Optional

from ..agent import Agent
from ..llm import RuleBasedBackend
from ..tools import Toolbox
from .judge import Judge, get_judge


def default_questions(toolbox: Toolbox, n: int = 12) -> List[str]:
    names = [row["name"] for row in toolbox.corpus][:n]
    return [
        f"Tell me about {name}, including whether it likely crosses the "
        f"blood-brain barrier and any similar molecules."
        for name in names
    ]


def _run_condition(
    toolbox: Toolbox, use_rag: bool, questions: List[str], judge: Judge
) -> Dict:
    known = [row["name"] for row in toolbox.corpus]
    agent = Agent(
        toolbox=toolbox, llm=RuleBasedBackend(known_names=known, use_rag=use_rag)
    )
    grounded, coverage, tool_counts = [], [], []
    for q in questions:
        result = agent.run(q)
        scored = judge.score(q, result.answer, result.trace)
        grounded.append(scored["groundedness"])
        coverage.append(scored["coverage"])
        tool_counts.append(len(result.trace))
    return {
        "condition": "rag" if use_rag else "no_rag",
        "n": len(questions),
        "mean_groundedness": round(mean(grounded), 3),
        "mean_coverage": round(mean(coverage), 3),
        "mean_tools_called": round(mean(tool_counts), 3),
    }


def run_agent_ab(
    questions: Optional[List[str]] = None,
    judge: Optional[Judge] = None,
    n: int = 12,
) -> Dict:
    toolbox = Toolbox()
    judge = judge or get_judge("rule")
    questions = questions or default_questions(toolbox, n=n)

    rag = _run_condition(toolbox, True, questions, judge)
    no_rag = _run_condition(toolbox, False, questions, judge)
    return {
        "judge": judge.backend,
        "conditions": [rag, no_rag],
        "delta_coverage": round(rag["mean_coverage"] - no_rag["mean_coverage"], 3),
        "delta_groundedness": round(
            rag["mean_groundedness"] - no_rag["mean_groundedness"], 3
        ),
    }
