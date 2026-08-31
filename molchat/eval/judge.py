"""Faithfulness judging: is the agent's answer grounded in its tool outputs?

Two backends, same interface:

- ``RuleBasedJudge`` (default, offline): deterministically checks how many of the
  tools' key facts (exact descriptor numbers, the predicted BBB label, retrieved
  neighbor names) actually appear in the answer. ``groundedness`` is the recall of
  those facts; ``coverage`` is how many tool categories are reflected. This runs
  in CI with no API key.
- ``LLMJudge`` (optional): asks a hosted LLM to rate faithfulness 1-5 given the
  question, answer, and tool results. Used only when a key is present.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List


def _trace_as_dicts(trace: List[Any]) -> List[Dict]:
    out = []
    for t in trace:
        if isinstance(t, dict):
            out.append(t)
        else:  # TraceEntry dataclass
            out.append({"tool": t.tool, "arguments": t.arguments, "result": t.result})
    return out


def _key_facts(tool: str, result: Dict) -> List[str]:
    """The strings that must appear in a faithful answer for this tool result."""
    if "error" in result:
        return []
    if tool == "rdkit_descriptors":
        return [str(result["molecular_weight"]), str(result["logp"]), str(result["tpsa"])]
    if tool == "moleco_predict":
        return [result["label"]]
    if tool == "rag_search":
        return [r["name"] for r in result.get("results", [])[:1]]  # nearest neighbor
    return []


class Judge(abc.ABC):
    backend: str = "abstract"

    @abc.abstractmethod
    def score(self, question: str, answer: str, trace: List[Any]) -> Dict: ...


class RuleBasedJudge(Judge):
    backend = "rule-based"

    def score(self, question: str, answer: str, trace: List[Any]) -> Dict:
        entries = _trace_as_dicts(trace)
        ans = answer.lower()
        expected = 0
        present = 0
        categories_present = set()
        for e in entries:
            facts = _key_facts(e["tool"], e["result"])
            for fact in facts:
                expected += 1
                if str(fact).lower() in ans:
                    present += 1
                    categories_present.add(e["tool"])
        groundedness = round(present / expected, 3) if expected else 1.0
        return {
            "backend": self.backend,
            "groundedness": groundedness,
            "coverage": len(categories_present),
            "expected_facts": expected,
            "present_facts": present,
        }


class LLMJudge(Judge):  # pragma: no cover - needs API key
    backend = "llm"

    def __init__(self, model: str = "claude-sonnet-5"):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic()

    def score(self, question: str, answer: str, trace: List[Any]) -> Dict:
        import json

        entries = _trace_as_dicts(trace)
        prompt = (
            "You are grading whether an assistant's answer is faithful to the tool "
            "outputs it was given (no invented facts, numbers match).\n\n"
            f"Question: {question}\nAnswer: {answer}\n"
            f"Tool outputs: {json.dumps(entries)}\n\n"
            "Reply with ONLY a JSON object: "
            '{"faithfulness_1_5": <int>, "reason": "<short>"}'
        )
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        try:
            data = json.loads(text)
            score_1_5 = int(data.get("faithfulness_1_5", 0))
        except Exception:
            score_1_5, data = 0, {"reason": text[:120]}
        return {
            "backend": self.backend,
            "groundedness": round(score_1_5 / 5.0, 3),
            "faithfulness_1_5": score_1_5,
            "reason": data.get("reason", ""),
        }


def get_judge(kind: str = "rule") -> Judge:
    kind = kind.lower()
    if kind in ("rule", "rule-based"):
        return RuleBasedJudge()
    if kind in ("llm", "anthropic"):
        return LLMJudge()
    raise ValueError(f"Unknown judge kind: {kind!r}")
