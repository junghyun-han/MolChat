"""The tool-calling agent loop.

Provider-agnostic: given a question, it repeatedly asks the LLM backend for the
next step, executes any requested tool calls against the ``Toolbox``, feeds the
results back, and stops when the backend returns a final answer. The full
tool-call trace is returned alongside the answer for transparency/eval.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm import LLMBackend, get_llm
from .tools import Toolbox


@dataclass
class TraceEntry:
    tool: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]


@dataclass
class AgentResult:
    answer: str
    trace: List[TraceEntry] = field(default_factory=list)
    steps: int = 0


class Agent:
    def __init__(
        self,
        toolbox: Optional[Toolbox] = None,
        llm: Optional[LLMBackend] = None,
        max_steps: int = 4,
    ):
        self.toolbox = toolbox or Toolbox()
        known = [row["name"] for row in self.toolbox.corpus]
        self.llm = llm or get_llm(known_names=known)
        self.max_steps = max_steps

    def run(self, question: str) -> AgentResult:
        messages: List[Dict[str, Any]] = [{"role": "user", "content": question}]
        trace: List[TraceEntry] = []

        for step in range(1, self.max_steps + 1):
            turn = self.llm.step(messages, self.toolbox.specs())

            if turn.is_final:
                return AgentResult(answer=turn.final_text, trace=trace, steps=step)

            if not turn.tool_calls:
                break

            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": c.id or _new_id(), "name": c.name, "arguments": c.arguments}
                        for c in turn.tool_calls
                    ],
                }
            )
            for call in turn.tool_calls:
                call_id = call.id or _new_id()
                try:
                    result = self.toolbox.dispatch(call.name, call.arguments)
                except Exception as exc:  # surface tool errors to the model
                    result = {"error": str(exc)}
                trace.append(TraceEntry(call.name, call.arguments, result))
                messages.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "tool_call_id": call_id,
                        "content": result,
                    }
                )

        # Ran out of steps without a final answer: compose from the trace.
        return AgentResult(
            answer="Reached the step limit without a final answer.",
            trace=trace,
            steps=self.max_steps,
        )


def _new_id() -> str:
    return uuid.uuid4().hex[:8]
