"""Pluggable LLM backends for the tool-calling agent.

The agent runs a provider-agnostic loop (see agent.py). A backend's job is,
given the running message list and the tool schemas, to return the next
``Turn``: either a set of tool calls to execute, or a final natural-language
answer.

Backends
--------
- ``RuleBasedBackend`` (default): deterministic planner — no API key, no network.
  It resolves the target molecule, calls the three tools, then composes an answer
  from their outputs. This keeps the whole pipeline runnable in CI and offline
  while still exercising the real RAG + tool-dispatch machinery.
- ``AnthropicBackend`` / ``OpenAIBackend`` (optional): drive the loop with a
  hosted model's native function-calling. Selected automatically when the
  matching API key is present.

``get_llm`` picks a backend from the environment.
"""

from __future__ import annotations

import abc
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    id: str = ""


@dataclass
class Turn:
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_text: Optional[str] = None

    @property
    def is_final(self) -> bool:
        return self.final_text is not None


class LLMBackend(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def step(self, messages: List[Dict[str, Any]], tools: List[Dict]) -> Turn: ...


# --------------------------------------------------------------------------
# Rule-based (default, offline, deterministic)
# --------------------------------------------------------------------------
_SMILES_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#/\\%.]{3,}$")


class RuleBasedBackend(LLMBackend):
    name = "rule-based"

    def __init__(self, known_names: Optional[List[str]] = None):
        self.known_names = sorted(known_names or [], key=len, reverse=True)

    def _extract_molecule(self, question: str) -> Optional[str]:
        q = question.lower()
        for nm in self.known_names:
            variants = {nm, nm.replace("-", " ")}
            for v in variants:
                if re.search(rf"\b{re.escape(v)}\b", q):
                    return nm
        for token in re.split(r"[\s,;?!]+", question.strip()):
            token = token.strip(".")
            if _SMILES_RE.match(token) and any(c in token for c in "()=[]#123456"):
                return token
        return None

    def step(self, messages: List[Dict[str, Any]], tools: List[Dict]) -> Turn:
        question = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        already_called = any(m.get("role") == "tool" for m in messages)

        if not already_called:
            molecule = self._extract_molecule(question)
            if molecule is None:
                return Turn(
                    final_text=(
                        "I couldn't find a molecule in the question. Please give a "
                        "molecule name (e.g. aspirin) or a SMILES string."
                    )
                )
            return Turn(
                tool_calls=[
                    ToolCall("rag_search", {"molecule": molecule, "k": 3}),
                    ToolCall("rdkit_descriptors", {"molecule": molecule}),
                    ToolCall("moleco_predict", {"molecule": molecule}),
                ]
            )

        results = {
            m["name"]: m["content"]
            for m in messages
            if m.get("role") == "tool" and isinstance(m.get("content"), dict)
        }
        return Turn(final_text=_compose_answer(question, results))


def _compose_answer(question: str, results: Dict[str, Dict]) -> str:
    parts: List[str] = []
    desc = results.get("rdkit_descriptors")
    pred = results.get("moleco_predict")
    rag = results.get("rag_search")

    if desc:
        parts.append(
            f"Descriptors ({desc['formula']}, {desc['canonical_smiles']}): "
            f"MW {desc['molecular_weight']}, logP {desc['logp']}, "
            f"TPSA {desc['tpsa']}, HBD {desc['h_bond_donors']}, "
            f"HBA {desc['h_bond_acceptors']}."
        )
    if pred:
        parts.append(
            f"BBB permeability: {pred['label']} (score {pred['score']}). "
            f"Basis: {pred['rationale']}. [{pred['backend']}]"
        )
    if rag and rag["results"]:
        neigh = ", ".join(
            f"{r['name']} ({r['similarity']})" for r in rag["results"]
        )
        parts.append(f"Nearest molecules by embedding: {neigh}.")

    if not parts:
        return "No tool results were available to answer the question."
    return " ".join(parts)


# --------------------------------------------------------------------------
# Optional hosted backends (only used when a key is present)
# --------------------------------------------------------------------------
class AnthropicBackend(LLMBackend):  # pragma: no cover - needs API key
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5"):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic()

    def step(self, messages: List[Dict[str, Any]], tools: List[Dict]) -> Turn:
        tool_defs = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]
        api_messages = _to_anthropic_messages(messages)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            tools=tool_defs,
            messages=api_messages,
        )
        calls = [
            ToolCall(name=b.name, arguments=b.input, id=b.id)
            for b in resp.content
            if getattr(b, "type", None) == "tool_use"
        ]
        if calls:
            return Turn(tool_calls=calls)
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
        return Turn(final_text=text or "(no answer)")


class OpenAIBackend(LLMBackend):  # pragma: no cover - needs API key
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        import openai

        self.model = model
        self._client = openai.OpenAI()

    def step(self, messages: List[Dict[str, Any]], tools: List[Dict]) -> Turn:
        tool_defs = [{"type": "function", "function": t} for t in tools]
        resp = self._client.chat.completions.create(
            model=self.model,
            tools=tool_defs,
            messages=_to_openai_messages(messages),
        )
        choice = resp.choices[0].message
        if choice.tool_calls:
            calls = [
                ToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                    id=tc.id,
                )
                for tc in choice.tool_calls
            ]
            return Turn(tool_calls=calls)
        return Turn(final_text=choice.content or "(no answer)")


def _to_anthropic_messages(messages):  # pragma: no cover - needs API key
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.get("tool_call_id", ""),
                            "content": json.dumps(m["content"]),
                        }
                    ],
                }
            )
    return out


def _to_openai_messages(messages):  # pragma: no cover - needs API key
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.get("tool_call_id", ""),
                    "content": json.dumps(m["content"]),
                }
            )
    return out


def get_llm(known_names: Optional[List[str]] = None) -> LLMBackend:
    """Auto-select a backend from the environment.

    Priority: MOLCHAT_LLM env override -> Anthropic key -> OpenAI key ->
    deterministic rule-based fallback.
    """
    override = os.getenv("MOLCHAT_LLM", "").lower()
    if override == "rule-based":
        return RuleBasedBackend(known_names)
    if override == "anthropic" or (not override and os.getenv("ANTHROPIC_API_KEY")):
        try:
            return AnthropicBackend()
        except Exception:
            pass
    if override == "openai" or (not override and os.getenv("OPENAI_API_KEY")):
        try:
            return OpenAIBackend()
        except Exception:
            pass
    return RuleBasedBackend(known_names)
