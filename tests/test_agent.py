"""End-to-end agent tests using the deterministic rule-based backend (no API key)."""

import os

from molchat.agent import Agent
from molchat.llm import RuleBasedBackend


def _agent():
    # Force the offline backend so tests never depend on env keys.
    os.environ["MOLCHAT_LLM"] = "rule-based"
    return Agent(max_steps=4)


def test_agent_answers_named_molecule_and_uses_all_tools():
    agent = _agent()
    result = agent.run("Is aspirin likely to cross the blood-brain barrier?")
    tools_used = {t.tool for t in result.trace}
    assert tools_used == {"rag_search", "rdkit_descriptors", "moleco_predict"}
    assert "BBB permeability" in result.answer
    assert result.steps >= 2


def test_agent_answers_smiles_query():
    agent = _agent()
    result = agent.run("descriptors of CC(=O)Oc1ccccc1C(=O)O please")
    assert any(t.tool == "rdkit_descriptors" for t in result.trace)
    assert "MW" in result.answer


def test_agent_asks_for_molecule_when_none_found():
    agent = _agent()
    result = agent.run("hello, what can you do?")
    assert not result.trace
    assert "molecule" in result.answer.lower()


def test_rule_based_backend_extracts_known_name():
    backend = RuleBasedBackend(known_names=["aspirin", "caffeine"])
    assert backend._extract_molecule("tell me about Caffeine") == "caffeine"
    assert backend._extract_molecule("no molecule here") is None
