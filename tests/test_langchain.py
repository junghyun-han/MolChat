"""LangChain integration tests (skipped unless langchain-core is installed).

Uses a stub LLM so no model is downloaded; verifies the tools are real LangChain
StructuredTools, the retriever is a BaseRetriever, and the LCEL chain runs.
"""

import pytest

pytest.importorskip("langchain_core")

from langchain_core.retrievers import BaseRetriever  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402

from molchat.integrations.langchain_tools import (  # noqa: E402
    MolChatRetriever,
    build_langchain_tools,
    build_rag_chain,
)
from molchat.knowledge import TextKnowledgeBase  # noqa: E402


def test_tools_are_langchain_structured_tools():
    tools = build_langchain_tools()
    assert {t.name for t in tools} == {
        "rdkit_descriptors",
        "moleco_predict",
        "rag_search",
        "knowledge_search",
    }
    assert all(isinstance(t, StructuredTool) for t in tools)
    out = next(t for t in tools if t.name == "rdkit_descriptors").invoke({"molecule": "aspirin"})
    assert "C9H8O4" in out


def test_retriever_is_base_retriever():
    r = MolChatRetriever(kb=TextKnowledgeBase(), k=2)
    assert isinstance(r, BaseRetriever)
    docs = r.invoke("blood brain barrier permeability")
    assert docs[0].metadata["id"] == "bbb_rules"


def test_lcel_rag_chain_runs_with_stub_llm():
    captured = {}

    def stub(system, user):
        captured["user"] = user
        return "STUB ANSWER"

    chain = build_rag_chain(generator=stub, k=2)
    out = chain.invoke("What makes a molecule cross the blood-brain barrier?")
    assert out == "STUB ANSWER"
    # the retrieved passage text reached the prompt
    assert "polar surface area" in captured["user"].lower()
