"""Tests the RAG-generation assembly with a stub generator (no model download)."""

from molchat.rag_generate import answer_with_rag


def stub_generator(system, user):
    # Echo back the prompt so we can assert what context the LLM received.
    stub_generator.last_prompt = user
    return "STUB ANSWER"


def test_answer_with_rag_retrieves_and_calls_generator():
    res = answer_with_rag(
        "Is aspirin likely to cross the blood-brain barrier?",
        molecule="aspirin",
        generator=stub_generator,
    )
    assert res["answer"] == "STUB ANSWER"
    # text passages were retrieved and placed in the prompt
    ids = [p["id"] for p in res["retrieved"]["passages"]]
    assert "bbb_rules" in ids
    # molecule context (descriptors + neighbors) was assembled
    assert res["retrieved"]["molecule"] is not None
    assert "aspirin" in stub_generator.last_prompt.lower() or "C9H8O4" in stub_generator.last_prompt


def test_answer_with_rag_extracts_molecule_from_question():
    res = answer_with_rag(
        "Tell me about caffeine.", generator=lambda s, u: "ok"
    )
    assert res["retrieved"]["molecule"] is not None  # caffeine auto-extracted
