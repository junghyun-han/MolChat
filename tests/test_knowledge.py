from molchat.knowledge import TextKnowledgeBase, load_passages
from molchat.tools import Toolbox


def test_passages_load():
    passages = load_passages()
    assert len(passages) >= 10
    assert all(p.text for p in passages)


def test_bbb_query_retrieves_bbb_passage():
    kb = TextKnowledgeBase()
    hits = kb.search("what lets a drug cross the blood brain barrier", k=1)
    assert hits[0].id == "bbb_rules"
    assert hits[0].score > 0


def test_rag_query_retrieves_rag_passage():
    kb = TextKnowledgeBase()
    ids = [p.id for p in kb.search("retrieval augmented generation with a knowledge base", k=2)]
    assert "rag" in ids


def test_knowledge_search_tool():
    tb = Toolbox()
    out = tb.dispatch("knowledge_search", {"query": "molecular fingerprint similarity", "k": 2})
    assert len(out["passages"]) == 2
    assert "fingerprint" in {p["id"] for p in out["passages"]}
