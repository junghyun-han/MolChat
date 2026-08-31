from molchat.agent import Agent
from molchat.eval.ab import run_agent_ab
from molchat.eval.judge import RuleBasedJudge
from molchat.eval.retrieval import ab_embedders, evaluate_embedder
from molchat.llm import RuleBasedBackend


def test_fingerprint_beats_random_on_category_consistency():
    ab = ab_embedders(["fingerprint", "random"], k=3)
    assert ab["winner"].startswith("fingerprint")
    assert ab["delta"] > 0.1  # chemically meaningful, decisively above baseline


def test_self_recall_is_perfect_for_fingerprint():
    m = evaluate_embedder("fingerprint", k=3)
    assert m["self_recall@1"] == 1.0


def test_rule_based_judge_rewards_grounded_answer():
    known = ["aspirin"]
    agent = Agent(llm=RuleBasedBackend(known_names=known, use_rag=True))
    result = agent.run("Tell me about aspirin and similar molecules.")
    scored = RuleBasedJudge().score("about aspirin", result.answer, result.trace)
    assert scored["groundedness"] >= 0.8
    assert scored["coverage"] == 3  # descriptors + prediction + retrieval


def test_agent_ab_rag_increases_coverage():
    ab = run_agent_ab(n=8)
    conds = {c["condition"]: c for c in ab["conditions"]}
    assert conds["rag"]["mean_coverage"] > conds["no_rag"]["mean_coverage"]
    assert ab["delta_coverage"] > 0
    # Both conditions should remain faithful to their own tools.
    assert conds["rag"]["mean_groundedness"] >= 0.8
    assert conds["no_rag"]["mean_groundedness"] >= 0.8
