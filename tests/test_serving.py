import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client():
    os.environ["MOLCHAT_LLM"] = "rule-based"
    from serving.app import app

    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["molecules"] >= 20


def test_ask_endpoint_runs_agent(client):
    body = client.post("/ask", json={"question": "Is aspirin BBB permeable?"}).json()
    assert "BBB permeability" in body["answer"]
    assert {t["tool"] for t in body["trace"]} == {
        "rag_search",
        "rdkit_descriptors",
        "moleco_predict",
    }


def test_predict_and_search_endpoints(client):
    p = client.post("/predict", json={"molecule": "caffeine"}).json()
    assert p["task"] == "blood_brain_barrier_permeability"
    s = client.post("/search", json={"molecule": "aspirin", "k": 3}).json()
    assert len(s["results"]) == 3


def test_invalid_molecule_returns_400(client):
    resp = client.post("/descriptors", json={"molecule": "not-a-molecule"})
    assert resp.status_code == 400
