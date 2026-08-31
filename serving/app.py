"""FastAPI service exposing MolChat as an HTTP API (P4 serving spine).

Endpoints
---------
- GET  /health              liveness + which LLM backend / corpus size
- POST /ask       {question} run the tool-calling agent, return answer + trace
- POST /predict   {molecule} BBB permeability prediction
- POST /search    {molecule, k} molecular-similarity retrieval
- POST /descriptors {molecule} exact RDKit descriptors

Run locally:  uvicorn serving.app:app --reload
The agent uses whatever LLM backend the environment selects (rule-based default),
so the service runs with no API key.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from molchat.agent import Agent
from molchat.tools import Toolbox, ToolError

app = FastAPI(title="MolChat", version="0.1.0")


@lru_cache(maxsize=1)
def _agent() -> Agent:
    # Built once (index + corpus + backend) and reused across requests.
    return Agent()


@lru_cache(maxsize=1)
def _toolbox() -> Toolbox:
    return _agent().toolbox


class AskRequest(BaseModel):
    question: str


class MoleculeRequest(BaseModel):
    molecule: str
    k: Optional[int] = 5


@app.get("/health")
def health():
    agent = _agent()
    return {
        "status": "ok",
        "llm_backend": agent.llm.name,
        "molecules": len(agent.toolbox.corpus),
    }


@app.post("/ask")
def ask(req: AskRequest):
    result = _agent().run(req.question)
    return {
        "answer": result.answer,
        "steps": result.steps,
        "trace": [
            {"tool": t.tool, "arguments": t.arguments, "result": t.result}
            for t in result.trace
        ],
    }


def _tool(name: str, **kwargs):
    try:
        return _toolbox().dispatch(name, kwargs)
    except ToolError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict")
def predict(req: MoleculeRequest):
    return _tool("moleco_predict", molecule=req.molecule)


@app.post("/search")
def search(req: MoleculeRequest):
    return _tool("rag_search", molecule=req.molecule, k=req.k or 5)


@app.post("/descriptors")
def descriptors(req: MoleculeRequest):
    return _tool("rdkit_descriptors", molecule=req.molecule)
