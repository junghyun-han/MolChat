# MolChat — a RAG + function-calling agent over molecular representations

MolChat answers natural-language questions about molecules by running a
**tool-calling LLM agent** over a **retrieval-augmented** molecular knowledge
base. It is a personal engineering extension built on top of the **Moleco**
chemical-language-model project (MoLFormer + fingerprint contrastive learning,
EMNLP 2024 Industry) — used with permission. Moleco's molecular embedding is
reused here as the vectorizer that populates the retrieval index.

> **What this repo is, in one line:** not "I used an LLM once", but "I *built*
> and *ran* an LLM system — RAG, function calling, tool orchestration,
> evaluation, and deployment — on top of my own representation model."

---

## 1. Why it exists

Across the AI/ML job descriptions I mapped (Furiosa, Shinhan, SKT, Naver Cloud,
LG H&H), the most frequently required skills I had *not* yet evidenced were
**agents / function calling**, **RAG + vector databases**, **LLM
serving/quantization**, and **applied evaluation (A/B)**. MolChat is the single
end-to-end project that turns those into working, testable code, reusing the
molecular-LLM foundation I already know.

## 2. Architecture

```
                        ┌──────────────────────────────────────────┐
  "Is aspirin BBB       │                 Agent loop                │
   permeable?"  ───────▶│  LLM backend  ⇄  tool dispatch  ⇄  trace  │
                        └───────┬───────────────┬──────────────┬────┘
                                │               │              │
                     rdkit_descriptors   moleco_predict    rag_search
                       (exact RDKit)     (property model)  (retrieval)
                                                                │
                                                  ┌─────────────▼─────────────┐
                                                  │  FAISS vector store        │
                                                  │  ← molecular embeddings    │
                                                  │    (fingerprint / MoLFormer│
                                                  │     / Moleco checkpoint)   │
                                                  └────────────────────────────┘
```

- **LLM backend** is pluggable: a deterministic **rule-based** planner by default
  (no API key, runs in CI and offline), or Anthropic / OpenAI function calling
  when a key is present. See `molchat/llm.py`.
- **Embedding backend** is pluggable: RDKit **fingerprint** (default), the
  **MoLFormer-XL** chemical LM, or a drop-in **Moleco** checkpoint — all behind
  one interface. See `molchat/embeddings.py`.

## 3. Features (status)

| Layer | What | Status |
|---|---|---|
| **P1 · RAG + agent** | molecular-similarity retrieval (FAISS) + function-calling agent (3 tools) | ✅ this repo |
| **P2 · PEFT + quantization + serving** | LoRA fine-tune (Qwen2.5-0.5B) → GGUF/AWQ quantize → serve; dataset + pipeline ready | ✅ pipeline (`p2/`), Colab run pending |
| **P3 · Evaluation / A/B** | retrieval A/B (embedding vs baseline) + faithfulness judge (rule-based/LLM) + agent A/B (RAG on/off) | ✅ this repo → [`docs/EVAL.md`](docs/EVAL.md) |
| P4 · Serving spine | FastAPI + Docker + edge/ONNX inference | 🔜 roadmap |

## 4. Install & run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # rdkit, faiss-cpu, numpy

# Ask a question (rule-based backend, fully offline)
python -m molchat.cli "Is aspirin likely to cross the blood-brain barrier?"
python -m molchat.cli --trace "Describe caffeine and show similar molecules"

# Run the demo and build a persisted index
python scripts/demo.py
python scripts/build_index.py --out data/index

# P3 evaluation (retrieval A/B + agent A/B, fully offline)
python scripts/run_eval.py --out docs/EVAL.md
```

Optional hosted LLM (drives the same agent via native function calling):

```bash
pip install anthropic          # or: openai
export ANTHROPIC_API_KEY=...    # or OPENAI_API_KEY
python -m molchat.cli "Compare diazepam and caffeine for CNS penetration"
```

## 5. Example output

```
$ python -m molchat.cli "Is aspirin likely to cross the blood-brain barrier?"
[backend: rule-based | steps: 2]
Descriptors (C9H8O4, CC(=O)Oc1ccccc1C(=O)O): MW 180.16, logP 1.31, TPSA 63.6,
HBD 1, HBA 3. BBB permeability: likely BBB-permeant (score 1.0). Basis: TPSA
63.6 <= 90 (favors permeation); ... [heuristic-placeholder]
Nearest molecules by embedding: aspirin (1.0), benzoic-acid (0.546), phenol (0.431).
```

## 6. Scope & honesty

- **`rdkit_descriptors`** returns *exact* physicochemical properties (ground truth).
- **`moleco_predict`** currently uses a transparent **rule-of-thumb** BBB estimator
  derived from those descriptors — it is **not** the trained Moleco classifier and
  not ground truth. The trained model (Moleco, ROC-AUC 75.8→77.3 on BBBP in the
  paper) plugs into the same `Predictor` interface via a checkpoint (P2).
- **Embeddings** default to RDKit fingerprints so the pipeline runs anywhere; the
  MoLFormer / Moleco backends produce the learned embeddings when their weights
  are available.
- The **Moleco paper/model is a lab research artifact used with permission**; the
  RAG, agent, tooling, and serving layers in this repo are my individual
  engineering work. Everything here is local/laptop-scale validation, not a
  production deployment.

## 7. Roadmap

P1 (this repo) → **P2** LoRA + quantization + serving (Colab T4 / Mac llama.cpp)
→ **P3** evaluation & A/B (base vs RAG, LLM-as-judge) → **P4** FastAPI + Docker +
edge/ONNX on-device inference. See `docs/ARCHITECTURE.md`.

## License

Apache-2.0 (matches the upstream Moleco project).
