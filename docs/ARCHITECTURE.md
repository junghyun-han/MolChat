# MolChat — Architecture

## Modules

| Module | Responsibility |
|---|---|
| `molchat/descriptors.py` | Exact RDKit physicochemical descriptors; SMILES parsing/validation. |
| `molchat/embeddings.py` | Pluggable embedding backends (`fingerprint` / `molformer` / `moleco`), all L2-normalized so cosine == inner product. |
| `molchat/vectorstore.py` | FAISS `IndexFlatIP` + parallel metadata; add / search / save / load. |
| `molchat/corpus.py` | Loads the bundled molecule corpus, enriches with descriptors, builds a name→SMILES lookup. |
| `molchat/rag.py` | `RagIndex`: embeds the corpus, retrieves nearest molecules for a query. |
| `molchat/predictor.py` | Property prediction (`Predictor` interface): heuristic BBB estimator now, Moleco checkpoint drop-in later. |
| `molchat/tools.py` | Function-calling tool registry: JSON schemas + `Toolbox.dispatch`; centralized name/SMILES resolution. |
| `molchat/llm.py` | LLM backends returning the next `Turn` (tool calls or final answer): rule-based default, Anthropic/OpenAI optional. |
| `molchat/agent.py` | Provider-agnostic tool-calling loop; returns answer + full trace. |
| `molchat/cli.py` | CLI entry point. |
| `molchat/dataset.py` | P2 dataset synthesis: tool-grounded molecular QA pairs. |
| `molchat/eval/retrieval.py` | P3 retrieval metrics + embedder A/B (fingerprint vs random baseline). |
| `molchat/eval/judge.py` | P3 faithfulness judge: rule-based (offline) + LLM-as-judge (optional). |
| `molchat/eval/ab.py` | P3 agent A/B: RAG on vs off, scored by the judge. |
| `serving/app.py` | P4 FastAPI service (`/ask` `/predict` `/search` `/descriptors` `/health`). |
| `serving/Dockerfile` | P4 container image for the service. |
| `edge/export_onnx.py` | P4 edge path: export a compact property model to ONNX + onnxruntime benchmark. |

## The agent loop

1. Seed messages with the user question.
2. `llm.step(messages, tool_specs)` → a `Turn`.
3. If the turn is final → return the answer + trace.
4. Otherwise execute each requested tool via `Toolbox.dispatch`, append the
   tool results to `messages`, and repeat (up to `max_steps`).

The message protocol is uniform (`user` / `assistant` with `tool_calls` /
`tool` with a result dict), so any backend can drive the same loop. The
rule-based backend implements a deterministic policy over this protocol, which
is what keeps the whole system runnable in CI without an API key.

## Pluggability (why the interfaces matter for the roadmap)

- **Embedding backend** — swapping `fingerprint` → `moleco` upgrades retrieval
  quality without touching the store, RAG, tools, or agent.
- **Predictor backend** — swapping the heuristic → the trained Moleco classifier
  (P2) changes only `predictor.py`.
- **LLM backend** — the same tools/loop run under a hosted model or a locally
  served, quantized model (P2/P4).

## Roadmap hooks

- **P2** LoRA fine-tune a small LLM into a molecular-QA assistant; quantize
  (GGUF/AWQ); serve OpenAI-compatible (llama.cpp / Ollama / small vLLM on Colab).
  The agent points its LLM backend at that endpoint.
- **P3** Evaluation harness: base vs RAG-augmented answers, LLM-as-judge scoring,
  A/B deltas over a question set.
- **P4** FastAPI wrapper (`/ask`, `/predict`, `/search`) + Dockerfile + CI, and
  an edge path: ONNX-export the predictor for `onnxruntime`, run the quantized
  LLM on-device via llama.cpp, with a latency/memory budget report.
