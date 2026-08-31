# MolChat — Evaluation report (P3)

## 1. Retrieval A/B — do embeddings capture chemistry?

Metric: `category_consistency@3` (higher = neighbors share chemical class).

| embedder | self_recall@1 | category_consistency@3 |
|---|---|---|
| fingerprint(morgan,r=2,bits=2048) | 1.0 | 0.289 |
| random(dim=256) | 1.0 | 0.033 |

**Winner: fingerprint(morgan,r=2,bits=2048) (Δ 0.256).** The fingerprint embedding beats the random baseline, i.e. retrieval is chemically meaningful.

## 2. Agent A/B — RAG on vs off

Judge: `rule-based` (faithfulness = tool-grounded).

| condition | groundedness | coverage | tools/answer |
|---|---|---|---|
| rag | 1.0 | 3 | 3 |
| no_rag | 1.0 | 2 | 2 |

**Δ coverage = 1, Δ groundedness = 0.0.** RAG raises coverage (neighbor context) while both conditions stay faithful to their tool outputs.

