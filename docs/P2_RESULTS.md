# P2 results — fine-tune → quantize → serve (measured)

All steps executed locally on an Apple Silicon Mac (MPS / Metal), no cloud GPU.

## Setup
- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Data: 143 train + 30 eval tool-grounded QA pairs, split **by molecule** (eval
  molecules unseen in training) — see `molchat/dataset.py`.

## 2-b · LoRA fine-tune (`p2/train_lora_local.py`, Apple MPS)
| metric | value |
|---|---|
| LoRA config | r=16, alpha=32, dropout=0.05, targets q/k/v/o |
| trainable params | 2,162,688 / 496,195,456 (**0.44%**) |
| epochs / device / time | 3 / MPS / ~139 s |
| train loss | 1.375 → 0.32 |
| **eval loss** | **3.386 → 0.457** (held-out molecules) |

## 2-c · Quantize to GGUF (llama.cpp, Q4_K_M)
| model | size |
|---|---|
| f16 GGUF | 994.2 MB |
| **Q4_K_M** | **397.8 MB** |
| reduction | **2.50× smaller (60.0%)** |

## 2-d · Serve (llama.cpp, Metal)
- The quantized fine-tuned model answers in the trained style, e.g.
  `Question: What is the molecular weight of ethanol? Answer:` →
  **"The molecular weight of ethanol is 46.07 g/mol."**
- Throughput: prompt 177 t/s, **generation 137 t/s** (Apple Metal).
- Served with an OpenAI-compatible endpoint (Ollama `p2/Modelfile`); the MolChat
  agent uses it via `MOLCHAT_LLM=local`.

## Honest scope
The dataset is small (30 molecules), so this demonstrates that the
fine-tune → quantize → serve **pipeline runs end-to-end with real engineering
trade-offs** (parameter efficiency, 2.5× compression, on-device throughput). It
is not a claim of state-of-the-art property-prediction accuracy.
