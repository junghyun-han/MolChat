# P2 — LoRA fine-tune → quantize → serve → wire into the agent

Turns the P1 tool-grounded dataset into a small, fine-tuned, quantized LLM that
becomes MolChat's reasoning engine. Base model: **Qwen2.5-0.5B-Instruct**
(runs LoRA + quantization + serving comfortably on a free Colab T4 or a Mac).

## 2-a · Data (done on laptop, no GPU)

```bash
python scripts/build_dataset.py --out data/qa
# -> data/qa/train.jsonl (143), data/qa/eval.jsonl (30), split by molecule
```

Answers are grounded in P1's own tools (exact RDKit descriptors, BBB predictor,
embedding retrieval), so the model is distilled from faithful outputs.

## 2-b · LoRA fine-tune (Colab T4)

```bash
pip install "transformers>=4.44" "peft>=0.13" "trl>=0.11" \
            "datasets>=2.20" "accelerate>=0.34"
python p2/train_lora.py --data data/qa --out p2/out/molchat-qwen-lora
```

LoRA config: r=16, alpha=32, dropout=0.05 on the attention projections — only a
small fraction of parameters are trained (PEFT). Eval runs on the held-out
molecules to measure generalization. See `notebooks/p2_lora_qwen.ipynb` for the
one-click Colab version.

## 2-c · Quantize to GGUF (Mac or Colab)

Merge the adapter, convert to GGUF, and quantize to 4-bit with llama.cpp:

```bash
# merge LoRA into the base weights (small script or peft merge_and_unload)
python p2/merge_adapter.py --adapter p2/out/molchat-qwen-lora --out p2/out/merged

git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
python convert_hf_to_gguf.py ../p2/out/merged --outfile molchat-f16.gguf
./llama-quantize molchat-f16.gguf molchat-q4_k_m.gguf Q4_K_M
```

Report the size / latency / quality trade-off (f16 vs Q4) in a small table —
this is the "inference optimization / quantization" evidence.

## 2-d · Serve + wire into the MolChat agent

Serve the quantized model with an OpenAI-compatible endpoint (Ollama shown):

```bash
ollama create molchat -f p2/Modelfile        # points at molchat-q4_k_m.gguf
ollama serve                                  # OpenAI-compatible at :11434/v1
```

Point the agent at it — no code change, just env:

```bash
pip install openai
export MOLCHAT_LLM=local
export MOLCHAT_LLM_BASE_URL=http://localhost:11434/v1
export MOLCHAT_LLM_MODEL=molchat
python -m molchat.cli --trace "Is diazepam BBB permeable?"
```

Now the agent's reasoning is done by *your* fine-tuned, quantized, locally-served
model, calling the same P1 tools. That is the full "built + ran an LLM system"
path: fine-tune → quantize → serve → orchestrate.

## Edge / embedded (P4 hook)

The same GGUF runs on-device via `llama.cpp` (C++), and the property predictor
can be ONNX-exported for `onnxruntime` — a resource-constrained inference path
with a latency/memory budget. Scoped to edge inference, not MCU firmware.
