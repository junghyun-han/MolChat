# Edge / on-device inference (P4)

Demonstrates the **edge deployment path**: export a compact molecular-property
model to ONNX and run it under `onnxruntime` on CPU, with size and latency
numbers. The trained Moleco classifier exports through the same path once its
checkpoint is available; here we distill the descriptor-based BBB screen into a
tiny logistic-regression surrogate so the path is runnable today.

**Scope:** on-device *inference* (onnxruntime, CPU). Not MCU firmware / RTOS.

## Run

```bash
pip install scikit-learn skl2onnx onnx onnxruntime
python edge/export_onnx.py --out edge/out/bbb_screen.onnx
```

## Measured (this machine, 30-molecule demo set)

| item | value |
|---|---|
| input features | 9 exact RDKit descriptors |
| ONNX model size | ~0.8 KB |
| onnxruntime latency | ~0.004 ms / molecule (CPU) |
| surrogate agreement with the screen | 1.000 |

The point is the pipeline (train → ONNX export → onnxruntime, tiny + fast), not
the accuracy of a 30-molecule demo model.

## Also part of the edge story

The P2 quantized LLM (GGUF Q4) runs on-device via `llama.cpp` (C++ runtime) —
see `p2/README.md`. Together: a quantized language model and an ONNX property
model, both runnable on constrained hardware.
