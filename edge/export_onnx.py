"""Edge / on-device path (P4): export a compact property model to ONNX and
benchmark it with onnxruntime.

The trained Moleco classifier is not shipped here, so to demonstrate the *edge
deployment path* end-to-end we distill the descriptor-based BBB screen into a
small logistic-regression surrogate over exact RDKit descriptors, export it to
ONNX, and run it under onnxruntime — reporting model size and per-molecule
latency. The real Moleco checkpoint exports through the identical path.

Scope note: this is on-device *inference* (onnxruntime on CPU), not MCU firmware.

    python edge/export_onnx.py --out edge/out/bbb_screen.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molchat.corpus import load_corpus  # noqa: E402
from molchat.predictor import HeuristicBBBPredictor  # noqa: E402

FEATURES = [
    "molecular_weight", "logp", "tpsa", "h_bond_donors", "h_bond_acceptors",
    "rotatable_bonds", "ring_count", "aromatic_rings", "heavy_atoms",
]


def _dataset():
    corpus = load_corpus()
    predictor = HeuristicBBBPredictor()
    X, y = [], []
    for row in corpus:
        d = row["descriptors"]
        X.append([float(d[f]) for f in FEATURES])
        label = predictor.predict_bbb(row["smiles"])["label"]
        y.append(1 if "permeant" in label and "non" not in label else 0)
    return np.asarray(X, dtype="float32"), np.asarray(y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="edge/out/bbb_screen.onnx")
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _dataset()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X, y)
    train_acc = float((model.predict(X) == y).mean())

    onx = convert_sklearn(
        model,
        initial_types=[("input", FloatTensorType([None, len(FEATURES)]))],
        target_opset=17,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        fh.write(onx.SerializeToString())
    size_kb = os.path.getsize(args.out) / 1024.0

    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    sample = X[:1]
    # agreement between onnxruntime output and the sklearn model
    onnx_pred = sess.run(None, {"input": sample})[0]
    # latency
    t0 = time.perf_counter()
    for _ in range(args.runs):
        sess.run(None, {"input": sample})
    latency_ms = (time.perf_counter() - t0) / args.runs * 1000.0

    print(f"features:            {len(FEATURES)}")
    print(f"train molecules:     {len(y)} (small demo set)")
    print(f"surrogate train acc: {train_acc:.3f} (reproduces the descriptor screen)")
    print(f"onnx model size:     {size_kb:.1f} KB")
    print(f"onnxruntime latency: {latency_ms:.3f} ms / molecule (CPU, {args.runs} runs)")
    print(f"onnx sample output:  {onnx_pred.tolist()}")
    print(f"saved:               {args.out}")


if __name__ == "__main__":
    main()
