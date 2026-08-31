"""Edge ONNX export test. Skipped unless the optional ONNX stack is installed."""

import pytest

pytest.importorskip("skl2onnx")
pytest.importorskip("onnxruntime")


def test_onnx_export_and_inference(tmp_path):
    import onnxruntime as ort

    from edge.export_onnx import FEATURES, _dataset, main

    # dataset shape sanity
    X, y = _dataset()
    assert X.shape[1] == len(FEATURES)
    assert set(y.tolist()) <= {0, 1}

    out = tmp_path / "model.onnx"
    import sys

    argv = sys.argv
    sys.argv = ["export_onnx.py", "--out", str(out), "--runs", "5"]
    try:
        main()
    finally:
        sys.argv = argv

    assert out.exists() and out.stat().st_size > 0
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    pred = sess.run(None, {"input": X[:2]})[0]
    assert len(pred) == 2
