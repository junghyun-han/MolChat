"""Write the synthesized molecular QA dataset to JSONL (train/eval).

    python scripts/build_dataset.py --out data/qa

Produces data/qa/train.jsonl and data/qa/eval.jsonl in chat-messages format,
ready for SFT/LoRA of an instruct model (see notebooks/p2_lora_qwen.ipynb).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molchat.dataset import build_dataset  # noqa: E402


def _write_jsonl(path: str, rows) -> None:
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/qa")
    parser.add_argument("--holdout", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = build_dataset(holdout=args.holdout)
    _write_jsonl(os.path.join(args.out, "train.jsonl"), data["train"])
    _write_jsonl(os.path.join(args.out, "eval.jsonl"), data["eval"])
    print(
        f"Wrote {len(data['train'])} train + {len(data['eval'])} eval examples "
        f"to {args.out}/ (held out {args.holdout} molecules for eval)."
    )


if __name__ == "__main__":
    main()
