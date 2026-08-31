"""P2 · LoRA fine-tune Qwen2.5-0.5B-Instruct on a laptop (Apple MPS or CPU).

Version-stable path using transformers Trainer + peft directly (no TRL), so it
runs on macOS without CUDA. Trains on the tool-grounded QA dataset, evaluates on
held-out molecules, and saves a LoRA adapter.

    python p2/train_lora_local.py --data data/qa --out p2/out/molchat-qwen-lora
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_chat_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/qa")
    parser.add_argument("--out", default="p2/out/molchat-qwen-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=384)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_text(rows):
        return [
            tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
            for r in rows
        ]

    train_texts = to_text(_load_chat_jsonl(os.path.join(args.data, "train.jsonl")))
    eval_texts = to_text(_load_chat_jsonl(os.path.join(args.data, "eval.jsonl")))

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    train_ds = Dataset.from_dict({"text": train_texts}).map(
        tokenize, batched=True, remove_columns=["text"]
    )
    eval_ds = Dataset.from_dict({"text": eval_texts}).map(
        tokenize, batched=True, remove_columns=["text"]
    )

    model = AutoModelForCausalLM.from_pretrained(args.base).to(device)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        use_cpu=(device == "cpu"),
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator,
    )
    pre = trainer.evaluate()
    print(f"eval loss BEFORE training: {pre['eval_loss']:.4f}")
    trainer.train()
    post = trainer.evaluate()
    print(f"eval loss AFTER training:  {post['eval_loss']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(
            {
                "trainable_params": trainable,
                "total_params": total,
                "trainable_pct": round(100 * trainable / total, 3),
                "eval_loss_before": round(float(pre["eval_loss"]), 4),
                "eval_loss_after": round(float(post["eval_loss"]), 4),
                "device": device,
                "epochs": args.epochs,
            },
            fh,
            indent=2,
        )
    print(f"saved adapter + metrics to {args.out}")


if __name__ == "__main__":
    main()
