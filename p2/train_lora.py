"""P2 · LoRA fine-tune Qwen2.5-0.5B-Instruct into a molecular-QA assistant.

Runs on a single Colab T4 (or any small GPU). Trains on the tool-grounded QA
dataset from P1 (data/qa/train.jsonl), evaluates on held-out molecules
(data/qa/eval.jsonl), and saves a LoRA adapter.

Pinned versions this script was written against (see p2/README.md):
    transformers>=4.44  peft>=0.13  trl>=0.11  datasets>=2.20  accelerate>=0.34

    python p2/train_lora.py --data data/qa --out p2/out/molchat-qwen-lora
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/qa")
    parser.add_argument("--out", default="p2/out/molchat-qwen-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(
        "json",
        data_files={
            "train": os.path.join(args.data, "train.jsonl"),
            "eval": os.path.join(args.data, "eval.jsonl"),
        },
    )

    def to_text(example):
        # Render chat messages with the model's own chat template.
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    ds = ds.map(to_text, remove_columns=["messages"])

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    sft_config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=args.base,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("eval metrics:", metrics)

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
