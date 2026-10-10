"""Task 5: TRL DPOTrainer + LoRA on Gordon Ramsay persona preferences.

Run on Kaggle T4x2:
    python scripts/train_lora_dpo.py \
        --output_dir /kaggle/working/outputs/lora_dpo
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
DATA_PATH = "data/persona_lora/ramsay_style_preferences.json"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]
LR = 5e-5
BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 3
BETA = 0.1
MAX_LENGTH = 512
SEED = 42


def load_persona_data(path: str):
    with open(path) as f:
        raw = json.load(f)

    records = []
    for row in raw:
        records.append({
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        })
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--data_path", default=DATA_PATH)
    parser.add_argument("--output_dir", default="outputs/lora_dpo")
    parser.add_argument("--lora_rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float16, trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_persona_data(args.data_path)
    print(f"Loaded {len(dataset)} preference pairs from {args.data_path}")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        beta=args.beta,
        max_length=args.max_length,
        logging_steps=5,
        save_strategy="epoch",
        seed=args.seed,
        bf16=False,
        fp16=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting LoRA DPO training...")
    trainer.train()

    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print(f"LoRA DPO model saved to {args.output_dir}/final")

    config = vars(args)
    config["lora_dropout"] = LORA_DROPOUT
    config["target_modules"] = TARGET_MODULES
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
