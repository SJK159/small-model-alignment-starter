"""Generate baseline responses from Qwen/Qwen2.5-0.5B on fixed eval subsets.

Run on Kaggle T4x2 after installing dependencies:
    python scripts/generate_baseline.py \
        --eval_set alpaca_eval \
        --output_path outputs/baseline/alpaca_eval_base.json

Supports two eval sets:
    alpaca_eval  — 300 prompts sampled from tatsu-lab/alpaca_eval with seed 42
    hh_rlhf      — the fixed 300 prompts from eval_sets/hh_rlhf_eval_300_seed42.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B"
ALPACA_EVAL_N = 300
ALPACA_EVAL_SEED = 42
MAX_NEW_TOKENS = 512
BATCH_SIZE = 8


def load_alpaca_eval_prompts(n: int = ALPACA_EVAL_N, seed: int = ALPACA_EVAL_SEED) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval", trust_remote_code=True)
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected = sorted(indices[:n])
    prompts = []
    for idx in selected:
        row = ds[idx]
        prompts.append({
            "id": f"alpaca-eval-{seed}-{len(prompts):04d}",
            "instruction": row["instruction"],
            "dataset": "alpaca_eval",
        })
    return prompts


def load_hh_rlhf_prompts(path: str = "eval_sets/hh_rlhf_eval_300_seed42.json") -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data


def build_prompt(instruction: str) -> str:
    return instruction.strip()


def generate_responses(
    model,
    tokenizer,
    prompts: list[dict],
    generator_name: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    import torch
    from tqdm import tqdm

    model.eval()
    results = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i : i + batch_size]
            texts = [build_prompt(p["instruction"]) for p in batch_prompts]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            for j, prompt_data in enumerate(batch_prompts):
                input_len = inputs["attention_mask"][j].sum().item()
                generated_ids = outputs[j][input_len:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                results.append({
                    "id": prompt_data["id"],
                    "instruction": prompt_data["instruction"],
                    "output": response,
                    "generator": generator_name,
                    "dataset": prompt_data["dataset"],
                })
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate baseline responses")
    parser.add_argument("--eval_set", choices=["alpaca_eval", "hh_rlhf"], default="alpaca_eval")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--generator_name", default=None)
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f"outputs/baseline/{args.eval_set}_base.json"
    if args.generator_name is None:
        args.generator_name = "qwen2.5-0.5b-base"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading eval set: {args.eval_set}")
    if args.eval_set == "alpaca_eval":
        prompts = load_alpaca_eval_prompts()
    else:
        prompts = load_hh_rlhf_prompts()
    print(f"  {len(prompts)} prompts loaded")

    print("Generating responses...")
    results = generate_responses(
        model, tokenizer, prompts,
        generator_name=args.generator_name,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} responses to {args.output_path}")


if __name__ == "__main__":
    main()
