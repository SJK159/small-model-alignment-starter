"""Task 6: Generate evaluation outputs from trained models and build pairwise comparisons.

Run on Kaggle after training:
    python scripts/generate_outputs.py \
        --sft_checkpoint outputs/sft/best \
        --dpo_checkpoint outputs/dpo_ultrafeedback/best \
        --eval_set alpaca_eval \
        --output_dir /kaggle/working/outputs/eval_ultrafeedback

    python scripts/generate_outputs.py \
        --sft_checkpoint outputs/sft/best \
        --dpo_checkpoint outputs/dpo_hh_rlhf/best \
        --eval_set hh_rlhf \
        --output_dir /kaggle/working/outputs/eval_hh_rlhf
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.generate_baseline import (
    build_prompt,
    load_alpaca_eval_prompts,
    load_hh_rlhf_prompts,
)


MAX_NEW_TOKENS = 512
BATCH_SIZE = 8


@torch.no_grad()
def generate_from_model(model, tokenizer, prompts, generator_name, max_new_tokens, batch_size):
    model.eval()
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating ({generator_name})"):
        batch_prompts = prompts[i : i + batch_size]
        texts = [build_prompt(p["instruction"]) for p in batch_prompts]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512,
        ).to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
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
                "dataset": prompt_data.get("dataset", "unknown"),
            })
    return results


def build_pairwise_comparisons(sft_outputs, dpo_outputs, seed=42):
    """Build pairwise JSONL with randomized A/B order."""
    rng = random.Random(seed)
    pairs = []

    sft_by_id = {r["id"]: r for r in sft_outputs}
    dpo_by_id = {r["id"]: r for r in dpo_outputs}

    for pid in sft_by_id:
        if pid not in dpo_by_id:
            continue
        sft_r = sft_by_id[pid]
        dpo_r = dpo_by_id[pid]

        if rng.random() < 0.5:
            response_a, response_b = sft_r["output"], dpo_r["output"]
            model_a, model_b = "sft", "dpo"
        else:
            response_a, response_b = dpo_r["output"], sft_r["output"]
            model_a, model_b = "dpo", "sft"

        pairs.append({
            "id": pid,
            "instruction": sft_r["instruction"],
            "response_a": response_a,
            "response_b": response_b,
            "_model_a": model_a,
            "_model_b": model_b,
        })

    return pairs


def load_model_and_tokenizer(checkpoint_path, device="auto"):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, dtype=torch.float16,
        device_map=device, trust_remote_code=True,
    )
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--dpo_checkpoint", required=True)
    parser.add_argument("--eval_set", choices=["alpaca_eval", "hh_rlhf"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.eval_set == "alpaca_eval":
        prompts = load_alpaca_eval_prompts()
    else:
        prompts = load_hh_rlhf_prompts()
    print(f"Loaded {len(prompts)} prompts ({args.eval_set})")

    # Generate SFT outputs
    print(f"\nLoading SFT model: {args.sft_checkpoint}")
    model, tokenizer = load_model_and_tokenizer(args.sft_checkpoint)
    sft_outputs = generate_from_model(
        model, tokenizer, prompts, "sft", args.max_new_tokens, args.batch_size,
    )
    sft_path = os.path.join(args.output_dir, "sft_outputs.json")
    with open(sft_path, "w") as f:
        json.dump(sft_outputs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(sft_outputs)} SFT outputs to {sft_path}")
    del model
    torch.cuda.empty_cache()

    # Generate DPO outputs
    print(f"\nLoading DPO model: {args.dpo_checkpoint}")
    model, tokenizer = load_model_and_tokenizer(args.dpo_checkpoint)
    dpo_outputs = generate_from_model(
        model, tokenizer, prompts, "dpo", args.max_new_tokens, args.batch_size,
    )
    dpo_path = os.path.join(args.output_dir, "dpo_outputs.json")
    with open(dpo_path, "w") as f:
        json.dump(dpo_outputs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(dpo_outputs)} DPO outputs to {dpo_path}")
    del model
    torch.cuda.empty_cache()

    # Build pairwise comparisons
    pairs = build_pairwise_comparisons(sft_outputs, dpo_outputs, seed=args.seed)
    pairs_path = os.path.join(args.output_dir, "dpo_vs_sft_pairs.jsonl")
    with open(pairs_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} pairwise comparisons to {pairs_path}")


if __name__ == "__main__":
    main()
