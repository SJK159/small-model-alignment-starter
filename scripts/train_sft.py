"""SFT training loop for Qwen2.5-0.5B on UltraChat data.

Run on Kaggle T4x2:
    python scripts/train_sft.py --output_dir /kaggle/working/outputs/sft
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sft as sft_core


# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MAX_LENGTH = 512
TRAIN_EXAMPLES = 30_000
VAL_EXAMPLES = 1_000
BATCH_SIZE = 2
GRAD_ACCUM = 16
LR = 2e-5
WARMUP_FRAC = 0.03
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EPOCHS = 1
SEED = 42


class SFTDataset(Dataset):
    def __init__(self, prompts: list[str], responses: list[str]):
        self.prompts = prompts
        self.responses = responses

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.responses[idx]


def load_ultrachat_data(n_train: int, n_val: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=seed)

    train_prompts, train_responses = [], []
    val_prompts, val_responses = [], []

    target = n_train + n_val
    count = 0
    for row in ds:
        messages = row["messages"]
        # Extract single-turn: first user message → first assistant response
        user_msg = None
        assistant_msg = None
        for m in messages:
            if m["role"] == "user" and user_msg is None:
                user_msg = m["content"].strip()
            elif m["role"] == "assistant" and user_msg is not None and assistant_msg is None:
                assistant_msg = m["content"].strip()
                break
        if not user_msg or not assistant_msg:
            continue

        if count < n_train:
            train_prompts.append(user_msg)
            train_responses.append(assistant_msg)
        elif count < target:
            val_prompts.append(user_msg)
            val_responses.append(assistant_msg)
        else:
            break
        count += 1

    print(f"Loaded {len(train_prompts)} train, {len(val_prompts)} val examples")
    return train_prompts, train_responses, val_prompts, val_responses


def collate_fn(batch, tokenizer, max_length):
    prompts, responses = zip(*batch)
    return sft_core.build_sft_batch(tokenizer, list(prompts), list(responses), max_length)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)
        response_mask = shift_labels.ne(-100).float()
        token_log_probs = token_log_probs * response_mask
        loss = -token_log_probs.sum(dim=-1).mean()

        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--output_dir", default="outputs/sft")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--train_examples", type=int, default=TRAIN_EXAMPLES)
    parser.add_argument("--val_examples", type=int, default=VAL_EXAMPLES)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--warmup_frac", type=float, default=WARMUP_FRAC)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad_clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--log_steps", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float32, trust_remote_code=True,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # ── Load data ─────────────────────────────────────────────────────────────
    train_prompts, train_responses, val_prompts, val_responses = load_ultrachat_data(
        args.train_examples, args.val_examples, args.seed,
    )

    train_dataset = SFTDataset(train_prompts, train_responses)
    val_dataset = SFTDataset(val_prompts, val_responses)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
        drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
        num_workers=0,
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"Training: {total_steps} optimizer steps, {warmup_steps} warmup, "
          f"effective batch {args.batch_size * args.grad_accum}")

    metrics_log = []
    best_val_loss = float("inf")
    global_step = 0
    accum_loss = 0.0

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for batch_idx, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                shift_logits = logits[:, :-1, :].float()
                shift_labels = labels[:, 1:]
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    2, shift_labels.clamp(min=0).unsqueeze(2),
                ).squeeze(2)
                response_mask = shift_labels.ne(-100).float()
                token_log_probs = token_log_probs * response_mask
                loss = -token_log_probs.sum(dim=-1).mean()
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % args.log_steps == 0:
                    avg_loss = accum_loss * args.grad_accum / args.log_steps
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                    metrics_log.append({
                        "step": global_step, "train_loss": avg_loss,
                        "lr": scheduler.get_last_lr()[0],
                    })
                    accum_loss = 0.0

                if global_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"\n  Step {global_step}: val_loss={val_loss:.4f}")
                    metrics_log.append({
                        "step": global_step, "val_loss": val_loss,
                    })
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.save_pretrained(os.path.join(args.output_dir, "best"))
                        tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))
                        print(f"  Saved best checkpoint (val_loss={val_loss:.4f})")

    # Final checkpoint
    model.save_pretrained(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    # Final validation
    val_loss = evaluate(model, val_loader, device)
    print(f"\nFinal val_loss={val_loss:.4f}, best val_loss={best_val_loss:.4f}")
    if val_loss < best_val_loss:
        model.save_pretrained(os.path.join(args.output_dir, "best"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))
        print("Updated best checkpoint with final model")

    # Save metrics
    with open(os.path.join(args.output_dir, "metrics.jsonl"), "w") as f:
        for entry in metrics_log:
            f.write(json.dumps(entry) + "\n")
    print(f"Metrics saved to {args.output_dir}/metrics.jsonl")

    # Save config
    config = vars(args)
    config["total_steps"] = total_steps
    config["warmup_steps"] = warmup_steps
    config["best_val_loss"] = best_val_loss
    config["final_val_loss"] = val_loss
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
