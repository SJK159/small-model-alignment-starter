"""From-scratch DPO training loop.

Run on Kaggle T4x2:
    python scripts/train_dpo.py \
        --sft_checkpoint outputs/sft/best \
        --dataset hh_rlhf \
        --output_dir /kaggle/working/outputs/dpo_hh_rlhf

    python scripts/train_dpo.py \
        --sft_checkpoint outputs/sft/best \
        --dataset ultrafeedback \
        --output_dir /kaggle/working/outputs/dpo_ultrafeedback
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sft as sft_core
from dpo import compute_dpo_loss_from_logps


MAX_LENGTH = 512
MAX_PROMPT_LENGTH = 256
TRAIN_PAIRS = 15_000
VAL_PAIRS = 1_000
BATCH_SIZE = 1
GRAD_ACCUM = 32
LR = 1e-6
BETA = 0.1
GRAD_CLIP = 1.0
EPOCHS = 1
SEED = 42


class DPODataset(Dataset):
    def __init__(self, prompts, chosen, rejected):
        self.prompts = prompts
        self.chosen = chosen
        self.rejected = rejected

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.chosen[idx], self.rejected[idx]


def load_hh_rlhf_data(n_train, n_val, seed):
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    ds = ds.shuffle(seed=seed)

    prompts, chosen, rejected = [], [], []
    target = n_train + n_val
    count = 0

    for row in ds:
        chosen_text = row["chosen"]
        rejected_text = row["rejected"]
        # Extract single-turn: split on "\n\nAssistant:" and "\n\nHuman:"
        parts_c = chosen_text.split("\n\nAssistant:")
        parts_r = rejected_text.split("\n\nAssistant:")
        if len(parts_c) < 2 or len(parts_r) < 2:
            continue

        prompt = parts_c[0].replace("\n\nHuman: ", "").strip()
        c_resp = parts_c[1].strip()
        r_resp = parts_r[1].strip()

        # Filter multi-turn
        if "\n\nHuman:" in c_resp or "\n\nHuman:" in r_resp:
            continue
        if not prompt or not c_resp or not r_resp:
            continue

        prompts.append(prompt)
        chosen.append(c_resp)
        rejected.append(r_resp)
        count += 1
        if count >= target:
            break

    train_end = min(n_train, len(prompts))
    val_end = min(n_train + n_val, len(prompts))
    print(f"HH-RLHF: {train_end} train, {val_end - train_end} val pairs")
    return (
        prompts[:train_end], chosen[:train_end], rejected[:train_end],
        prompts[train_end:val_end], chosen[train_end:val_end], rejected[train_end:val_end],
    )


def load_ultrafeedback_data(n_train, n_val, seed):
    from datasets import load_dataset

    ds = load_dataset("openbmb/UltraFeedback", split="train")
    ds = ds.shuffle(seed=seed)

    prompts, chosen, rejected = [], [], []
    target = n_train + n_val
    count = 0

    for row in ds:
        instruction = row.get("instruction", "")
        completions = row.get("completions", [])
        if not instruction or len(completions) < 2:
            continue

        # Sort by overall_score descending, pick best and worst
        scored = []
        for c in completions:
            annotations = c.get("annotations", {})
            score = 0.0
            n_scores = 0
            for ann in annotations.values():
                try:
                    s = float(ann.get("Rating", 0))
                    score += s
                    n_scores += 1
                except (ValueError, TypeError):
                    pass
            if n_scores > 0:
                scored.append((score / n_scores, c.get("response", "")))

        if len(scored) < 2:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        best_resp = scored[0][1].strip()
        worst_resp = scored[-1][1].strip()

        if not best_resp or not worst_resp or best_resp == worst_resp:
            continue

        prompts.append(instruction.strip())
        chosen.append(best_resp)
        rejected.append(worst_resp)
        count += 1
        if count >= target:
            break

    train_end = min(n_train, len(prompts))
    val_end = min(n_train + n_val, len(prompts))
    print(f"UltraFeedback: {train_end} train, {val_end - train_end} val pairs")
    return (
        prompts[:train_end], chosen[:train_end], rejected[:train_end],
        prompts[train_end:val_end], chosen[train_end:val_end], rejected[train_end:val_end],
    )


def collate_fn(batch, tokenizer, max_length):
    prompts, chosen, rejected = zip(*batch)
    chosen_batch = sft_core.build_sft_batch(tokenizer, list(prompts), list(chosen), max_length)
    rejected_batch = sft_core.build_sft_batch(tokenizer, list(prompts), list(rejected), max_length)
    return chosen_batch, rejected_batch


def _response_logprobs_with_grad(model, batch, device):
    """Compute summed response log probs with gradients."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)
    response_mask = shift_labels.ne(-100).float()
    return (token_log_probs * response_mask).sum(dim=-1)


@torch.no_grad()
def _response_logprobs_no_grad(model, batch, device):
    """Compute summed response log probs without gradients."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)
    response_mask = shift_labels.ne(-100).float()
    return (token_log_probs * response_mask).sum(dim=-1)


@torch.no_grad()
def evaluate_dpo(policy_model, reference_model, val_loader, policy_device, ref_device, beta):
    policy_model.eval()
    total_loss = 0.0
    total_reward_acc = 0.0
    total_reward_margin = 0.0
    n_examples = 0

    for chosen_batch, rejected_batch in val_loader:
        pc = _response_logprobs_no_grad(policy_model, chosen_batch, policy_device)
        pr = _response_logprobs_no_grad(policy_model, rejected_batch, policy_device)
        rc = _response_logprobs_no_grad(reference_model, chosen_batch, ref_device)
        rr = _response_logprobs_no_grad(reference_model, rejected_batch, ref_device)

        pc, pr = pc.to(policy_device), pr.to(policy_device)
        rc, rr = rc.to(policy_device), rr.to(policy_device)

        loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)

        r_w = beta * (pc - rc)
        r_l = beta * (pr - rr)
        reward_acc = (r_w > r_l).float().mean()
        reward_margin = (r_w - r_l).mean()

        bs = pc.shape[0]
        total_loss += loss.item() * bs
        total_reward_acc += reward_acc.item() * bs
        total_reward_margin += reward_margin.item() * bs
        n_examples += bs

    policy_model.train()
    n = max(n_examples, 1)
    return total_loss / n, total_reward_acc / n, total_reward_margin / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--dataset", choices=["hh_rlhf", "ultrafeedback"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--train_pairs", type=int, default=TRAIN_PAIRS)
    parser.add_argument("--val_pairs", type=int, default=VAL_PAIRS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--grad_clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--log_steps", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"Loading policy model from: {args.sft_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    policy_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else policy_device)

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_checkpoint, dtype=torch.float32, trust_remote_code=True,
    ).to(policy_device)
    policy_model.train()

    print(f"Loading reference model from: {args.sft_checkpoint}")
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.sft_checkpoint, dtype=torch.float16, trust_remote_code=True,
    ).to(ref_device)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.dataset == "hh_rlhf":
        (train_p, train_c, train_r,
         val_p, val_c, val_r) = load_hh_rlhf_data(args.train_pairs, args.val_pairs, args.seed)
    else:
        (train_p, train_c, train_r,
         val_p, val_c, val_r) = load_ultrafeedback_data(args.train_pairs, args.val_pairs, args.seed)

    train_dataset = DPODataset(train_p, train_c, train_r)
    val_dataset = DPODataset(val_p, val_c, val_r)

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

    # ── Optimizer ─────────────────────────────────────────────────────────────
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = GradScaler("cuda")

    print(f"Training DPO ({args.dataset}): {total_steps} optimizer steps, "
          f"effective batch {args.batch_size * args.grad_accum}, beta={args.beta}")

    # ── Training loop ─────────────────────────────────────────────────────────
    metrics_log = []
    best_reward_acc = 0.0
    global_step = 0
    accum_loss = 0.0

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for batch_idx, (chosen_batch, rejected_batch) in pbar:
            with autocast("cuda", dtype=torch.float16):
                pc = _response_logprobs_with_grad(policy_model, chosen_batch, policy_device)
                pr = _response_logprobs_with_grad(policy_model, rejected_batch, policy_device)

            with torch.no_grad():
                rc = _response_logprobs_no_grad(reference_model, chosen_batch, ref_device)
                rr = _response_logprobs_no_grad(reference_model, rejected_batch, ref_device)

            rc, rr = rc.to(policy_device), rr.to(policy_device)

            with autocast("cuda", dtype=torch.float16):
                loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, args.beta)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_steps == 0:
                    avg_loss = accum_loss * args.grad_accum / args.log_steps
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", step=global_step)
                    metrics_log.append({"step": global_step, "train_loss": avg_loss})
                    accum_loss = 0.0

                if global_step % args.eval_steps == 0:
                    val_loss, val_racc, val_rmargin = evaluate_dpo(
                        policy_model, reference_model, val_loader,
                        policy_device, ref_device, args.beta,
                    )
                    print(f"\n  Step {global_step}: val_loss={val_loss:.4f} "
                          f"reward_acc={val_racc:.4f} reward_margin={val_rmargin:.4f}")
                    metrics_log.append({
                        "step": global_step, "val_loss": val_loss,
                        "reward_accuracy": val_racc, "reward_margin": val_rmargin,
                    })
                    if val_racc > best_reward_acc:
                        best_reward_acc = val_racc
                        policy_model.save_pretrained(os.path.join(args.output_dir, "best"))
                        tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))
                        print(f"  Saved best checkpoint (reward_acc={val_racc:.4f})")

    # Final checkpoint
    policy_model.save_pretrained(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    val_loss, val_racc, val_rmargin = evaluate_dpo(
        policy_model, reference_model, val_loader,
        policy_device, ref_device, args.beta,
    )
    print(f"\nFinal: val_loss={val_loss:.4f} reward_acc={val_racc:.4f} "
          f"reward_margin={val_rmargin:.4f}, best reward_acc={best_reward_acc:.4f}")
    if val_racc > best_reward_acc:
        policy_model.save_pretrained(os.path.join(args.output_dir, "best"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))

    with open(os.path.join(args.output_dir, "metrics.jsonl"), "w") as f:
        for entry in metrics_log:
            f.write(json.dumps(entry) + "\n")

    config = vars(args)
    config["total_steps"] = total_steps
    config["best_reward_accuracy"] = best_reward_acc
    config["final_reward_accuracy"] = val_racc
    config["final_reward_margin"] = val_rmargin
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Done. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
