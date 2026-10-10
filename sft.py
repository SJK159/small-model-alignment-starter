"""Core SFT batch-building and loss functions.

Used by submission_adapter.py and scripts/train_sft.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Tokenize prompt+response+EOS, pad/truncate to max_length, mask prompt and padding in labels."""
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        response_ids = response_ids + [eos_id]

        input_ids = prompt_ids + response_ids
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            prompt_len = len(prompt_ids)
            if prompt_len >= max_length:
                prompt_len = max_length
                resp_len = 0
            else:
                resp_len = max_length - prompt_len
        else:
            prompt_len = len(prompt_ids)
            resp_len = len(response_ids)

        seq_len = len(input_ids)
        pad_len = max_length - seq_len

        labels = ([-100] * prompt_len
                  + input_ids[prompt_len:prompt_len + resp_len]
                  + [-100] * pad_len)
        attention_mask = [1] * seq_len + [0] * pad_len
        input_ids = input_ids + [pad_id] * pad_len

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Causal LM: logits[t] predicts token at t+1
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)

    # Mask: only response tokens (where shifted labels != -100)
    response_mask = shift_labels.ne(-100).float()
    token_log_probs = token_log_probs * response_mask

    # Sum per example
    return token_log_probs.sum(dim=-1)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return scalar SFT loss: mean of per-example summed negative log probabilities."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)

    response_mask = shift_labels.ne(-100).float()
    token_log_probs = token_log_probs * response_mask

    per_example_logprobs = token_log_probs.sum(dim=-1)
    return -per_example_logprobs.mean()
