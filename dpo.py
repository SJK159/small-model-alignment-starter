"""Core DPO batch-building and loss functions.

Used by submission_adapter.py and scripts/train_dpo.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sft import build_sft_batch, compute_response_logprobs


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""
    chosen_batch = build_sft_batch(tokenizer, prompts, chosen_responses, max_length)
    rejected_batch = build_sft_batch(tokenizer, prompts, rejected_responses, max_length)
    return {"chosen": chosen_batch, "rejected": rejected_batch}


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""
    policy_diff = policy_chosen_logps - policy_rejected_logps
    reference_diff = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_diff - reference_diff)
    return -F.logsigmoid(logits).mean()


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""
    policy_chosen_logps = compute_response_logprobs(
        policy_model, tokenizer, prompts, chosen_responses, max_length,
    )
    policy_rejected_logps = compute_response_logprobs(
        policy_model, tokenizer, prompts, rejected_responses, max_length,
    )

    # Re-enable gradients for policy log probs (compute_response_logprobs uses no_grad)
    policy_chosen_logps = _logprobs_with_grad(
        policy_model, tokenizer, prompts, chosen_responses, max_length,
    )
    policy_rejected_logps = _logprobs_with_grad(
        policy_model, tokenizer, prompts, rejected_responses, max_length,
    )

    with torch.no_grad():
        reference_chosen_logps = compute_response_logprobs(
            reference_model, tokenizer, prompts, chosen_responses, max_length,
        )
        reference_rejected_logps = compute_response_logprobs(
            reference_model, tokenizer, prompts, rejected_responses, max_length,
        )

    return compute_dpo_loss_from_logps(
        policy_chosen_logps, policy_rejected_logps,
        reference_chosen_logps, reference_rejected_logps, beta,
    )


def _logprobs_with_grad(model, tokenizer, prompts, responses, max_length):
    """Like compute_response_logprobs but keeps gradients for the policy model."""
    from sft import build_sft_batch

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

    return token_log_probs.sum(dim=-1)
