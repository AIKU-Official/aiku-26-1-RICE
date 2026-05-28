"""Configurable forget + RMU loss for forget/retain paired batches.

Split into separate functions for chunked backward on memory-constrained GPUs.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _model_device(model):
    return next(model.parameters()).device


def _batch_to_device(inputs, device):
    return tuple(tensor.to(device) for tensor in inputs)


def _shift_logits_and_labels(logits, labels):
    return logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()


def _token_ce_from_logits(logits, labels):
    shifted_logits, shifted_labels = _shift_logits_and_labels(logits, labels)
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=-100,
    )


def _sequence_nll_from_logits(logits, labels):
    shifted_logits, shifted_labels = _shift_logits_and_labels(logits, labels)
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    return loss_function(shifted_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)


def compute_batch_nll(model, inputs):
    device = _model_device(model)
    input_ids, labels, attention_mask = _batch_to_device(inputs, device)
    outputs = model(input_ids, attention_mask=attention_mask)
    loss = _sequence_nll_from_logits(outputs.logits, labels)
    return loss, outputs


def compute_forget_rmu_loss(model, oracle_model, forget_inputs,
                            beta=1.0, gamma=1.0,
                            rmu_layers=None, rmu_scale=300.0, rmu_lambda=1.0,
                            random_vectors=None, forget_loss_type="npo"):
    """Compute configurable forget loss + RMU loss in a single forward pass."""
    input_ids, labels, attention_mask = forget_inputs
    device = _model_device(model)
    forget_loss_type = str(forget_loss_type).lower()

    outputs = model(
        input_ids.to(device),
        attention_mask=attention_mask.to(device),
        output_hidden_states=True,
    )

    logits = outputs.logits
    labels = labels.to(device)
    if forget_loss_type == "npo":
        if oracle_model is None:
            raise ValueError("forget_loss_type='npo' requires an oracle/reference model.")
        forget_nll = _sequence_nll_from_logits(logits, labels)
        with torch.no_grad():
            ref_nll, _ = compute_batch_nll(oracle_model, forget_inputs)
            ref_nll = ref_nll.to(device)

        lose_log_ratio = -(forget_nll - ref_nll)
        forget_loss = -2 / beta * F.logsigmoid(beta * (0.0 - lose_log_ratio)).mean()
    elif forget_loss_type in {"ga", "gradient_ascent", "gradient-ascent"}:
        forget_loss = -_token_ce_from_logits(logits, labels)
    else:
        raise ValueError(f"Unknown forget_loss_type: {forget_loss_type}")

    # RMU loss from hidden states
    rmu_loss = torch.tensor(0.0, device=device)
    if rmu_layers and random_vectors:
        mask = attention_mask.to(device).unsqueeze(-1).float()
        active_hidden_count = mask.sum().clamp_min(1.0) * outputs.hidden_states[-1].shape[-1]
        for layer_idx in rmu_layers:
            h = outputs.hidden_states[layer_idx + 1].float()
            r = random_vectors[layer_idx].to(device).float()
            target = r * rmu_scale
            squared_error = F.mse_loss(
                h * mask,
                target.unsqueeze(0).unsqueeze(0).expand_as(h) * mask,
                reduction="sum",
            )
            rmu_loss = rmu_loss + squared_error / active_hidden_count
        rmu_loss = rmu_loss / len(rmu_layers)

    return gamma * forget_loss + rmu_lambda * rmu_loss


def compute_retain_loss(model, oracle_model, retain_inputs, retain_loss_type="ce"):
    """Compute retain loss (separate forward pass)."""
    device = _model_device(model)
    retain_loss_type = str(retain_loss_type).lower()

    if retain_loss_type == "kl":
        if oracle_model is None:
            raise ValueError("retain_loss_type='kl' requires an oracle/reference model.")
        retain_outputs = model(
            retain_inputs[0].to(device),
            attention_mask=retain_inputs[2].to(device),
        )
        current_probs = F.log_softmax(retain_outputs.logits, dim=-1)
        current_probs = current_probs.view(-1, retain_outputs.logits.shape[-1])

        with torch.no_grad():
            oracle_device = _model_device(oracle_model)
            oracle_outputs = oracle_model(
                retain_inputs[0].to(oracle_device),
                attention_mask=retain_inputs[2].to(oracle_device),
            )
        oracle_probs = F.log_softmax(oracle_outputs.logits, dim=-1)
        oracle_probs = oracle_probs.view(-1, oracle_outputs.logits.shape[-1])

        retain_loss = F.kl_div(
            current_probs, oracle_probs.to(device),
            reduction='batchmean', log_target=True,
        )
    else:
        retain_input_ids, retain_labels, retain_attention_mask = _batch_to_device(retain_inputs, device)
        retain_outputs = model(retain_input_ids, attention_mask=retain_attention_mask)
        retain_logits = retain_outputs.logits[..., :-1, :].contiguous()
        retain_shifted_labels = retain_labels[..., 1:].contiguous()
        retain_loss = F.cross_entropy(
            retain_logits.view(-1, retain_logits.size(-1)),
            retain_shifted_labels.view(-1),
            ignore_index=-100,
        )

    return retain_loss


def compute_forget_rmu_total_loss(model, oracle_model, inputs,
                                  beta=1.0, gamma=1.0, alpha=1.0,
                                  rmu_layers=None, rmu_scale=300.0, rmu_lambda=1.0,
                                  random_vectors=None, retain_loss_type="ce",
                                  forget_loss_type="npo"):
    """Combined loss (for environments with enough VRAM)."""
    forget_inputs, retain_inputs = inputs
    forget_rmu_loss = compute_forget_rmu_loss(
        model, oracle_model, forget_inputs,
        beta=beta, gamma=gamma,
        rmu_layers=rmu_layers, rmu_scale=rmu_scale, rmu_lambda=rmu_lambda,
        random_vectors=random_vectors, forget_loss_type=forget_loss_type,
    )
    retain_loss = compute_retain_loss(model, oracle_model, retain_inputs, retain_loss_type)
    total_loss = forget_rmu_loss + alpha * retain_loss
    return total_loss, None


# Backward-compatible name for older dry runs/scripts.
def compute_npo_rmu_loss(model, oracle_model, inputs,
                         beta=1.0, gamma=1.0, alpha=1.0,
                         rmu_layers=None, rmu_scale=300.0, rmu_lambda=1.0,
                         random_vectors=None, retain_loss_type="ce",
                         forget_loss_type="npo"):
    return compute_forget_rmu_total_loss(
        model, oracle_model, inputs,
        beta=beta, gamma=gamma, alpha=alpha,
        rmu_layers=rmu_layers, rmu_scale=rmu_scale, rmu_lambda=rmu_lambda,
        random_vectors=random_vectors, retain_loss_type=retain_loss_type,
        forget_loss_type=forget_loss_type,
    )
