"""WMDP-style RMU activation loss for TOFU batches."""

import torch
import torch.nn.functional as F


def _model_device(model):
    return next(model.parameters()).device


def _batch_to_device(inputs, device):
    return tuple(tensor.to(device) for tensor in inputs)


def _masked_mse(source, target, attention_mask):
    mask = attention_mask.to(source.device).unsqueeze(-1).float()
    diff = (source.float() - target.float()) * mask
    active_hidden_count = mask.sum().clamp_min(1.0) * source.shape[-1]
    return diff.pow(2).sum() / active_hidden_count


def _masked_norm_mean(hidden_states, attention_mask):
    mask = attention_mask.to(hidden_states.device).float()
    norms = hidden_states.float().norm(dim=-1)
    return (norms * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_cosine_mean(left, right, attention_mask):
    mask = attention_mask.to(left.device).float()
    cosine = F.cosine_similarity(left.float(), right.float().to(left.device), dim=-1)
    return (cosine * mask).sum() / mask.sum().clamp_min(1.0)


def _layer_hidden(outputs, layer_idx):
    return outputs.hidden_states[int(layer_idx) + 1]


def compute_rmu_loss(updated_model, frozen_model, batch, cfg, control_vectors):
    """Compute RMU forget steering + frozen retain activation matching."""
    forget_inputs, retain_inputs = batch
    update_device = _model_device(updated_model)
    frozen_device = _model_device(frozen_model)
    activation_layers = [int(layer_idx) for layer_idx in cfg.activation_layers]

    forget_input_ids, _, forget_attention_mask = _batch_to_device(forget_inputs, update_device)
    retain_input_ids, _, retain_attention_mask = _batch_to_device(retain_inputs, update_device)

    forget_outputs = updated_model(
        input_ids=forget_input_ids,
        attention_mask=forget_attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    updated_retain_outputs = updated_model(
        input_ids=retain_input_ids,
        attention_mask=retain_attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    frozen_retain_input_ids, _, frozen_retain_attention_mask = _batch_to_device(retain_inputs, frozen_device)
    with torch.no_grad():
        frozen_retain_outputs = frozen_model(
            input_ids=frozen_retain_input_ids,
            attention_mask=frozen_retain_attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    forget_loss = torch.tensor(0.0, device=update_device)
    retain_loss = torch.tensor(0.0, device=update_device)
    forget_norm = torch.tensor(0.0, device=update_device)
    retain_cosine = torch.tensor(0.0, device=update_device)
    forget_control_cosine = torch.tensor(0.0, device=update_device)

    for layer_idx in activation_layers:
        updated_forget_hidden = _layer_hidden(forget_outputs, layer_idx)
        updated_retain_hidden = _layer_hidden(updated_retain_outputs, layer_idx)
        frozen_retain_hidden = _layer_hidden(frozen_retain_outputs, layer_idx).to(update_device)

        control_vector = control_vectors[layer_idx].to(update_device).float()
        control_target = control_vector.view(1, 1, -1).expand_as(updated_forget_hidden) * float(cfg.steering_coeff)

        forget_loss = forget_loss + _masked_mse(updated_forget_hidden, control_target, forget_attention_mask)
        retain_loss = retain_loss + _masked_mse(updated_retain_hidden, frozen_retain_hidden, retain_attention_mask)
        forget_norm = forget_norm + _masked_norm_mean(updated_forget_hidden, forget_attention_mask)
        retain_cosine = retain_cosine + _masked_cosine_mean(
            updated_retain_hidden,
            frozen_retain_hidden,
            retain_attention_mask,
        )
        forget_control_cosine = forget_control_cosine + _masked_cosine_mean(
            updated_forget_hidden,
            control_target,
            forget_attention_mask,
        )

    num_layers = max(1, len(activation_layers))
    forget_loss = forget_loss / num_layers
    retain_loss = retain_loss / num_layers
    total_loss = forget_loss + float(cfg.retain_alpha) * retain_loss

    logs = {
        "loss": total_loss.detach(),
        "forget_loss": forget_loss.detach(),
        "retain_loss": retain_loss.detach(),
        "forget_hidden_norm": (forget_norm / num_layers).detach(),
        "retain_cosine": (retain_cosine / num_layers).detach(),
        "forget_control_cosine": (forget_control_cosine / num_layers).detach(),
    }
    return total_loss, logs
