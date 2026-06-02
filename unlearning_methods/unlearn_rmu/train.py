"""End-to-end WMDP-style RMU unlearning runner for TOFU.

Run from the NLP_rice root:
    CUDA_VISIBLE_DEVICES=0,1 python unlearning_methods/unlearn_RMU/train.py \
        model_path=/path/to/full_tofu_finetuned_model
"""

import json
import math
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from unlearning_methods.unlearn_RMU.dataloader import RMUForgetDataset, rmu_collator
from unlearning_methods.unlearn_RMU.loss import compute_rmu_loss
from utils import get_model_identifiers_from_yaml


def resolve_project_path(path):
    if path is None:
        return None
    path = str(path)
    if path.startswith("/"):
        return path
    if path.startswith(("./", "../")):
        return str((PROJECT_ROOT / path).resolve())
    return path


def pick_dtype(cfg):
    if not torch.cuda.is_available():
        return torch.float32
    if bool(cfg.get("bf16", True)) and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if bool(cfg.get("fp16", False)) or bool(cfg.get("bf16", True)):
        return torch.float16
    return torch.float32


def pick_device(value, fallback=None):
    if value is None:
        return fallback or torch.device("cpu")
    value = str(value).lower()
    if value == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if value.startswith("cuda:"):
        index = int(value.split(":", 1)[1])
    else:
        index = int(value)
    if index >= torch.cuda.device_count():
        raise ValueError(f"Requested cuda:{index}, but only {torch.cuda.device_count()} visible CUDA devices exist.")
    return torch.device(f"cuda:{index}")


def resolve_teacher_device(cfg, train_device):
    value = str(cfg.get("teacher_device", "auto")).lower()
    if value == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            return torch.device("cuda:1")
        return train_device
    if value == "same_as_student":
        return train_device
    return pick_device(value, fallback=train_device)


def load_tokenizer(cfg, model_id):
    model_path = resolve_project_path(cfg.model_path)
    source = model_path if Path(model_path, "tokenizer_config.json").exists() else model_id
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(model_path, model_cfg, dtype, device):
    load_dtype = torch.float32 if device.type == "cpu" else dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
        torch_dtype=load_dtype,
        trust_remote_code=True,
    )
    return model.to(device)


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise AttributeError("Could not find transformer layers. Expected model.model.layers for Qwen-style models.")


def validate_layers(layers, layer_indices, name):
    num_layers = len(layers)
    for layer_idx in layer_indices:
        if int(layer_idx) < 0 or int(layer_idx) >= num_layers:
            raise ValueError(f"{name} contains invalid layer {layer_idx}; model has layers 0..{num_layers - 1}.")


def set_trainable_layers(model, update_layers):
    for param in model.parameters():
        param.requires_grad = False

    layers = get_transformer_layers(model)
    validate_layers(layers, update_layers, "update_layers")
    for layer_idx in update_layers:
        for param in layers[int(layer_idx)].parameters():
            param.requires_grad = True

    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters selected. Check update_layers.")
    return trainable


def ensure_save_dir(cfg):
    save_dir = Path(resolve_project_path(cfg.save_dir))
    if save_dir.exists() and any(save_dir.iterdir()) and not bool(cfg.overwrite_dir):
        raise FileExistsError(
            f"save_dir already exists and is not empty: {save_dir}\n"
            "Set overwrite_dir=true or choose a new save_dir."
        )
    if save_dir.exists() and bool(cfg.overwrite_dir):
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def initialize_control_vectors(hidden_size, activation_layers, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    vectors = {}
    for layer_idx in activation_layers:
        vector = torch.randn(hidden_size, generator=generator)
        vectors[int(layer_idx)] = vector / vector.norm().clamp_min(1e-12)
    return vectors


def write_jsonl(handle, record):
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    os.chdir(PROJECT_ROOT)
    set_seed(int(cfg.seed))
    os.environ["WANDB_DISABLED"] = "true"

    cfg.model_path = resolve_project_path(cfg.model_path)
    cfg.save_dir = resolve_project_path(cfg.save_dir)
    model_path = Path(cfg.model_path)
    if not model_path.exists() or not Path(model_path, "config.json").exists():
        raise FileNotFoundError(
            f"model_path must point to a saved fine-tuned checkpoint with config.json: {model_path}"
        )

    save_dir = ensure_save_dir(cfg)
    with open(save_dir / "resolved_config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    dtype = pick_dtype(cfg)
    train_device = pick_device("0") if torch.cuda.is_available() else torch.device("cpu")
    teacher_device = resolve_teacher_device(cfg, train_device)

    print("######################")
    print("RMU save_dir:", save_dir)
    print("model_path:", cfg.model_path)
    print("train_device:", train_device, "teacher_device:", teacher_device, "dtype:", dtype)
    print("######################")

    tokenizer = load_tokenizer(cfg, model_id)

    dataset = RMUForgetDataset(
        cfg.data_path,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=int(cfg.max_length),
        split=cfg.split,
        language=cfg.language,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        collate_fn=rmu_collator,
        num_workers=0,
    )

    updated_model = build_model(cfg.model_path, model_cfg, dtype, train_device)
    frozen_model = build_model(cfg.model_path, model_cfg, dtype, teacher_device)
    frozen_model.eval()
    for param in frozen_model.parameters():
        param.requires_grad = False

    if bool(cfg.get("gradient_checkpointing", True)) and model_cfg.get("gradient_checkpointing", "false") == "true":
        updated_model.gradient_checkpointing_enable()

    updated_model.config.use_cache = False
    frozen_model.config.use_cache = False

    layers = get_transformer_layers(updated_model)
    activation_layers = [int(layer_idx) for layer_idx in cfg.activation_layers]
    update_layers = [int(layer_idx) for layer_idx in cfg.update_layers]
    validate_layers(layers, activation_layers, "activation_layers")
    trainable_params = set_trainable_layers(updated_model, update_layers)
    trainable_count = sum(param.numel() for param in trainable_params)
    total_count = sum(param.numel() for param in updated_model.parameters())

    hidden_size = int(getattr(updated_model.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise ValueError("Model config does not expose a valid hidden_size.")
    control_vectors = initialize_control_vectors(hidden_size, activation_layers, cfg.seed)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    grad_accum = max(1, int(cfg.gradient_accumulation_steps))
    optimizer_steps_per_epoch = max(1, math.ceil(len(dataloader) / grad_accum))
    configured_max_steps = cfg.get("max_steps", None)
    max_steps = (
        int(configured_max_steps)
        if configured_max_steps is not None
        else max(1, int(float(cfg.num_epochs) * optimizer_steps_per_epoch))
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Trainable params: {trainable_count}/{total_count}")
    print(f"activation_layers={activation_layers}, update_layers={update_layers}")
    print(f"max_steps={max_steps}, optimizer_steps_per_epoch={optimizer_steps_per_epoch}")

    updated_model.train()
    global_step = 0
    micro_step = 0
    accum_logs = {}
    optimizer.zero_grad(set_to_none=True)
    log_path = save_dir / "train_log.jsonl"

    with open(log_path, "w") as log_handle:
        for epoch in range(int(math.ceil(float(cfg.num_epochs)))):
            for batch_idx, batch in enumerate(dataloader):
                loss, logs = compute_rmu_loss(updated_model, frozen_model, batch, cfg, control_vectors)
                (loss / grad_accum).backward()
                micro_step += 1

                for key, value in logs.items():
                    accum_logs[key] = accum_logs.get(key, 0.0) + float(value.detach().float().cpu()) / grad_accum

                is_last_batch = batch_idx == len(dataloader) - 1
                should_step = micro_step % grad_accum == 0 or is_last_batch
                if not should_step:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(cfg.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                record = {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "global_step": global_step,
                    "grad_norm": float(grad_norm.detach().float().cpu()),
                    **accum_logs,
                }
                write_jsonl(log_handle, record)

                if global_step == 1 or global_step % max(1, int(cfg.log_steps)) == 0:
                    print(
                        f"step {global_step}/{max_steps} | "
                        f"loss={record['loss']:.4f} | "
                        f"forget={record['forget_loss']:.4f} | "
                        f"retain={record['retain_loss']:.4f} | "
                        f"grad={record['grad_norm']:.4f}"
                    )

                accum_logs = {}
                if global_step >= max_steps:
                    break

            if global_step >= max_steps:
                break

    print(f"Training complete. global_step={global_step}")

    if bool(cfg.save_model):
        print(f"Saving RMU model to {save_dir}")
        updated_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("Model saved.")


if __name__ == "__main__":
    main()
