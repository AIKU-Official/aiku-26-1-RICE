"""End-to-end configurable forget+RMU unlearning runner.

Custom training loop with chunked backward for Titan Xp (12GB, fp16).
Forget+RMU backward and retain backward are done separately to halve peak activation memory.

Run from the repo root with:
    python unlearning_methods/unlearn_rice/train.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bitsandbytes as bnb
import hydra
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from unlearning_methods.unlearn_rice.dataloader import NPORMUForgetDataset, npo_rmu_collator
from unlearning_methods.unlearn_rice.loss import compute_forget_rmu_loss, compute_retain_loss
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


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "true"

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]
    cfg.model_path = resolve_project_path(cfg.model_path)
    cfg.save_dir = resolve_project_path(cfg.save_dir)

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = NPORMUForgetDataset(
        cfg.data_path,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=500,
        split=cfg.split,
        language=cfg.language,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=npo_rmu_collator,
        num_workers=0,
    )

    batch_size = cfg.batch_size
    grad_accum = cfg.gradient_accumulation_steps
    max_steps = max(1, int(cfg.num_epochs * len(dataset)) // (batch_size * grad_accum))
    log_interval = max(1, max_steps // 20)
    print(f"Dataset size: {len(dataset)}, batch_size: {batch_size}, grad_accum: {grad_accum}")
    print(f"max_steps: {max_steps}, log_interval: {log_interval}")

    # Titan Xp: fp16 (no bf16 support on Pascal)
    attn_impl = "flash_attention_2" if model_cfg.get("flash_attention2", "false") == "true" else None
    print(f"Loading model to cuda:0 (attn={attn_impl})...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        attn_implementation=attn_impl,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda:0")

    forget_loss_type = str(cfg.get("forget_loss_type", "npo")).lower()
    retain_loss_type = str(cfg.get("retain_loss_type", "ce")).lower()

    needs_oracle = forget_loss_type == "npo" or retain_loss_type == "kl"
    if needs_oracle:
        print("Loading oracle model to cuda:1...")
        oracle_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            attn_implementation=attn_impl,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda:1")
        oracle_model.eval()
    else:
        oracle_model = None
        print("Skipping oracle model: GA forget loss with CE retain does not use it.")

    if model_cfg.get("gradient_checkpointing", "false") == "true":
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    model.config.use_cache = False

    # RMU random vector initialization
    hidden_size = model.config.hidden_size
    random_vectors = {}
    for layer_idx in cfg.rmu_layers:
        rv = torch.randn(hidden_size)
        random_vectors[layer_idx] = rv / rv.norm()
    print(f"RMU initialized: layers={list(cfg.rmu_layers)}, hidden_size={hidden_size}, scale={cfg.rmu_scale}")

    # Optimizer (paged to offload states to CPU)
    optimizer = bnb.optim.PagedAdamW32bit(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Warmup scheduler
    steps_per_epoch = max(1, len(dataset) // (batch_size * grad_accum))
    warmup_steps = max(1, steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    alpha = cfg.alpha
    print(f"forget_loss_type: {forget_loss_type}")
    print(f"retain_loss_type: {retain_loss_type}")
    print(f"Training params: beta={cfg.beta}, gamma={cfg.gamma}, alpha={alpha}")
    print(f"RMU params: layers={list(cfg.rmu_layers)}, scale={cfg.rmu_scale}, lambda={cfg.rmu_lambda}")
    print("=" * 60)
    print("Starting training (chunked backward + NaN gradient sanitization)...")

    def sanitize_gradients(model):
        """Replace NaN/Inf gradients with zeros (fp16 linear_attn backward workaround)."""
        for p in model.parameters():
            if p.grad is not None:
                bad = ~torch.isfinite(p.grad.data)
                if bad.any():
                    p.grad.data[bad] = 0.0

    model.train()
    global_step = 0
    accum_loss_forget = 0.0
    accum_loss_retain = 0.0
    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(dataloader):
            forget_inputs, retain_inputs = batch

            # --- Phase 1: Forget + RMU forward & backward ---
            forget_rmu_loss = compute_forget_rmu_loss(
                model, oracle_model, forget_inputs,
                beta=cfg.beta, gamma=cfg.gamma,
                rmu_layers=list(cfg.rmu_layers),
                rmu_scale=cfg.rmu_scale, rmu_lambda=cfg.rmu_lambda,
                random_vectors=random_vectors,
                forget_loss_type=forget_loss_type,
            )
            (forget_rmu_loss / grad_accum).backward()
            sanitize_gradients(model)
            accum_loss_forget += forget_rmu_loss.item() / grad_accum
            del forget_rmu_loss
            torch.cuda.empty_cache()

            # --- Phase 2: Retain forward & backward ---
            retain_loss = compute_retain_loss(
                model, oracle_model, retain_inputs,
                retain_loss_type=retain_loss_type,
            )
            (alpha * retain_loss / grad_accum).backward()
            sanitize_gradients(model)
            accum_loss_retain += retain_loss.item() / grad_accum
            del retain_loss
            torch.cuda.empty_cache()

            # --- Optimizer step ---
            if (step + 1) % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_interval == 0 or global_step == 1:
                    lr_now = scheduler.get_last_lr()[0]
                    gpu_mem = torch.cuda.memory_allocated(0) / 1e9
                    total_loss = accum_loss_forget + accum_loss_retain
                    print(f"  Step {global_step}/{max_steps} | loss={total_loss:.4f} "
                          f"(forget+rmu={accum_loss_forget:.4f}, retain={accum_loss_retain:.4f}) "
                          f"| lr={lr_now:.2e} | GPU0={gpu_mem:.2f}GB")

                accum_loss_forget = 0.0
                accum_loss_retain = 0.0

                if global_step >= max_steps:
                    break

        if global_step >= max_steps:
            break

    print("=" * 60)
    print(f"Training complete. Global step: {global_step}")

    # Save model
    if cfg.save_model:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving model to {cfg.save_dir}...")
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)
        print("Model saved.")

    del model, oracle_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
