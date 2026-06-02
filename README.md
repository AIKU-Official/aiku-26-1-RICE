# RICE: Reference-guided Internal Cross-lingual Erasure for Multilingual LLM Unlearning

This repository contains the code and data used for the experiments in the accompanying RICE paper. It provides a reproducible pipeline for multilingual TOFU fine-tuning, unlearning, baseline comparison, and evaluation.

RICE is a multilingual LLM unlearning method designed to remove target factual behavior while preserving non-target utility. The repository includes the proposed method, baseline methods, translated TOFU-derived datasets, and scripts for running the main experiments. Methodological details are described in the paper.

## Repository Structure

```text
config/                  Model, fine-tuning, DeepSpeed, and evaluation configs
dataset/                 TOFU-derived multilingual datasets used by the scripts
scripts/                 End-to-end and convenience run scripts
unlearning_methods/
  unlearn_rice/          RICE implementation
  unlearn_npo/           NPO baseline
  unlearn_grad_diff/     Grad-Diff baseline
finetune.py              Fine-tuning entry point
evaluate.py              Evaluation entry point
data_module.py           Dataset formatting utilities
utils.py                 Evaluation and aggregation utilities
finetuned/               Generated fine-tuned checkpoints, not tracked by git
results/                 Generated unlearning and evaluation outputs, not tracked by git
```


## Setup

Use Python 3.10 with a CUDA-enabled PyTorch environment. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The repository also includes `environment.yml` as a reference environment file.

The default model family is `qwen3_5_2b`, configured in `config/model_config.yaml`. Generated model checkpoints are not included in the repository.

## Data and Checkpoints

The `dataset/` directory contains the local TOFU-derived multilingual splits used by the training and evaluation scripts:

```text
dataset/full_merged_all_10_lang
dataset/retain99_merged_all_10_lang
dataset/forget01_*
dataset/forget01_perturbed_*
dataset/retain_perturbed_*
dataset/real_authors_perturbed_*
dataset/world_facts_perturbed_*
```

The scripts generate checkpoints under `finetuned/` and experiment outputs under `results/`. These directories are intentionally excluded from git except for `finetuned/.gitkeep`.

## Running Experiments

Run the full fine-tuning, unlearning, and evaluation pipeline:

```bash
bash scripts/run_full_pipeline.sh
```

Useful GPU overrides:

```bash
FT_GPUS=0,1 RICE_GPUS=0,1 EVAL_GPU=0 bash scripts/run_full_pipeline.sh
```

RICE training uses a frozen oracle/reference model in addition to the trainable model, so the default RICE run expects two visible CUDA devices.

Fine-tune the full-data model:

```bash
bash scripts/run_finetune_100.sh
```

Fine-tune the retain99 reference model:

```bash
bash scripts/run_finetune_99.sh
```

Run the baseline unlearning methods:

```bash
bash scripts/run_baselines.sh
```

Run RICE directly:

```bash
CUDA_VISIBLE_DEVICES=0,1 python unlearning_methods/unlearn_rice/train.py \
  model_path=./finetuned/finetuned_100 \
  save_dir=./results/rice
```

Evaluate a trained or unlearned model:

```bash
python evaluate.py \
  model_path=./results/rice \
  save_dir=./results/eval_rice \
  retain_result_template='./results/eval_finetuned_99/{language}/eval_log_aggregated.json'
```

Run RICE ablations:

```bash
bash unlearning_methods/unlearn_rice/run_ablations.sh
```

## Outputs

The main scripts produce the following artifacts:

```text
finetuned/finetuned_100          Full-data fine-tuned model
finetuned/finetuned_99           Retain99 fine-tuned reference model
results/npo                     NPO baseline checkpoint
results/grad_diff               Grad-Diff baseline checkpoint
results/rice                    RICE checkpoint
results/eval_*                  Evaluation logs, generated text, and summaries
```

Evaluation outputs include per-task JSON logs and aggregate summaries. The exact files depend on the selected evaluation config and language list.

## Smoke Checks

Run a syntax/import check:

```bash
python -m compileall .
```

Run a small evaluation smoke test:

```bash
python evaluate.py \
  model_path=./finetuned/finetuned_99 \
  save_dir=./results/smoke_eval_finetuned_99 \
  ds_size=1 \
  'languages=[en]'
```

Run a small RICE training smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,1 python unlearning_methods/unlearn_rice/train.py \
  model_path=./finetuned/finetuned_100 \
  save_dir=./results/smoke_rice \
  num_epochs=1 \
  batch_size=1 \
  gradient_accumulation_steps=1
```

## Citation

If you use this repository, please cite the accompanying paper:

```bibtex
@misc{rice2026,
  title = {RICE: Reference-guided Internal Cross-lingual Erasure for Multilingual LLM Unlearning},
  note = {Manuscript},
  year = {2026}
}
```

## License and Attribution

This project is based on the MIT-licensed repository `alirezafarashah/multilingual_unlearning`, which builds on TOFU resources from `locuslab/tofu`. The original MIT license notice is preserved in `LICENSE`, with an additional modification notice for the COSE461 project changes.
