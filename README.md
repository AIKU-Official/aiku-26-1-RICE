# COSE461 RICE Unlearning Submission

이 저장소는 TOFU 기반 multilingual QA fine-tuning 모델에 baseline unlearning과 RICE 방법론을 적용하고 평가하기 위한 제출용 코드입니다.

최종 실행 흐름은 다음과 같습니다.

```text
TOFU fine-tune -> baseline/RICE unlearning -> evaluation
```

## Included Methods

남긴 방법론은 세 가지입니다.

```text
unlearning_methods/
  unlearn_npo/          # baseline: NPO
  unlearn_grad_diff/    # baseline: Grad-Diff
  unlearn_rice/         # proposed method: RICE
```

RICE는 NPO forget loss, RMU hidden-state regularization, retain CE loss를 결합한 방법론입니다. 기본 설정은 `unlearning_methods/unlearn_rice/config.yaml`에 있습니다.

## Setup

Python 3.10 환경에서 아래 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

CUDA/PyTorch 버전은 `requirements.txt`와 `environment.yml`을 기준으로 맞춥니다. Qwen3.5-2B-Base를 기본 모델로 사용하며, 모델 체크포인트는 제출물에 포함하지 않습니다.

## Data

제출물에는 실행에 필요한 TOFU-derived 데이터만 포함합니다.

```text
dataset/full_merged_all_10_lang          # full fine-tuning
dataset/retain99_merged_all_10_lang      # retain99 baseline fine-tuning
dataset/forget01*                        # forget set and perturbed forget set
dataset/retain_perturbed*                # retain evaluation
dataset/real_authors_perturbed*          # real-author evaluation
dataset/world_facts_perturbed*           # real-world evaluation
```

English TOFU splits are loaded from `locuslab/TOFU` through Hugging Face `datasets`; multilingual splits are loaded from local `dataset/`.

## Run

Fine-tune the full model:

```bash
bash scripts/run_finetune_100.sh
```

Fine-tune the retain99 baseline model used for Forget Quality:

```bash
bash scripts/run_finetune_99.sh
```

Both fine-tuning scripts support multi-GPU training through `FT_GPUS` and run evaluation after training succeeds:

```bash
FT_GPUS=0,1 bash scripts/run_finetune_100.sh
FT_GPUS=0,1 bash scripts/run_finetune_99.sh
```

Run baselines:

```bash
python unlearning_methods/unlearn_npo/train.py model_path=./finetuned/finetuned_100
python unlearning_methods/unlearn_grad_diff/train.py model_path=./finetuned/finetuned_100
```

Run RICE:

```bash
CUDA_VISIBLE_DEVICES=0,1 python unlearning_methods/unlearn_rice/train.py \
  model_path=./finetuned/finetuned_100 \
  save_dir=./results/rice
```

기본 RICE는 NPO reference/oracle 모델을 함께 사용하므로 visible CUDA device 2개가 필요합니다.

Evaluate a model:

```bash
python evaluate.py \
  model_path=./results/rice \
  save_dir=./results/eval_rice \
  retain_result_template='./results/eval_finetuned_99/{language}/eval_log_aggregated.json'
```

Run the full submission pipeline:

```bash
bash scripts/run_full_pipeline.sh
```

Run only baseline training/evaluation:

```bash
bash scripts/run_baselines.sh
```

Run RICE ablations:

```bash
bash unlearning_methods/unlearn_rice/run_ablations.sh
```

Useful GPU overrides:

```bash
FT_GPUS=0,1 RICE_GPUS=0,1 EVAL_GPU=0 bash scripts/run_full_pipeline.sh
NPO_GPUS=0,1 bash scripts/run_baselines.sh
TRAIN_GPU=0 EVAL_GPU=0 bash unlearning_methods/unlearn_rice/run_ablation_ga_forget.sh
```

## RICE Ablations

RICE ablation scripts are under `unlearning_methods/unlearn_rice/`.

```text
run_ablation_no_retain.sh     # NPO + RMU, alpha=0.0
run_ablation_ga_forget.sh     # GA forget + RMU + retain CE
run_ablation_all_layers.sh    # NPO + RMU on all transformer layers + retain CE
run_ablations.sh              # runs all ablations
```

All scripts run from the repository root and write generated models/evaluation results under `./results/`.

## Smoke Checks

Syntax/import check:

```bash
python -m compileall .
```

Small evaluation smoke test:

```bash
python evaluate.py \
  model_path=./finetuned/finetuned_99 \
  save_dir=./results/smoke_eval_finetuned_99 \
  ds_size=1 \
  'languages=[en]'
```

Small RICE training smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,1 python unlearning_methods/unlearn_rice/train.py \
  model_path=./finetuned/finetuned_100 \
  save_dir=./results/smoke_rice \
  num_epochs=1 \
  batch_size=1 \
  gradient_accumulation_steps=1
```

## Outputs

Model checkpoints and generated evaluation results are intentionally excluded by `.gitignore`.

```text
finetuned/
results/
outputs/
```

Regenerate them with the commands above.

## Attribution and License

This project is based on the MIT-licensed repository [alirezafarashah/multilingual_unlearning](https://github.com/alirezafarashah/multilingual_unlearning), which builds on TOFU resources from [locuslab/tofu](https://github.com/locuslab/tofu).

The original MIT license notice is preserved in `LICENSE`, with an additional modification notice for the COSE461 project changes.
