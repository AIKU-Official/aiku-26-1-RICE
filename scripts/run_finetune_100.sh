#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/_common.sh"
ensure_python_bin

MODEL_FAMILY="${MODEL_FAMILY:-qwen3_5_2b}"
FT_GPUS="${FT_GPUS:-${FT_GPU:-0}}"
IFS=',' read -r -a FT_GPU_LIST <<< "${FT_GPUS}"
FT_NPROC="${FT_NPROC:-${#FT_GPU_LIST[@]}}"
MASTER_PORT="${MASTER_PORT:-29510}"
FINETUNED_100="${FINETUNED_100:-./finetuned/finetuned_100}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
EVAL_GPU="${EVAL_GPU:-${FT_GPU_LIST[0]}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_DIR="${EVAL_DIR:-${RESULTS_DIR}/eval_finetuned_100}"
RUN_EVAL="${RUN_EVAL:-true}"

export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_eval() {
    if [ "${RUN_EVAL}" != "true" ]; then
        echo "Evaluation disabled: RUN_EVAL=${RUN_EVAL}"
        return 0
    fi

    run_model_eval "${EVAL_GPU}" "${FINETUNED_100}" "${EVAL_DIR}" ""
    echo "Evaluation complete: ${EVAL_DIR}"
}

echo "========================================"
echo "Full TOFU fine-tuning started at $(date)"
echo "  repo: ${REPO_ROOT}"
echo "  output: ${FINETUNED_100}"
echo "  GPUs: ${FT_GPUS} (nproc=${FT_NPROC}, master_port=${MASTER_PORT})"
echo "========================================"

if has_model "${FINETUNED_100}"; then
    echo "Fine-tuned full model already exists: ${FINETUNED_100}"
else
    mkdir -p "${FINETUNED_100}"
    echo "Fine-tuning full TOFU model -> ${FINETUNED_100}"
    CUDA_VISIBLE_DEVICES="${FT_GPUS}" "${PYTHON_BIN}" -m torch.distributed.run \
        --nproc_per_node="${FT_NPROC}" \
        --master_port="${MASTER_PORT}" \
        finetune.py \
        save_dir="${FINETUNED_100}" \
        > "${FINETUNED_100}/finetune.log" 2>&1
    echo "Full TOFU fine-tuning complete: ${FINETUNED_100}"
fi

run_eval

echo "Full TOFU fine-tuning script finished at $(date)"
