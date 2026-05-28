#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/_common.sh"
ensure_python_bin
MODEL_FAMILY="${MODEL_FAMILY:-qwen3_5_2b}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
FINETUNED_100="${FINETUNED_100:-./finetuned/finetuned_100}"
FINETUNED_99="${FINETUNED_99:-./finetuned/finetuned_99}"

FT_GPUS="${FT_GPUS:-${FT_GPU:-0}}"
NPO_GPUS="${NPO_GPUS:-0,1}"
GRAD_DIFF_GPU="${GRAD_DIFF_GPU:-0}"
RICE_GPUS="${RICE_GPUS:-${RICE_GPU:-0,1}}"
EVAL_GPU="${EVAL_GPU:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
RETAIN_EVAL_DIR="${RETAIN_EVAL_DIR:-${RESULTS_DIR}/eval_finetuned_99}"
RETAIN_TEMPLATE="${RETAIN_TEMPLATE:-${RETAIN_EVAL_DIR}/{language}/eval_log_aggregated.json}"

export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Running full submission pipeline from ${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN}" FT_GPUS="${FT_GPUS}" FINETUNED_100="${FINETUNED_100}" \
    RESULTS_DIR="${RESULTS_DIR}" EVAL_GPU="${EVAL_GPU}" EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    bash scripts/run_finetune_100.sh
PYTHON_BIN="${PYTHON_BIN}" FT_GPUS="${FT_GPUS}" FINETUNED_99="${FINETUNED_99}" \
    RESULTS_DIR="${RESULTS_DIR}" EVAL_GPU="${EVAL_GPU}" EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    bash scripts/run_finetune_99.sh

echo "Training baseline and RICE methods"

if has_model "${RESULTS_DIR}/npo"; then
    echo "NPO baseline already exists: ${RESULTS_DIR}/npo"
else
    mkdir -p "${RESULTS_DIR}/npo"
    CUDA_VISIBLE_DEVICES="${NPO_GPUS}" "${PYTHON_BIN}" unlearning_methods/unlearn_npo/train.py \
        model_path="${FINETUNED_100}" \
        save_dir="${RESULTS_DIR}/npo" \
        > "${RESULTS_DIR}/npo/train.log" 2>&1
fi

if has_model "${RESULTS_DIR}/grad_diff"; then
    echo "Grad-Diff baseline already exists: ${RESULTS_DIR}/grad_diff"
else
    mkdir -p "${RESULTS_DIR}/grad_diff"
    CUDA_VISIBLE_DEVICES="${GRAD_DIFF_GPU}" "${PYTHON_BIN}" unlearning_methods/unlearn_grad_diff/train.py \
        model_path="${FINETUNED_100}" \
        save_dir="${RESULTS_DIR}/grad_diff" \
        > "${RESULTS_DIR}/grad_diff/train.log" 2>&1
fi

if has_model "${RESULTS_DIR}/rice"; then
    echo "RICE model already exists: ${RESULTS_DIR}/rice"
else
    mkdir -p "${RESULTS_DIR}/rice"
    CUDA_VISIBLE_DEVICES="${RICE_GPUS}" "${PYTHON_BIN}" unlearning_methods/unlearn_rice/train.py \
        model_path="${FINETUNED_100}" \
        forget_loss_type=npo \
        retain_loss_type=ce \
        alpha=0.5 \
        save_dir="${RESULTS_DIR}/rice" \
        > "${RESULTS_DIR}/rice/train.log" 2>&1
fi

echo "Evaluating retain baseline, baselines, and RICE"
run_model_eval "${EVAL_GPU}" "${FINETUNED_99}" "${RETAIN_EVAL_DIR}" ""
run_model_eval "${EVAL_GPU}" "${RESULTS_DIR}/npo" "${RESULTS_DIR}/eval_npo" "${RETAIN_TEMPLATE}"
run_model_eval "${EVAL_GPU}" "${RESULTS_DIR}/grad_diff" "${RESULTS_DIR}/eval_grad_diff" "${RETAIN_TEMPLATE}"
run_model_eval "${EVAL_GPU}" "${RESULTS_DIR}/rice" "${RESULTS_DIR}/eval_rice" "${RETAIN_TEMPLATE}"

echo "Full pipeline complete. Evaluation summaries are under ${RESULTS_DIR}/eval_*."
