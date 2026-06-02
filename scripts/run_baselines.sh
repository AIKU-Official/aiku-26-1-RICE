#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/_common.sh"
ensure_python_bin
MODEL_FAMILY="${MODEL_FAMILY:-qwen3_5_2b}"
MODEL_PATH="${MODEL_PATH:-./finetuned/finetuned_100}"
RETAIN_MODEL_PATH="${RETAIN_MODEL_PATH:-./finetuned/finetuned_99}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
NPO_GPUS="${NPO_GPUS:-0,1}"
GRAD_DIFF_GPU="${GRAD_DIFF_GPU:-0}"
GRAD_DIFF_KL_GPUS="${GRAD_DIFF_KL_GPUS:-0,1}"
RMU_GPUS="${RMU_GPUS:-0,1}"
EVAL_GPU="${EVAL_GPU:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
RETAIN_EVAL_DIR="${RETAIN_EVAL_DIR:-${RESULTS_DIR}/eval_finetuned_99}"
RETAIN_TEMPLATE="${RETAIN_TEMPLATE:-${RETAIN_EVAL_DIR}/{language}/eval_log_aggregated.json}"

export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Running baseline methods from ${REPO_ROOT}"

if has_model "${RESULTS_DIR}/npo"; then
    echo "NPO baseline already exists: ${RESULTS_DIR}/npo"
else
    mkdir -p "${RESULTS_DIR}/npo"
    CUDA_VISIBLE_DEVICES="${NPO_GPUS}" "${PYTHON_BIN}" unlearning_methods/unlearn_npo/train.py \
        model_path="${MODEL_PATH}" \
        save_dir="${RESULTS_DIR}/npo" \
        > "${RESULTS_DIR}/npo/train.log" 2>&1
fi

if has_model "${RESULTS_DIR}/grad_diff"; then
    echo "Grad-Diff baseline already exists: ${RESULTS_DIR}/grad_diff"
else
    mkdir -p "${RESULTS_DIR}/grad_diff"
    CUDA_VISIBLE_DEVICES="${GRAD_DIFF_GPU}" "${PYTHON_BIN}" unlearning_methods/unlearn_grad_diff/train.py \
        model_path="${MODEL_PATH}" \
        save_dir="${RESULTS_DIR}/grad_diff" \
        > "${RESULTS_DIR}/grad_diff/train.log" 2>&1
fi

if has_model "${RESULTS_DIR}/grad_diff_kl"; then
    echo "Grad-Diff-KL baseline already exists: ${RESULTS_DIR}/grad_diff_kl"
else
    mkdir -p "${RESULTS_DIR}/grad_diff_kl"
    CUDA_VISIBLE_DEVICES="${GRAD_DIFF_KL_GPUS}" "${PYTHON_BIN}" unlearning_methods/unlearn_grad_diff_kl/train.py \
        model_path="${MODEL_PATH}" \
        save_dir="${RESULTS_DIR}/grad_diff_kl" \
        > "${RESULTS_DIR}/grad_diff_kl/train.log" 2>&1
fi

if has_model "${RESULTS_DIR}/rmu"; then
    echo "RMU baseline already exists: ${RESULTS_DIR}/rmu"
else
    mkdir -p "${RESULTS_DIR}/rmu"
    CUDA_VISIBLE_DEVICES="${RMU_GPUS}" "${PYTHON_BIN}" unlearning_methods/unlearn_rmu/train.py \
        model_path="${MODEL_PATH}" \
        save_dir="${RESULTS_DIR}/rmu" \
        > "${RESULTS_DIR}/rmu/train.log" 2>&1
fi

run_model_eval "${EVAL_GPU}" "${RETAIN_MODEL_PATH}" "${RETAIN_EVAL_DIR}" ""
run_model_eval "${EVAL_GPU}" "${RESULTS_DIR}/npo" "${RESULTS_DIR}/eval_npo" "${RETAIN_TEMPLATE}"
run_model_eval "${EVAL_GPU}" "${RESULTS_DIR}/grad_diff" "${RESULTS_DIR}/eval_grad_diff" "${RETAIN_TEMPLATE}"
run_model_eval "${EVAL_GPU}" "${RESULTS_DIR}/grad_diff_kl" "${RESULTS_DIR}/eval_grad_diff_kl" "${RETAIN_TEMPLATE}"
run_model_eval "${EVAL_GPU}" "${RESULTS_DIR}/rmu" "${RESULTS_DIR}/eval_rmu" "${RETAIN_TEMPLATE}"

echo "Baseline pipeline complete."
