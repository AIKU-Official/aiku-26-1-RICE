#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/_common.sh"
ensure_python_bin
MODEL_FAMILY="${MODEL_FAMILY:-qwen3_5_2b}"
TRAIN_GPU="${TRAIN_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-${TRAIN_GPU}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MODEL_PATH="${MODEL_PATH:-./finetuned/finetuned_100}"
RETAIN_MODEL_PATH="${RETAIN_MODEL_PATH:-./finetuned/finetuned_99}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
TRAIN_DIR="${TRAIN_DIR:-${RESULTS_DIR}/rice_ablation_ga_forget}"
EVAL_DIR="${EVAL_DIR:-${RESULTS_DIR}/eval_rice_ablation_ga_forget}"
RETAIN_EVAL_DIR="${RETAIN_EVAL_DIR:-${RESULTS_DIR}/eval_finetuned_99}"
RETAIN_TEMPLATE="${RETAIN_TEMPLATE:-${RETAIN_EVAL_DIR}/{language}/eval_log_aggregated.json}"

export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Running RICE ablation: GA forget loss + RMU + retain CE"

if [ ! -f "${TRAIN_DIR}/model.safetensors" ]; then
    mkdir -p "${TRAIN_DIR}"
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${PYTHON_BIN}" unlearning_methods/unlearn_rice/train.py \
        model_path="${MODEL_PATH}" \
        forget_loss_type=ga \
        retain_loss_type=ce \
        alpha=0.5 \
        method_name=rice_ablation_ga_forget \
        save_dir="${TRAIN_DIR}" \
        > "${TRAIN_DIR}/train.log" 2>&1
else
    echo "Model already exists: ${TRAIN_DIR}"
fi

run_model_eval "${EVAL_GPU}" "${RETAIN_MODEL_PATH}" "${RETAIN_EVAL_DIR}" ""
run_model_eval "${EVAL_GPU}" "${TRAIN_DIR}" "${EVAL_DIR}" "${RETAIN_TEMPLATE}"

echo "Done. Summary: ${EVAL_DIR}/eval_summary.json"
