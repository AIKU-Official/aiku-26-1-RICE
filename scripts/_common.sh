#!/usr/bin/env bash

ensure_python_bin() {
    if [ -n "${PYTHON_BIN:-}" ]; then
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        PYTHON_BIN="python3"
    fi
    export PYTHON_BIN
}

has_model() {
    local dir="$1"
    [ -f "${dir}/model.safetensors" ] || [ -f "${dir}/pytorch_model.bin" ]
}

run_model_eval() {
    local gpu_id="$1"
    local model_path="$2"
    local save_dir="$3"
    local retain_tpl="${4:-}"

    if [ -f "${save_dir}/eval_summary.json" ]; then
        echo "Eval already exists: ${save_dir}"
        return 0
    fi

    mkdir -p "${save_dir}"
    local extra_args=()
    if [ -n "${retain_tpl}" ]; then
        extra_args=("retain_result_template=${retain_tpl}")
    fi

    echo "Evaluating ${model_path} -> ${save_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" evaluate.py \
        model_path="${model_path}" \
        model_family="${MODEL_FAMILY}" \
        batch_size="${EVAL_BATCH_SIZE}" \
        save_dir="${save_dir}" \
        overwrite=true \
        "${extra_args[@]}" \
        > "${save_dir}/eval.log" 2>&1
}
