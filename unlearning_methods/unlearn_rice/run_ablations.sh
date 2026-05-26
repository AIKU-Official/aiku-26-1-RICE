#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

echo "Running RICE ablations from ${REPO_ROOT}"

bash unlearning_methods/unlearn_rice/run_ablation_no_retain.sh
bash unlearning_methods/unlearn_rice/run_ablation_ga_forget.sh
bash unlearning_methods/unlearn_rice/run_ablation_all_layers.sh

echo "All RICE ablations finished."
