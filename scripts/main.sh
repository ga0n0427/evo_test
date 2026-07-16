#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/main.sh <base_model> <model_abbr>" >&2
    exit 2
fi

BASE_MODEL=$1
MODEL_ABBR=$2

STORAGE_PATH=${STORAGE_PATH:?Set STORAGE_PATH first}
NUM_ITERATIONS=${NUM_ITERATIONS:-5}
QUESTIONER_STEPS=${QUESTIONER_STEPS:-6}
SOLVER_STEPS=${SOLVER_STEPS:-20}
export QUESTIONER_STEPS SOLVER_STEPS

MODEL_ROOT="${STORAGE_PATH}/models"

# Iteration 1 starts new LoRA adapters from the immutable Base model.
# Later iterations replace these empty values with the previous role's adapter.
QUESTIONER_ADAPTER=""
SOLVER_ADAPTER=""
SOLVER_MERGED="${BASE_MODEL}"

for ((i = 1; i <= NUM_ITERATIONS; i++)); do
    QUESTIONER_EXPERIMENT="${MODEL_ABBR}_questioner_v${i}"
    SOLVER_EXPERIMENT="${MODEL_ABBR}_solver_v${i}"

    echo "=== Iteration ${i}: Questioner ==="
    bash scripts/questioner_train_penalty.sh \
        "${BASE_MODEL}" \
        "${QUESTIONER_ADAPTER}" \
        "${SOLVER_MERGED}" \
        "${QUESTIONER_EXPERIMENT}"

    QUESTIONER_ACTOR="${MODEL_ROOT}/${QUESTIONER_EXPERIMENT}/global_step_${QUESTIONER_STEPS}/actor"
    QUESTIONER_ADAPTER="${QUESTIONER_ACTOR}/lora_adapter"
    QUESTIONER_MERGED="${QUESTIONER_ACTOR}/huggingface"

    echo "=== Iteration ${i}: Solver ==="
    bash scripts/solver_train.sh \
        "${BASE_MODEL}" \
        "${SOLVER_ADAPTER}" \
        "${QUESTIONER_MERGED}" \
        "${SOLVER_MERGED}" \
        "${SOLVER_EXPERIMENT}"

    SOLVER_ACTOR="${MODEL_ROOT}/${SOLVER_EXPERIMENT}/global_step_${SOLVER_STEPS}/actor"
    SOLVER_ADAPTER="${SOLVER_ACTOR}/lora_adapter"
    SOLVER_MERGED="${SOLVER_ACTOR}/huggingface"
done

bash evaluation/evaluate_aggregated.bash "${SOLVER_MERGED}"
