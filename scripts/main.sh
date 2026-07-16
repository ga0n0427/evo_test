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
START_ITERATION=${START_ITERATION:-1}
QUESTIONER_STEPS=${QUESTIONER_STEPS:-6}
SOLVER_STEPS=${SOLVER_STEPS:-20}
export QUESTIONER_STEPS SOLVER_STEPS

if (( START_ITERATION < 1 || NUM_ITERATIONS < START_ITERATION )); then
    echo "Invalid iteration range: START_ITERATION=${START_ITERATION}, NUM_ITERATIONS=${NUM_ITERATIONS}" >&2
    exit 2
fi

MODEL_ROOT="${STORAGE_PATH}/models"
POINTER_ROOT="${MODEL_ROOT}/${MODEL_ABBR}_evolution"
QUESTIONER_LATEST="${POINTER_ROOT}/questioner/latest"
SOLVER_LATEST="${POINTER_ROOT}/solver/latest"

mkdir -p "${POINTER_ROOT}/questioner" "${POINTER_ROOT}/solver"

validate_adapter() {
    local adapter_path=$1
    if [[ ! -f "${adapter_path}/adapter_config.json" ]]; then
        echo "Missing LoRA config: ${adapter_path}/adapter_config.json" >&2
        return 1
    fi
    if [[ ! -f "${adapter_path}/adapter_model.safetensors" && ! -f "${adapter_path}/adapter_model.bin" ]]; then
        echo "Missing LoRA weights below: ${adapter_path}" >&2
        return 1
    fi
}

validate_merged_model() {
    local merged_path=$1
    if [[ ! -f "${merged_path}/config.json" ]]; then
        echo "Missing merged model config: ${merged_path}/config.json" >&2
        return 1
    fi
    if ! compgen -G "${merged_path}/model*.safetensors" >/dev/null \
        && [[ ! -f "${merged_path}/pytorch_model.bin" ]]; then
        echo "Missing merged model weights below: ${merged_path}" >&2
        return 1
    fi
}

publish_latest() {
    local actor_path=$1
    local latest_link=$2
    local tmp_link="${latest_link}.tmp.$$"

    validate_adapter "${actor_path}/lora_adapter"
    validate_merged_model "${actor_path}/huggingface"

    ln -s "${actor_path}" "${tmp_link}"
    mv -Tf "${tmp_link}" "${latest_link}"
    echo "Published latest actor: ${latest_link} -> ${actor_path}"
}

QUESTIONER_INIT_ADAPTER=""
SOLVER_INIT_ADAPTER=""
SOLVER_EVAL_MERGED="${BASE_MODEL}"

if (( START_ITERATION > 1 )); then
    if [[ ! -L "${QUESTIONER_LATEST}" || ! -L "${SOLVER_LATEST}" ]]; then
        echo "START_ITERATION=${START_ITERATION} requires existing Questioner and Solver latest links." >&2
        exit 1
    fi
    validate_adapter "${QUESTIONER_LATEST}/lora_adapter"
    validate_merged_model "${QUESTIONER_LATEST}/huggingface"
    validate_adapter "${SOLVER_LATEST}/lora_adapter"
    validate_merged_model "${SOLVER_LATEST}/huggingface"

    QUESTIONER_INIT_ADAPTER="${QUESTIONER_LATEST}/lora_adapter"
    SOLVER_INIT_ADAPTER="${SOLVER_LATEST}/lora_adapter"
    SOLVER_EVAL_MERGED="${SOLVER_LATEST}/huggingface"
fi

echo "Model abbreviation: ${MODEL_ABBR}"
echo "Iterations: ${START_ITERATION}..${NUM_ITERATIONS}"

for ((i = START_ITERATION; i <= NUM_ITERATIONS; i++)); do
    QUESTIONER_EXPERIMENT="${MODEL_ABBR}_questioner_v${i}"
    SOLVER_EXPERIMENT="${MODEL_ABBR}_solver_v${i}"
    QUESTIONER_ACTOR="${MODEL_ROOT}/${QUESTIONER_EXPERIMENT}/global_step_${QUESTIONER_STEPS}/actor"
    SOLVER_ACTOR="${MODEL_ROOT}/${SOLVER_EXPERIMENT}/global_step_${SOLVER_STEPS}/actor"

    echo "=== Iteration ${i}: train Questioner ==="
    echo "Questioner train input: Base + ${QUESTIONER_INIT_ADAPTER:-new LoRA}"
    echo "Questioner evaluation opponent: ${SOLVER_EVAL_MERGED}"
    bash scripts/questioner_train_penalty.sh \
        "${BASE_MODEL}" \
        "${QUESTIONER_INIT_ADAPTER}" \
        "${SOLVER_EVAL_MERGED}" \
        "${QUESTIONER_EXPERIMENT}"

    publish_latest "${QUESTIONER_ACTOR}" "${QUESTIONER_LATEST}"
    QUESTIONER_INIT_ADAPTER="${QUESTIONER_LATEST}/lora_adapter"
    QUESTIONER_GENERATION_MERGED="${QUESTIONER_LATEST}/huggingface"

    echo "=== Iteration ${i}: train Solver ==="
    echo "Solver train input: Base + ${SOLVER_INIT_ADAPTER:-new LoRA}"
    echo "Question generation model: ${QUESTIONER_GENERATION_MERGED}"
    echo "Pseudo-label evaluation model: ${SOLVER_EVAL_MERGED}"
    bash scripts/solver_train.sh \
        "${BASE_MODEL}" \
        "${SOLVER_INIT_ADAPTER}" \
        "${QUESTIONER_GENERATION_MERGED}" \
        "${SOLVER_EVAL_MERGED}" \
        "${SOLVER_EXPERIMENT}"

    publish_latest "${SOLVER_ACTOR}" "${SOLVER_LATEST}"
    SOLVER_INIT_ADAPTER="${SOLVER_LATEST}/lora_adapter"
    SOLVER_EVAL_MERGED="${SOLVER_LATEST}/huggingface"
done

echo "Final evaluation model: ${SOLVER_LATEST}/huggingface"
bash evaluation/evaluate.bash "${SOLVER_LATEST}/huggingface"
