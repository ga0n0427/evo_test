#!/usr/bin/env bash
# Usage:
#   bash scripts/questioner_train_penalty.sh <base_model> \
#       <previous_questioner_adapter_or_empty> <solver_merged_model> <experiment_name>

set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <base_model> <previous_questioner_adapter_or_empty> <solver_merged_model> <experiment_name>" >&2
    exit 2
fi

BASE_MODEL=$1
QUESTIONER_INIT_ADAPTER=${2:-}
SOLVER_MERGED_MODEL=$3
EXPERIMENT_NAME=$4

STORAGE_PATH=${STORAGE_PATH:?Set STORAGE_PATH first}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
LORA_EXCLUDE_MODULES=${LORA_EXCLUDE_MODULES:-.*visual.*}
QUESTIONER_STEPS=${QUESTIONER_STEPS:-6}
QUESTIONER_TRAIN_GPUS=${QUESTIONER_TRAIN_GPUS:-0,1,2,3}
QUESTIONER_N_GPUS=${QUESTIONER_N_GPUS:-4}

SAVE_ROOT="${STORAGE_PATH}/models/${EXPERIMENT_NAME}"
ACTOR_PATH="${SAVE_ROOT}/global_step_${QUESTIONER_STEPS}/actor"
RUN_ID=$(date +%s%N)
SERVER_PID=""

cleanup_solver_service() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup_solver_service EXIT

echo "Start merged Solver service: ${SOLVER_MERGED_MODEL}"
bash vllm_service_init/start.sh "${SOLVER_MERGED_MODEL}" "${RUN_ID}" &
SERVER_PID=$!

echo "Train Questioner from Base + adapter: ${QUESTIONER_INIT_ADAPTER:-<new LoRA>}"
TRAIN_CMD=(
    python3 -m verl.trainer.main
    config=examples/config.yaml
    data.max_response_length=4096
    worker.actor.model.model_path="${BASE_MODEL}"
    worker.actor.model.freeze_vision_tower=false
    worker.actor.model.lora.rank="${LORA_RANK}"
    worker.actor.model.lora.alpha="${LORA_ALPHA}"
    worker.actor.model.lora.target_modules="${LORA_TARGET_MODULES}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.save_checkpoint_path="${SAVE_ROOT}"
    trainer.total_epochs=1000
    trainer.max_steps="${QUESTIONER_STEPS}"
    trainer.save_freq="${QUESTIONER_STEPS}"
    trainer.save_limit=1
    trainer.save_model_only=true
    trainer.find_last_checkpoint=false
    worker.reward.reward_function=./examples/reward_function/evo_vid_questioner_reward.py:compute_score
    trainer.val_freq=-1
    trainer.n_gpus_per_node="${QUESTIONER_N_GPUS}"
    data.format_prompt=./examples/format_prompt/questioner.jinja
    worker.rollout.n=4
    worker.actor.global_batch_size=16
)

if [[ -n "${QUESTIONER_INIT_ADAPTER}" ]]; then
    TRAIN_CMD+=("worker.actor.model.lora.init_adapter_path=${QUESTIONER_INIT_ADAPTER}")
fi
if [[ -n "${LORA_EXCLUDE_MODULES}" ]]; then
    TRAIN_CMD+=("worker.actor.model.lora.exclude_modules=${LORA_EXCLUDE_MODULES}")
fi

CUDA_VISIBLE_DEVICES="${QUESTIONER_TRAIN_GPUS}" "${TRAIN_CMD[@]}"

if [[ ! -f "${ACTOR_PATH}/lora_adapter/adapter_config.json" ]]; then
    echo "Questioner training did not produce a LoRA adapter: ${ACTOR_PATH}/lora_adapter" >&2
    exit 1
fi

echo "Merge Questioner LoRA for evaluation only"
python3 -m scripts.merge_lora_adapter \
    --base_model "${BASE_MODEL}" \
    --adapter_path "${ACTOR_PATH}/lora_adapter" \
    --output_dir "${ACTOR_PATH}/huggingface"

echo "Questioner adapter: ${ACTOR_PATH}/lora_adapter"
echo "Questioner merged model: ${ACTOR_PATH}/huggingface"
