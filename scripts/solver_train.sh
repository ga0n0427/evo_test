#!/usr/bin/env bash
# Usage:
#   bash scripts/solver_train.sh <base_model> <previous_solver_adapter_or_empty> \
#       <questioner_merged_model> <previous_solver_merged_model> <experiment_name>

set -euo pipefail

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 <base_model> <previous_solver_adapter_or_empty> <questioner_merged_model> <previous_solver_merged_model> <experiment_name>" >&2
    exit 2
fi

BASE_MODEL=$1
SOLVER_INIT_ADAPTER=${2:-}
QUESTIONER_MERGED_MODEL=$3
SOLVER_EVAL_MERGED_MODEL=$4
EXPERIMENT_NAME=$5

STORAGE_PATH=${STORAGE_PATH:?Set STORAGE_PATH first}
HUGGINGFACENAME=${HUGGINGFACENAME:?Set HUGGINGFACENAME first}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
LORA_EXCLUDE_MODULES=${LORA_EXCLUDE_MODULES:-.*visual.*}
SOLVER_STEPS=${SOLVER_STEPS:-20}

SAVE_ROOT="${STORAGE_PATH}/models/${EXPERIMENT_NAME}"
ACTOR_PATH="${SAVE_ROOT}/global_step_${SOLVER_STEPS}/actor"

export VLLM_DISABLE_COMPILE_CACHE=1

echo "Generate questions with merged Questioner: ${QUESTIONER_MERGED_MODEL}"
bash question_generate/question_generate.bash \
    "${QUESTIONER_MERGED_MODEL}" 1000 "${EXPERIMENT_NAME}"

echo "Create pseudo labels with merged previous Solver: ${SOLVER_EVAL_MERGED_MODEL}"
bash question_evaluate/evaluate.sh \
    "${SOLVER_EVAL_MERGED_MODEL}" "${EXPERIMENT_NAME}"

python3 question_evaluate/upload.py \
    --repo_name "${EXPERIMENT_NAME}" \
    --max_score 0.8 \
    --min_score 0.3 \
    --experiment_name "${EXPERIMENT_NAME}"

echo "Train Solver from Base + adapter: ${SOLVER_INIT_ADAPTER:-<new LoRA>}"
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
    data.train_files="${HUGGINGFACENAME}/${EXPERIMENT_NAME}@train"
    trainer.total_epochs=100
    trainer.max_steps="${SOLVER_STEPS}"
    trainer.save_freq="${SOLVER_STEPS}"
    trainer.save_limit=1
    trainer.save_model_only=true
    trainer.find_last_checkpoint=false
    data.format_prompt=./examples/format_prompt/solver.jinja
    trainer.val_freq=-1
    worker.actor.micro_batch_size_per_device_for_update=1
    worker.actor.micro_batch_size_per_device_for_experience=1
)

if [[ -n "${SOLVER_INIT_ADAPTER}" ]]; then
    TRAIN_CMD+=("worker.actor.model.lora.init_adapter_path=${SOLVER_INIT_ADAPTER}")
fi
if [[ -n "${LORA_EXCLUDE_MODULES}" ]]; then
    TRAIN_CMD+=("worker.actor.model.lora.exclude_modules=${LORA_EXCLUDE_MODULES}")
fi

"${TRAIN_CMD[@]}"

if [[ ! -f "${ACTOR_PATH}/lora_adapter/adapter_config.json" ]]; then
    echo "Solver training did not produce a LoRA adapter: ${ACTOR_PATH}/lora_adapter" >&2
    exit 1
fi

echo "Merge Solver LoRA for evaluation only"
python3 -m scripts.merge_lora_adapter \
    --base_model "${BASE_MODEL}" \
    --adapter_path "${ACTOR_PATH}/lora_adapter" \
    --output_dir "${ACTOR_PATH}/huggingface"

echo "Solver adapter: ${ACTOR_PATH}/lora_adapter"
echo "Solver merged model: ${ACTOR_PATH}/huggingface"
