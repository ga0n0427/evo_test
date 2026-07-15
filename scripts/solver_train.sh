#!/usr/bin/env bash
# Usage:
#   bash scripts/solver_train.sh <base_model> <previous_solver_lora_or_empty> \
#       <current_questioner_lora> <solver_experiment_name>

set -euo pipefail

BASE_MODEL=$1
SOLVER_INIT_LORA=${2:-}
QUESTIONER_LORA=$3
EXPERIMENT_NAME=$4

STORAGE_PATH=${STORAGE_PATH:?Set STORAGE_PATH first}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
SOLVER_STEPS=${SOLVER_STEPS:-20}

if [[ -z "${QUESTIONER_LORA}" ]]; then
    echo "Questioner LoRA is required to generate Solver training questions." >&2
    exit 1
fi

export VLLM_DISABLE_COMPILE_CACHE=1

echo "Generate questions with Base + Questioner LoRA"
bash question_generate/question_generate.bash \
    "${BASE_MODEL}" 1000 "${EXPERIMENT_NAME}" "${QUESTIONER_LORA}"

echo "Create pseudo labels with Base + previous Solver LoRA"
bash question_evaluate/evaluate.sh \
    "${BASE_MODEL}" "${EXPERIMENT_NAME}" "${SOLVER_INIT_LORA}"

python question_evaluate/upload.py \
    --repo_name "${EXPERIMENT_NAME}" \
    --max_score 0.8 \
    --min_score 0.3 \
    --experiment_name "${EXPERIMENT_NAME}"

# Train S(t). The LoRA loader uses init_adapter_path only when S(t-1) exists.
TRAIN_CMD=(
    python3 -m verl.trainer.main
    config=examples/config.yaml
    data.max_response_length=4096
    worker.actor.model.model_path="${BASE_MODEL}"
    worker.actor.model.lora.rank="${LORA_RANK}"
    worker.actor.model.lora.alpha="${LORA_ALPHA}"
    worker.actor.model.lora.target_modules=all-linear
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.save_checkpoint_path="${STORAGE_PATH}/models/${EXPERIMENT_NAME}"
    data.train_files="${HUGGINGFACENAME}/${EXPERIMENT_NAME}@train"
    trainer.total_epochs=100
    trainer.max_steps="${SOLVER_STEPS}"
    data.format_prompt=./examples/format_prompt/solver.jinja
    trainer.val_freq=-1
    worker.actor.micro_batch_size_per_device_for_update=1
    worker.actor.micro_batch_size_per_device_for_experience=1
)

if [[ -n "${SOLVER_INIT_LORA}" ]]; then
    TRAIN_CMD+=("worker.actor.model.lora.init_adapter_path=${SOLVER_INIT_LORA}")
fi

"${TRAIN_CMD[@]}"

echo "Solver LoRA: ${STORAGE_PATH}/models/${EXPERIMENT_NAME}/global_step_${SOLVER_STEPS}/actor/lora_adapter"
