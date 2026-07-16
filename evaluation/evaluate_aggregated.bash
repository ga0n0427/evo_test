#!/usr/bin/env bash
# Run the original benchmark scripts with isolated outputs, then merge summaries.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_path>" >&2
  exit 2
fi

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
MODEL_NAME=$1
STORAGE_PATH=${STORAGE_PATH:?Set STORAGE_PATH first}
MODEL_KEY=${MODEL_NAME//\//_}
OUTPUT_DIR="${STORAGE_PATH}/evaluation/${MODEL_KEY}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
PARTS_DIR="${OUTPUT_DIR}/summary_parts/${RUN_ID}"

TASKS=(
  math
  gsm8k
  amc
  minerva
  olympiad
  aime2024
  aime2025
)

mkdir -p \
  "${OUTPUT_DIR}" \
  "${PARTS_DIR}/math" \
  "${PARTS_DIR}/supergpqa" \
  "${PARTS_DIR}/bbeh" \
  "${PARTS_DIR}/mmlupro"

export VLLM_DISABLE_COMPILE_CACHE=1
mapfile -t GPU_QUEUE < <(nvidia-smi --query-gpu=index --format=csv,noheader)
if [[ ${#GPU_QUEUE[@]} -eq 0 ]]; then
  echo "No GPUs are available for evaluation." >&2
  exit 1
fi
echo "Available GPUs: ${GPU_QUEUE[*]}"

declare -A pids
evaluation_failed=0

start_job() {
  local gpu_id=$1
  local task=$2
  echo "==> Start task [${task}] with model [${MODEL_NAME}] on GPU [${gpu_id}]"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python "${PROJECT_DIR}/evaluation/generate.py" \
      --model "${MODEL_NAME}" \
      --dataset "${task}" &
  pids["${gpu_id}"]=$!
}

task_index=0
while :; do
  while [[ ${#GPU_QUEUE[@]} -gt 0 && ${task_index} -lt ${#TASKS[@]} ]]; do
    gpu_id=${GPU_QUEUE[0]}
    GPU_QUEUE=("${GPU_QUEUE[@]:1}")
    task=${TASKS[$task_index]}
    ((task_index += 1))
    start_job "$gpu_id" "$task"
  done

  if [[ ${task_index} -ge ${#TASKS[@]} && ${#pids[@]} -eq 0 ]]; then
    break
  fi

  for gpu_id in "${!pids[@]}"; do
    pid=${pids[$gpu_id]}
    if ! kill -0 "$pid" 2>/dev/null; then
      if ! wait "$pid"; then
        evaluation_failed=1
      fi
      unset 'pids[$gpu_id]'
      GPU_QUEUE+=("$gpu_id")
    fi
  done
  sleep 1
done

if [[ "$evaluation_failed" != "0" ]]; then
  echo "At least one benchmark generation job failed." >&2
  exit 1
fi

(
  cd "${PARTS_DIR}/math"
  PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    python "${PROJECT_DIR}/evaluation/results_recheck.py" --model_name "${MODEL_NAME}"
)

(
  cd "${PARTS_DIR}/supergpqa"
  PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    python "${PROJECT_DIR}/evaluation/eval_supergpqa.py" \
      --model_path "${MODEL_NAME}" \
      --output_file "${OUTPUT_DIR}/supergpqa_answers.json"
)
(
  cd "${PARTS_DIR}/bbeh"
  PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    python "${PROJECT_DIR}/evaluation/eval_bbeh.py" \
      --model_path "${MODEL_NAME}" \
      --output_file "${OUTPUT_DIR}/bbeh_answers.json"
)
(
  cd "${PARTS_DIR}/mmlupro"
  PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    python "${PROJECT_DIR}/evaluation/eval_mmlupro.py" \
      --model_path "${MODEL_NAME}" \
      --output_file "${OUTPUT_DIR}/mmlupro_answers.json"
)

python "${PROJECT_DIR}/evaluation/merge_final_results.py" \
  --input_files \
    "${PARTS_DIR}/math/final_results.jsonl" \
    "${PARTS_DIR}/supergpqa/final_results.jsonl" \
    "${PARTS_DIR}/bbeh/final_results.jsonl" \
    "${PARTS_DIR}/mmlupro/final_results.jsonl" \
  --output_file "${OUTPUT_DIR}/final_results.jsonl"

echo "==> All benchmark summaries: ${OUTPUT_DIR}/final_results.jsonl"
