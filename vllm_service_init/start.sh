#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <merged_model_path> <run_id>" >&2
    exit 2
fi

MODEL_PATH=$1
RUN_ID=$2
export VLLM_DISABLE_COMPILE_CACHE=1

pids=()

cleanup() {
    local pid
    for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

echo "Starting merged Solver services for RUN_ID=${RUN_ID}: ${MODEL_PATH}"
CUDA_VISIBLE_DEVICES=4 python3 vllm_service_init/start_vllm_server.py --port 5000 --model_path "${MODEL_PATH}" &
pids+=("$!")
CUDA_VISIBLE_DEVICES=5 python3 vllm_service_init/start_vllm_server.py --port 5001 --model_path "${MODEL_PATH}" &
pids+=("$!")
CUDA_VISIBLE_DEVICES=6 python3 vllm_service_init/start_vllm_server.py --port 5002 --model_path "${MODEL_PATH}" &
pids+=("$!")
CUDA_VISIBLE_DEVICES=7 python3 vllm_service_init/start_vllm_server.py --port 5003 --model_path "${MODEL_PATH}" &
pids+=("$!")

wait
