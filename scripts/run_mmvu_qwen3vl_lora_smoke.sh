#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CONFIG_PATH=${CONFIG_PATH:-"${PROJECT_DIR}/examples/smoke/mmvu_qwen3vl_lora.yaml"}
MMVU_DIR=${MMVU_DIR:-"/home/ssun/eval/data/MMVU"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-4B-Instruct"}
SMOKE_SAMPLE_COUNT=${SMOKE_SAMPLE_COUNT:-2}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
MOCK_SOLVER_PORT=${MOCK_SOLVER_PORT:-5000}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
WORK_DIR=${WORK_DIR:-"/tmp/evo_test_mmvu_lora_${RUN_ID}"}

if [[ "${CONDA_DEFAULT_ENV:-}" != "easyvideor1" ]]; then
    echo "Activate the easyvideor1 conda environment before running this script." >&2
    exit 2
fi

RAW_DATA="${WORK_DIR}/mmvu_raw.jsonl"
PREPROCESSED_DATA="${WORK_DIR}/mmvu_preprocessed.jsonl"
PREPROCESSED_DIR="${WORK_DIR}/preprocessed_videos"
TASK_DIR="${WORK_DIR}/solver_tasks"
STAGE_A_DIR="${WORK_DIR}/stage_a"
STAGE_B_DIR="${WORK_DIR}/stage_b"
LOG_DIR="${WORK_DIR}/logs"

mkdir -p "$PREPROCESSED_DIR" "$TASK_DIR" "$STAGE_A_DIR" "$STAGE_B_DIR" "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_DISABLE_COMPILE_CACHE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export EVOVID_SOLVER_HOST=127.0.0.1
export EVOVID_SOLVER_PORTS="$MOCK_SOLVER_PORT"
export EVOVID_REWARD_TASK_DIR="$TASK_DIR"

cd "$PROJECT_DIR"

python -m scripts.smoke.prepare_mmvu_smoke \
    --mmvu_dir "$MMVU_DIR" \
    --output "$RAW_DATA" \
    --count "$SMOKE_SAMPLE_COUNT"

python -m scripts.preprocess_videos \
    --input_file "$RAW_DATA" \
    --output_dir "$PREPROCESSED_DIR" \
    --output_file "$PREPROCESSED_DATA" \
    --video_fps 0.5 \
    --video_max_frames 8 \
    --video_min_pixels 50176 \
    --video_max_pixels 50176 \
    --video_total_pixels 401408 \
    --workers 1

python -u -m scripts.smoke.mock_solver_server \
    --host 127.0.0.1 \
    --port "$MOCK_SOLVER_PORT" \
    >"${LOG_DIR}/mock_solver.log" 2>&1 &
MOCK_SOLVER_PID=$!

cleanup() {
    if kill -0 "$MOCK_SOLVER_PID" 2>/dev/null; then
        kill "$MOCK_SOLVER_PID" 2>/dev/null || true
        wait "$MOCK_SOLVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

server_ready=0
for _ in $(seq 1 50); do
    if python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:${MOCK_SOLVER_PORT}/health', timeout=1).read()"; then
        server_ready=1
        break
    fi
    sleep 0.2
done
if [[ "$server_ready" != "1" ]]; then
    echo "Mock Solver did not become ready. See ${LOG_DIR}/mock_solver.log" >&2
    exit 3
fi

COMMON_ARGS=(
    "config=${CONFIG_PATH}"
    "data.train_files=${PREPROCESSED_DATA}"
    "data.val_files=${PREPROCESSED_DATA}"
    "data.preprocessed_video_dir=${PREPROCESSED_DIR}"
    "data.val_preprocessed_video_dir=${PREPROCESSED_DIR}"
    "worker.actor.model.model_path=${MODEL_PATH}"
    "worker.actor.model.lora.rank=${LORA_RANK}"
    "worker.actor.model.lora.alpha=${LORA_ALPHA}"
    "trainer.n_gpus_per_node=1"
    "trainer.max_steps=1"
    "trainer.save_freq=1"
    "trainer.find_last_checkpoint=false"
)

echo "Stage A: initialize and train a new LoRA adapter"
python -m verl.trainer.main \
    "${COMMON_ARGS[@]}" \
    "trainer.experiment_name=mmvu_lora_stage_a" \
    "trainer.save_checkpoint_path=${STAGE_A_DIR}" \
    2>&1 | tee "${LOG_DIR}/stage_a.log"

STAGE_A_ADAPTER="${STAGE_A_DIR}/global_step_1/actor/lora_adapter"
if [[ ! -f "${STAGE_A_ADAPTER}/adapter_config.json" || ! -f "${STAGE_A_ADAPTER}/adapter_model.safetensors" ]]; then
    echo "Stage A did not create a complete LoRA adapter at ${STAGE_A_ADAPTER}" >&2
    exit 4
fi

echo "Stage B: load Stage A adapter as trainable and continue training"
python -m verl.trainer.main \
    "${COMMON_ARGS[@]}" \
    "worker.actor.model.lora.init_adapter_path=${STAGE_A_ADAPTER}" \
    "trainer.experiment_name=mmvu_lora_stage_b" \
    "trainer.save_checkpoint_path=${STAGE_B_DIR}" \
    2>&1 | tee "${LOG_DIR}/stage_b.log"

python -m scripts.smoke.verify_lora_smoke \
    --stage_a "$STAGE_A_DIR" \
    --stage_b "$STAGE_B_DIR" \
    --stage_b_console_log "${LOG_DIR}/stage_b.log" \
    | tee "${LOG_DIR}/verification.json"

echo "Smoke test passed. Artifacts: ${WORK_DIR}"
