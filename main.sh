Base_model=$1
Model_abbr=$2
echo "Model_abbr: $Model_abbr"

# Iteration 1: train Questioner from base, evaluated by base Solver.
bash scripts/questioner_train_penalty.sh \
    "$Base_model" \
    "$Base_model" \
    "${Model_abbr}_questioner_v1"

questioner_lora_path="${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/lora_adapter"

# Then generate Solver data with Questioner LoRA v1, evaluate with base Solver,
# and train Solver from base.
bash scripts/solver_train.sh \
    "$Base_model" \
    "$Base_model" \
    "${Model_abbr}_solver_v1" \
    "" \
    "$questioner_lora_path" \
    "$Base_model"

# From iteration 2 onward:
# - train Questioner from previous Questioner checkpoint
# - evaluate Questioner/Solver data with previous Solver LoRA
# - generate Solver data with the newly trained Questioner LoRA
# - train Solver from previous Solver checkpoint
for i in {2..5}; do
    prev=$((i - 1))
    solver_lora_path="${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/lora_adapter"
    questioner_model_path="${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}/global_step_5/actor/huggingface"
    solver_model_path="${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface"

    bash scripts/questioner_train_penalty.sh \
        "$Base_model" \
        "$questioner_model_path" \
        "${Model_abbr}_questioner_v${i}" \
        "$solver_lora_path"

    questioner_lora_path="${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_5/actor/lora_adapter"

    bash scripts/solver_train.sh \
        "$solver_model_path" \
        "$Base_model" \
        "${Model_abbr}_solver_v${i}" \
        "$solver_lora_path" \
        "$questioner_lora_path" \
        "$Base_model"
done

bash evaluation/evaluate.bash "$Base_model"
