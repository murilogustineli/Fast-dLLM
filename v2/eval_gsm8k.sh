#!/bin/bash
set -e

# Usage:
# bash eval_gsm8k.sh             → full evaluation
# bash eval_gsm8k.sh --limit 10  → test mode (10 samples)

# parse optional --limit argument
LIMIT_ARG=""
if [ "$1" == "--limit" ] && [ -n "$2" ]; then
    LIMIT_ARG="--limit $2"
    echo "[INFO] Limiting evaluation to $2 samples"
fi

# parse optional --batch_size argument
BATCH_SIZE=1
if [ "$3" == "--batch_size" ] && [ -n "$4" ]; then
    BATCH_SIZE=$4
    echo "[INFO] Setting batch size to $BATCH_SIZE"
fi

# environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# model and experiment info
MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_7B"
TASK="gsm8k"
TAG="threshold1_run"

# datetime for saving results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# pass variables to eval.py for logging
export TASK_NAME=$TASK
export RUN_TAG=$TAG
export RUN_TIMESTAMP=$TIMESTAMP

# create results directory if not exists
mkdir -p results/${TIMESTAMP}

# run evaluation
echo "[INFO] Starting evaluation: $TASK (${MODEL_PATH})"
accelerate launch eval.py \
--tasks ${TASK} \
--batch_size 1 \
--num_fewshot 0 \
${LIMIT_ARG} \
--confirm_run_unsafe_code \
--model fast_dllm_v2 \
--fewshot_as_multiturn \
--apply_chat_template \
--model_args model_path=${MODEL_PATH},threshold=1,show_speed=True \
--output_path results/${TIMESTAMP}/${TASK}_${TAG}_raw/

echo "[INFO] Evaluation complete."