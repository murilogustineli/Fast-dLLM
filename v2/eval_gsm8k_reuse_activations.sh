#!/bin/bash
set -e

# Example: 
# bash eval_gsm8k_reuse_activations.sh --limit 10
# If no --limit is given, it runs the full evaluation.

# parse optional --limit argument
LIMIT_ARG=""
NUM_SAMPLES="all"
if [ "$1" == "--limit" ] && [ -n "$2" ]; then
    LIMIT_ARG="--limit $2"
    NUM_SAMPLES=$2
    echo "[INFO] Limiting evaluation to $NUM_SAMPLES samples"
else
    echo "[INFO] Running full evaluation (no limit specified)"
fi

# environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# model and experiment info
MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_7B"
TASK="gsm8k"

# create results directory if not exists
mkdir -p results

# sweep over k = 1, 2, 3
for K in 1 2 3; do
    TAG="reuse_k${K}_limit${NUM_SAMPLES}"

    # pass variables to eval.py for logging
    export TASK_NAME=$TASK
    export RUN_TAG=$TAG

    echo ""
    echo "================================================================"
    echo "[INFO] Starting evaluation: $TASK  |  reuse_k=${K}  |  ${NUM_SAMPLES} samples"
    echo "================================================================"

    accelerate launch eval.py \
        --tasks ${TASK} \
        --batch_size 1 \
        --num_fewshot 0 \
        ${LIMIT_ARG} \
        --confirm_run_unsafe_code \
        --model fast_dllm_v2 \
        --fewshot_as_multiturn \
        --apply_chat_template \
        --model_args model_path=${MODEL_PATH},threshold=1,show_speed=True,use_block_cache=True,reuse_k=${K} \
        --output_path results/${TASK}_${TAG}_raw/

    echo "[INFO] Evaluation complete for reuse_k=${K}"
done

echo ""
echo "[INFO] All k-tests completed. Results stored under ./results/"
