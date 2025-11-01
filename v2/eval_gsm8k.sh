export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

model_path=Efficient-Large-Model/Fast_dLLM_v2_7B
task=gsm8k

accelerate launch eval.py \
--tasks ${task} \
--batch_size 1 \
--num_fewshot 0 \
--confirm_run_unsafe_code \
--model fast_dllm_v2 \
--fewshot_as_multiturn \
--apply_chat_template \
--model_args model_path=${model_path},threshold=1,show_speed=True
