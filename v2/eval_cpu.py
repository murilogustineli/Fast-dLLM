import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import types
import generation_functions  # this is v2/generation_functions.py

model_path = "Efficient-Large-Model/Fast_dLLM_v2_7B"

print("Loading model... this may take a few minutes on CPU.")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float32,  # use float32 since bfloat16 unsupported on Mac CPU
    device_map={"": "cpu"},
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# attach the Fast-dLLM sampling method manually (same as in eval.py)
model.mdm_sample = types.MethodType(
    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model
)

print("Model and tokenizer loaded successfully!")
