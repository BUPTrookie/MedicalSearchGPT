#!/bin/bash
# Merge QLoRA adapter into base model to get a full SFT model
# Required before DPO training (DPO needs a full model, not an adapter)

CUDA_VISIBLE_DEVICES=0 python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_path = 'Qwen/Qwen3.5-0.8B'
adapter_path = 'outputs-sft-qwen3.5-0.8b-qlora'
output_path = 'outputs-sft-qwen3.5-0.8b-merged'

print(f'Loading base model: {base_model_path}')
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map='cpu',
    trust_remote_code=True,
)

print(f'Loading adapter: {adapter_path}')
model = PeftModel.from_pretrained(model, adapter_path)

print('Merging adapter into base model...')
model = model.merge_and_unload()

print(f'Saving merged model to: {output_path}')
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print('Done! Merged model ready for DPO training.')
"
