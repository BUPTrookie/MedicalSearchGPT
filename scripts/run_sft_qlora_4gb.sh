#!/bin/bash
# SFT training script for 4GB GPU (RTX 3050/3060)
# Model: Qwen2.5-0.5B-Instruct with QLoRA (4-bit quantization)
# Run from MedicalSearchGPT root directory

CUDA_VISIBLE_DEVICES=0 python medical_search_gpt/stages/supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --template_name qwen \
    --train_file_dir ./data/finetune \
    --validation_file_dir ./data/finetune \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --use_peft True \
    --qlora True \
    --load_in_4bit True \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --warmup_steps 10 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_steps 200 \
    --save_strategy steps \
    --save_total_limit 3 \
    --max_source_length 512 \
    --model_max_length 512 \
    --output_dir outputs-sft-qwen-0.5b-qlora \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --preprocessing_num_workers 2 \
    --seed 42

echo "SFT training complete!"
