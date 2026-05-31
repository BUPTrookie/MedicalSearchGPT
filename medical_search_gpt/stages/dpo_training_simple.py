# -*- coding: utf-8 -*-
"""
Standalone DPO Training — no TRL dependency.

DPO loss:
    L = -E[ log σ( β * (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)) ) ]

Where:
    y_w = chosen (better response)
    y_l = rejected (worse response)
    π_θ = policy model (training)
    π_ref = reference model (frozen)
    β = temperature (default 0.1)
    σ = sigmoid
"""
import os
from dataclasses import dataclass, field
from glob import glob

import torch
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"


@dataclass
class DPOArgs:
    # Model
    model_name_or_path: str = field(default=None, metadata={"help": "Path to the SFT model (merged)"})
    # Data
    train_file_dir: str = field(default=None)
    validation_file_dir: str = field(default=None)
    max_source_length: int = field(default=1024)
    max_target_length: int = field(default=512)
    max_train_samples: int = field(default=None)
    # LoRA
    use_peft: bool = field(default=True)
    lora_rank: int = field(default=8)
    lora_alpha: float = field(default=16.0)
    lora_dropout: float = field(default=0.05)
    qlora: bool = field(default=False)
    # Training
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=5e-5)
    max_steps: int = field(default=50)
    warmup_steps: int = field(default=10)
    output_dir: str = field(default="outputs-dpo")
    # DPO
    beta: float = field(default=0.1, metadata={"help": "DPO temperature. Lower = stronger preference signal"})
    # Misc
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    save_steps: int = field(default=50)
    eval_steps: int = field(default=25)
    logging_steps: int = field(default=1)
    seed: int = field(default=42)


def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


def load_dpo_data(args, tokenizer):
    """Load and tokenize preference data."""
    max_len = args.max_source_length + args.max_target_length

    # Load jsonl
    data_files = glob(f"{args.train_file_dir}/**/*.jsonl", recursive=True)
    data_files += glob(f"{args.train_file_dir}/**/*.json", recursive=True)
    logger.info(f"Found {len(data_files)} data files: {data_files}")
    dataset = load_dataset("json", data_files=data_files, split="train")

    if args.max_train_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_train_samples)))

    def tokenize_example(example):
        prompt = example["question"]
        if example.get("system"):
            prompt = example["system"] + "\n" + prompt

        chosen = prompt + example["response_chosen"]
        rejected = prompt + example["response_rejected"]

        chosen_tokens = tokenizer(
            chosen, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt"
        )
        rejected_tokens = tokenizer(
            rejected, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt"
        )
        prompt_len = len(tokenizer(prompt, truncation=True, padding=False)["input_ids"])

        # Response mask: 1 for response tokens, 0 for prompt + padding
        chosen_response_mask = torch.zeros(max_len, dtype=torch.long)
        chosen_response_mask[prompt_len:] = chosen_tokens["attention_mask"][0, prompt_len:]

        rejected_response_mask = torch.zeros(max_len, dtype=torch.long)
        rejected_response_mask[prompt_len:] = rejected_tokens["attention_mask"][0, prompt_len:]

        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "chosen_response_mask": chosen_response_mask,
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
            "rejected_response_mask": rejected_response_mask,
        }

    dataset = dataset.map(tokenize_example, remove_columns=dataset.column_names)
    dataset.set_format(type="torch")
    return dataset


def get_log_probs(model, input_ids, attention_mask, response_mask):
    """
    Compute log π(y|x) for each token in the response part.
    response_mask: 1 for response tokens, 0 for prompt and padding.
    Returns sum of log probs for the response tokens only.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Shift: logits[i] predicts token[i+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Shift response_mask to align with shift_logits
    shift_response_mask = response_mask[:, 1:]

    # Log softmax for numerical stability
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log prob of the actual next token
    per_token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Only count response tokens
    per_token_log_probs = per_token_log_probs * shift_response_mask

    # Sum over response tokens → log π(y|x)
    return per_token_log_probs.sum(dim=-1)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    """
    DPO loss:
    L = -E[ log σ( β * (log π_θ(y_w) - log π_ref(y_w) - log π_θ(y_l) + log π_ref(y_l)) ) ]
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    # Metrics
    with torch.no_grad():
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
        chosen_reward_mean = chosen_rewards.mean()
        rejected_reward_mean = rejected_rewards.mean()

    return loss, {
        "loss": loss.item(),
        "chosen_reward": chosen_reward_mean.item(),
        "rejected_reward": rejected_reward_mean.item(),
        "reward_accuracy": reward_accuracy.item(),
        "reward_margin": (chosen_reward_mean - rejected_reward_mean).item(),
    }


def main():
    # Parse args from command line
    import sys
    args = DPOArgs()

    # Simple arg parsing
    # Use default values to infer types (Optional[int] defaults to None → treat as int)
    INT_FIELDS = {
        "max_source_length", "max_target_length", "max_train_samples", "lora_rank",
        "per_device_train_batch_size", "gradient_accumulation_steps", "max_steps",
        "warmup_steps", "save_steps", "eval_steps", "logging_steps", "seed",
    }
    FLOAT_FIELDS = {"lora_alpha", "lora_dropout", "learning_rate", "beta"}
    BOOL_FIELDS = {"use_peft", "qlora", "bf16", "gradient_checkpointing"}

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        key = argv[i].lstrip("-").replace("-", "_")
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            val = argv[i + 1]
            if key in INT_FIELDS:
                setattr(args, key, int(val))
            elif key in FLOAT_FIELDS:
                setattr(args, key, float(val))
            elif key in BOOL_FIELDS:
                setattr(args, key, val.lower() in ("true", "1", "yes"))
            elif hasattr(args, key):
                setattr(args, key, val)
            i += 2
        else:
            i += 1

    logger.info(f"Args: {args}")
    torch.manual_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load data
    dataset = load_dpo_data(args, tokenizer)
    logger.info(f"Dataset size: {len(dataset)}")

    # Split train/eval (90/10)
    split = int(len(dataset) * 0.9)
    train_dataset = dataset.select(range(split))
    eval_dataset = dataset.select(range(split, len(dataset)))
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)

    # Load policy model
    logger.info(f"Loading model from {args.model_name_or_path}")
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    # Apply LoRA
    if args.use_peft:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Load reference model (frozen copy)
    logger.info("Loading reference model (frozen)")
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # Training loop
    device = next(model.parameters()).device
    model.train()

    logger.info(f"Starting DPO training for {args.max_steps} steps, beta={args.beta}")
    logger.info(f"Device: {device}")

    data_iter = iter(train_loader)
    for step in range(1, args.max_steps + 1):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward: compute log probs for chosen and rejected under both models
        with torch.no_grad():
            ref_chosen_logps = get_log_probs(
                ref_model, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_response_mask"]
            )
            ref_rejected_logps = get_log_probs(
                ref_model, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_response_mask"]
            )

        policy_chosen_logps = get_log_probs(
            model, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_response_mask"]
        )
        policy_rejected_logps = get_log_probs(
            model, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_response_mask"]
        )

        # DPO loss
        loss, metrics = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            args.beta,
        )
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        # Gradient accumulation
        if step % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if step % args.logging_steps == 0:
            logger.info(
                f"Step {step}/{args.max_steps} | "
                f"loss={metrics['loss']:.4f} | "
                f"chosen_r={metrics['chosen_reward']:.4f} | "
                f"rejected_r={metrics['rejected_reward']:.4f} | "
                f"margin={metrics['reward_margin']:.4f} | "
                f"accuracy={metrics['reward_accuracy']:.2%}"
            )

        if step % args.save_steps == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    # Final save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
