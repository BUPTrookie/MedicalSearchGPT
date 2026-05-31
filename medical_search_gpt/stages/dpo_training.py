# -*- coding: utf-8 -*-
"""
DPO (Direct Preference Optimization) 训练脚本

@author: XuMing(xuming624@qq.com)
@description: 使用 DPO 算法对 SFT 模型进行偏好对齐训练

====================================================================
DPO 算法原理
====================================================================

1. 为什么需要 DPO？
   SFT（监督微调）后的模型能生成合理的回答，但无法区分回答的"好坏"。
   例如对医学问题"感冒了怎么办"，模型可能生成专业建议（好）或民间偏方（差）。
   DPO 的目标：让模型学会偏好"好的回答"，回避"差的回答"。

2. RLHF 的传统路线 vs DPO 的简化路线

   传统 RLHF（PPO 路线）：
     SFT 模型 → 训练 Reward Model → 用 PPO 优化策略模型
     需要：策略模型 + 参考模型 + 奖励模型 + 价值模型（4个模型！）
     问题：训练不稳定，超参数敏感，工程复杂

   DPO（直接偏好优化）：
     SFT 模型 → 直接用偏好数据优化（1个训练模型 + 1个参考模型）
     关键洞察：把 RLHF 的约束优化问题转化为分类问题
     优势：不需要 Reward Model，训练简单稳定

3. DPO 的数学核心

   偏好数据格式: (prompt, chosen_response, rejected_response)

   DPO Loss:
     L = -log σ( β * ( log π_θ(y_w|x) / π_ref(y_w|x)
                      - log π_θ(y_l|x) / π_ref(y_l|x) ) )

   其中:
   - π_θ:   当前正在训练的策略模型（policy model）
   - π_ref: 冻结的参考模型（通常是 SFT 后的模型，训练中不更新）
   - y_w:   chosen（偏好的/好的）回答
   - y_l:   rejected（不偏好的/差的）回答
   - σ:     sigmoid 函数
   - β:     温度超参数，控制偏好强度（默认 0.1）
   - log π(y|x) / π_ref(y|x): 当前模型相对于参考模型的 log 概率比
     这个比值衡量了"当前模型比参考模型更倾向于生成这个回答的程度"

   直觉理解:
   - 如果 π_θ 相对于 π_ref 更偏好 chosen > rejected → loss 趋近 0（好）
   - 如果 π_θ 相对于 π_ref 更偏好 rejected > chosen → loss 变大（惩罚）
   - 训练目标：让 π_θ 相对于 π_ref 的概率比在 chosen 上更高，在 rejected 上更低

4. 本脚本的数据流

   JSONL 数据文件:
     {"system": "...", "history": [...], "question": "...",
      "response_chosen": "好的回答", "response_rejected": "差的回答"}
           ↓
   数据预处理 (return_prompt_and_responses):
     将 system + history + question 组装为 prompt 模板
           ↓
   DPOTrainer 内部处理:
     1. 对 (prompt + chosen) 和 (prompt + rejected) 分别做 tokenize
     2. 用当前模型 π_θ 计算两者的 log probability
     3. 用参考模型 π_ref 计算两者的 log probability
     4. 计算 DPO loss = -log σ(β * (log_ratio_chosen - log_ratio_rejected))
     5. 反向传播更新 π_θ 的参数
           ↓
   输出: 对齐后的模型（偏好好的回答，回避差的回答）

5. LoRA / QLoRA 的作用

   DPO 训练通常使用 LoRA（低秩适配），原因:
   - 全参数训练显存需求巨大（7B 模型全参数 DPO 需要 80GB+ 显存）
   - LoRA 只训练低秩矩阵，显存需求降低到单卡可运行
   - QLoRA 进一步量化基底模型到 4-bit，显存更低
   - ref_model 在 LoRA 模式下自动处理（不需要显式创建，因为基底权重冻结）

====================================================================
"""

import os
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, Optional

import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import DPOTrainer, DPOConfig

from medical_search_gpt.template import get_conv_template

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"  # 禁用 tokenizer 多进程（避免死锁）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"     # 允许重复加载 OpenMP（Windows 兼容）


# ===========================================================================
# 训练参数配置
# ===========================================================================

@dataclass
class ScriptArguments:
    """DPO 训练的全部可配置参数

    分为五组:
    1. 模型参数: 用哪个模型、怎么加载
    2. 数据参数: 用什么数据、怎么处理
    3. 训练参数: 学习率、batch size、优化器等
    4. PEFT 参数: LoRA 配置
    5. 输出参数: 保存路径、日志等
    """

    # ---- 1. 模型参数 ----
    # 必须提供，通常是 SFT 训练后的模型路径
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "模型路径，通常是 SFT 后的模型"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "分词器路径，默认与模型相同"}
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "8-bit 量化加载"})
    load_in_4bit: bool = field(default=False, metadata={"help": "4-bit 量化加载"})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "模型缓存目录"}
    )
    use_fast_tokenizer: bool = field(
        default=False, metadata={"help": "是否使用 fast tokenizer"}
    )
    # torch_dtype 决定模型权重的数值精度:
    # - bfloat16: 推荐（A100/4090+），数值范围大，不易溢出
    # - float16: 旧卡（V100 等），可能溢出
    # - float32: 全精度，显存翻倍，一般不需要
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "模型权重精度: auto/bfloat16/float16/float32",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "设备映射，auto 表示自动分配 GPU/CPU"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程代码（Qwen 等模型需要）"},
    )

    # ---- 2. 数据参数 ----
    # 支持两种数据源:
    # (a) HuggingFace Hub 数据集: 指定 dataset_name
    # (b) 本地 JSONL 文件: 指定 train_file_dir
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "HuggingFace 数据集名称"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "数据集子配置名"}
    )
    train_file_dir: Optional[str] = field(
        default=None, metadata={"help": "本地训练数据目录（JSONL 文件）"}
    )
    validation_file_dir: Optional[str] = field(
        default=None, metadata={"help": "本地验证数据目录"}
    )
    template_name: Optional[str] = field(
        default="vicuna", metadata={"help": "对话模板名称（vicuna/qwen/chatml 等）"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=4, metadata={"help": "每张卡的训练 batch size"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "每张卡的验证 batch size"}
    )
    max_source_length: Optional[int] = field(
        default=2048, metadata={"help": "prompt 最大 token 数"}
    )
    max_target_length: Optional[int] = field(
        default=512, metadata={"help": "response 最大 token 数"}
    )
    min_target_length: Optional[int] = field(
        default=4, metadata={"help": "response 最小 token 数（过滤过短回答）"}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "截断训练样本数（调试用）"}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "截断验证样本数（调试用）"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "是否覆盖预处理缓存"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={"help": "无验证集时，从训练集切出百分之几做验证"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4, metadata={"help": "数据预处理的工作进程数"},
    )

    # ---- 3. 训练参数 ----
    use_peft: bool = field(default=True, metadata={"help": "是否使用 LoRA/PEFT"})
    qlora: bool = field(default=False, metadata={"help": "是否使用 QLoRA（4-bit 量化 + LoRA）"})
    target_modules: Optional[str] = field(
        default=None, metadata={"help": "LoRA 目标模块，逗号分隔，或 'all'"}
    )
    lora_rank: Optional[int] = field(
        default=8, metadata={"help": "LoRA 秩 r，越大参数越多但表达力越强"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "LoRA dropout，防止过拟合"}
    )
    lora_alpha: Optional[float] = field(
        default=16.0, metadata={"help": "LoRA 缩放因子 α，实际缩放 = α/r"}
    )
    peft_path: Optional[str] = field(
        default=None, metadata={"help": "已有的 LoRA 权重路径（增量训练）"}
    )
    do_train: bool = field(default=False, metadata={"help": "是否执行训练"})
    do_eval: bool = field(default=False, metadata={"help": "是否执行验证"})
    learning_rate: Optional[float] = field(
        default=5e-4, metadata={"help": "学习率（LoRA 常用 1e-4 ~ 5e-4）"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "学习率调度器: cosine/linear/constant 等"}
    )
    warmup_steps: Optional[int] = field(
        default=100, metadata={"help": "预热步数（从 0 线性增到目标 lr）"}
    )
    weight_decay: Optional[float] = field(
        default=0.05, metadata={"help": "权重衰减（L2 正则化）"}
    )
    optim: Optional[str] = field(
        default="adamw_hf", metadata={"help": "优化器（adamw 对 LoRA 最常用）"}
    )
    fp16: Optional[bool] = field(default=True, metadata={"help": "FP16 混合精度"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "BF16 混合精度（A100 推荐）"})
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "梯度检查点（用计算换显存，DPO 必须开，因为要同时过 chosen 和 rejected）"},
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4,
        metadata={"help": "梯度累积步数（等效 batch_size = per_device_bs × accum_steps × num_gpus）"},
    )
    save_steps: Optional[int] = field(default=50, metadata={"help": "每 N 步保存检查点"})
    eval_steps: Optional[int] = field(default=50, metadata={"help": "每 N 步验证一次"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "每 N 步打印日志"})
    output_dir: Optional[str] = field(default="outputs-dpo", metadata={"help": "输出目录"})
    max_steps: Optional[int] = field(default=200, metadata={"help": "最大训练步数"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "验证策略: steps/epoch/no"})
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "DPO 需要保留所有列（prompt/chosen/rejected），必须设 False"},
    )
    report_to: Optional[str] = field(
        default="tensorboard", metadata={"help": "日志后端: tensorboard/wandb"}
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("必须指定 model_name_or_path")


def print_trainable_parameters(model):
    """打印模型的可训练参数统计

    LoRA 训练时，只有低秩矩阵的参数是可训练的（通常 < 1% 的总参数量）。
    例: Qwen2.5-0.5B 全参数 5 亿，LoRA r=8 只训练约 200 万参数。
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """自动发现模型中所有线性层名称，用于 LoRA 的 target_modules

    为什么不手动指定？不同模型的线性层名称不同:
    - LLaMA: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    - Qwen: 同上但可能多一些层
    自动发现可以兼容所有模型架构。

    排除 lm_head / output_layer: 最后的分类层通常不加 LoRA（会影响输出分布）
    """
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit      # QLoRA 4-bit 量化线性层
        elif int8:
            cls = bnb.nn.Linear8bitLt    # 8-bit 量化线性层
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # 排除输出层（加 LoRA 会破坏已学到的输出分布）
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            # 取最后一级名称（如 "q_proj" 而非完整路径）
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def main():
    # ==================================================================
    # Step 1: 解析命令行参数
    # ==================================================================
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    logger.info(f"Parse args: {args}")

    # ==================================================================
    # Step 2: 加载分词器
    # ==================================================================
    # 分词器必须与模型匹配，否则 token id 对不上
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "trust_remote_code": args.trust_remote_code,
    }
    tokenizer_name_or_path = args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = args.model_name_or_path  # 默认与模型同路径
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    # 加载对话模板（将 system + history + question 格式化为模型期望的输入格式）
    prompt_template = get_conv_template(args.template_name)

    # 确保 tokenizer 具备必要的特殊 token
    # eos_token: 生成停止标记，DPO 计算log概率时需要知道序列在哪里结束
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = prompt_template.stop_str
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    # bos_token: 序列起始标记
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    # pad_token: 批量填充标记（DPO 中 chosen/rejected 长度不同时需要 padding）
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    logger.debug(f"Tokenizer: {tokenizer}")

    # ==================================================================
    # Step 3: 加载数据集
    # ==================================================================
    # DPO 需要偏好数据: 每条数据包含 prompt + chosen + rejected
    # 支持两种来源:
    #   (a) HuggingFace Hub（指定 dataset_name）
    #   (b) 本地 JSONL 文件（指定 train_file_dir）
    if args.dataset_name is not None:
        # 从 HuggingFace Hub 加载
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            # 没有验证集时，从训练集切出 1% 做验证
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )
    else:
        # 从本地目录加载 JSON/JSONL 文件
        data_files = {}
        if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
            train_data_files = glob(f'{args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {', '.join(train_data_files)}")
            data_files["train"] = train_data_files
        if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
            eval_data_files = glob(f'{args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {', '.join(eval_data_files)}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )
    logger.info(f"Raw datasets: {raw_datasets}")

    # ==================================================================
    # Step 4: 数据预处理 — 将原始数据转为 DPO 需要的格式
    # ==================================================================
    # DPO 需要三个字段: prompt, chosen, rejected
    # 原始数据格式: system, history, question, response_chosen, response_rejected
    # 这一步把 system + history + question 组装成完整的 prompt
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length
    full_max_length = max_source_length + max_target_length

    def return_prompt_and_responses(examples) -> Dict[str, str]:
        """将原始数据转换为 DPO 格式: {prompt, chosen, rejected}

        数据流:
          原始: {"system": "你是医生", "history": [[q1,a1]], "question": "感冒怎么办",
                 "response_chosen": "多休息多喝水", "response_rejected": "喝板蓝根"}
          输出: {"prompt": "<system>你是医生</s><user>q1</s><assistant>a1</s><user>感冒怎么办</s>",
                 "chosen": "多休息多喝水",
                 "rejected": "喝板蓝根"}

        prompt 拼接规则:
          system_prompt + history[[q,a], [q,a]...] + question
          空回答的 question 放在最后，等待模型生成
        """
        prompts = []
        for system, history, question in zip(examples["system"], examples["history"], examples["question"]):
            system_prompt = system or ""
            # 把 question 追加到 history 末尾（answer 为空，等待模型生成）
            history_with_question = history + [[question, '']] if history else [[question, '']]
            prompts.append(prompt_template.get_prompt(messages=history_with_question, system_prompt=system_prompt))
        return {
            "prompt": prompts,
            "chosen": examples["response_chosen"],
            "rejected": examples["response_rejected"],
        }

    # 处理训练集
    train_dataset = None
    max_train_samples = 0
    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train']
        max_train_samples = len(train_dataset)
        if args.max_train_samples is not None and args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        tokenized_dataset = train_dataset.shuffle().map(
            return_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        # 过滤: 去掉 prompt+response 总长度超过上限的样本
        # DPO 中 chosen 和 rejected 的总长度都不能超过 full_max_length
        # 否则截断会导致 loss 计算不准确（丢失了部分 response 的 log prob）
        train_dataset = tokenized_dataset.filter(
            lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                      and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
        )
        logger.debug(f"Num train_samples: {len(train_dataset)}")
        logger.debug("First train example:")
        first_example = train_dataset[0]
        logger.debug(f"prompt:\n{first_example['prompt']}")
        logger.debug(f"chosen:\n{first_example['chosen']}")
        logger.debug(f"rejected:\n{first_example['rejected']}")

    # 处理验证集（逻辑与训练集相同）
    eval_dataset = None
    max_eval_samples = 0
    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
        eval_dataset = eval_dataset.map(
            return_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        eval_dataset = eval_dataset.filter(
            lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                      and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
        )
        logger.debug(f"Num eval_samples: {len(eval_dataset)}")
        logger.debug("First eval example:")
        first_example = eval_dataset[0]
        logger.debug(f"prompt:\n{first_example['prompt']}")
        logger.debug(f"chosen:\n{first_example['chosen']}")
        logger.debug(f"rejected:\n{first_example['rejected']}")

    # ==================================================================
    # Step 5: 加载模型
    # ==================================================================
    # 模型来源: SFT 后的 checkpoint（已有基本的对话能力）
    # DPO 在此基础上微调，使其学会偏好好的回答
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    # 多卡分布式训练时，不用 device_map（由 DeepSpeed/DDP 管理设备分配）
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    if ddp:
        args.device_map = None
    if args.device_map in ["None", "none", ""]:
        args.device_map = None
    logger.info(f"Device map: {args.device_map}")
    if args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir
    )
    if args.load_in_4bit or args.load_in_8bit:
        logger.info(f"Quantizing model, load_in_4bit: {args.load_in_4bit}, load_in_8bit: {args.load_in_8bit}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        # QLoRA: 将模型权重量化到 4-bit 或 8-bit，大幅降低显存
        # - load_in_4bit: NF4 量化（推荐），7B 模型从 14GB 降到 ~4GB
        # - load_in_8bit: 8-bit 量化，7B 模型从 14GB 降到 ~7GB
        # - bnb_4bit_use_double_quant: 对量化常数再做一次量化，再省 ~0.4bit/param
        # - bnb_4bit_compute_dtype: 计算时用什么精度（推荐与 torch_dtype 一致）
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        ) if args.qlora else None,
    )
    # QLoRA 模式下，可训练参数需要 float32（量化基底 + float32 LoRA）
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)

    # ==================================================================
    # Step 6: 配置 DPO 训练器
    # ==================================================================
    # 梯度检查点: DPO 对每个样本需要同时前向 (prompt+chosen) 和 (prompt+rejected)，
    # 以及参考模型的同等计算，显存消耗是普通训练的 2~3 倍。
    # 梯度检查点通过"用计算换显存"来缓解: 不保存中间激活，反向时重新计算。
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # 梯度检查点与 KV Cache 不兼容
    else:
        model.config.use_cache = True

    # DPOConfig: TRL 库的 DPO 训练配置
    # 关键参数说明:
    # - max_prompt_length: prompt 的最大长度（DPO 只对 response 部分计算 loss）
    # - max_length: prompt + response 的总长度上限
    # - remove_unused_columns: 必须为 False，DPO 需要 prompt/chosen/rejected 三个字段
    training_args = DPOConfig(
        max_prompt_length=args.max_source_length,
        max_length=full_max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=args.remove_unused_columns,
        run_name=f"dpo_v1",
    )

    # 配置 LoRA（如果启用）
    # LoRA (Low-Rank Adaptation): 冻结原始权重 W，只训练低秩矩阵 A 和 B
    #   W' = W + α/r * B @ A
    # 其中 A: (d, r), B: (r, d), r << d（如 r=8, d=3584）
    # 可训练参数量: 2 × r × d × num_layers（只有全参数的 ~0.5%）
    peft_config = None
    if args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")
        target_modules = args.target_modules.split(',') if args.target_modules else None
        if target_modules and 'all' in target_modules:
            target_modules = find_all_linear_names(model, int4=args.load_in_4bit, int8=args.load_in_8bit)
        logger.info(f"Peft target_modules: {target_modules}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,     # 因果语言模型任务
            target_modules=target_modules,     # 要加 LoRA 的线性层名称列表
            inference_mode=False,              # 训练模式
            r=args.lora_rank,                  # LoRA 秩（核心超参数，控制参数量和表达力）
            lora_alpha=args.lora_alpha,        # 缩放因子（通常设为 2×r），越大 LoRA 影响越强
            lora_dropout=args.lora_dropout,    # Dropout 防过拟合
        )
    else:
        logger.info("Fine-tuning method: Full parameters training")

    # ==================================================================
    # Step 7: 创建 DPOTrainer
    # ==================================================================
    # DPOTrainer 的核心工作:
    # 1. 持有 policy model (model) 和 reference model (ref_model)
    # 2. 对每个 batch:
    #    a. tokenize (prompt + chosen) 和 (prompt + rejected)
    #    b. policy model 前向 → 得到 π_θ(chosen|x) 和 π_θ(rejected|x) 的 log prob
    #    c. reference model 前向 → 得到 π_ref(chosen|x) 和 π_ref(rejected|x) 的 log prob
    #    d. 计算 DPO loss = -log σ(β * (log_ratio_w - log_ratio_l))
    #    e. 反向传播更新 policy model 的参数
    #
    # ref_model 的处理:
    # - LoRA 模式 (ref_model=None): DPOTrainer 自动用 model 的基底权重作为 reference
    #   因为 LoRA 冻结了原始权重，所以 reference = 去掉 LoRA 后的模型
    # - 全参数模式: 需要显式传入 deepcopy(model) 作为 reference
    #   因为全参数训练会修改所有权重，所以必须保留一份原始副本
    trainer = DPOTrainer(
        model,
        ref_model=None if args.use_peft else deepcopy(model),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config if args.use_peft else None,
    )
    print_trainable_parameters(trainer.model)

    # ==================================================================
    # Step 8: 训练
    # ==================================================================
    if args.do_train:
        if trainer.is_world_process_zero():
            logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {args.output_dir}")
            trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            trainer.model.save_pretrained(args.output_dir)

    # ==================================================================
    # Step 9: 验证
    # ==================================================================
    if args.do_eval:
        if trainer.is_world_process_zero():
            logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = max_eval_samples
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
