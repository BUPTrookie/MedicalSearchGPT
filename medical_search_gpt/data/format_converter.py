# -*- coding: utf-8 -*-
"""
Bidirectional converter between MedicalGPT JSONL format and Search-R1 Parquet format.

MedicalGPT GRPO format (JSONL):
    {"question": "...", "answer": "..."}

Search-R1/veRL format (Parquet):
    data_source, prompt, ability, reward_model, extra_info
"""

import json
import os
import argparse
from typing import List, Dict, Optional

import pandas as pd

# Medical search prompt template
MEDICAL_SEARCH_PROMPT = """Answer the given medical question. \
You must conduct reasoning inside <think/> </think/> first every time you get new information. \
After reasoning, if you find you lack some medical knowledge, \
you can call a medical search engine by <search> query </search> and it will return the top searched \
medical literature results between <information> and </information>. \
You can search as many times as you want. \
If you find no further external knowledge needed, you can directly provide the answer \
inside <answer> and </answer>, without detailed illustrations. \
For example, <answer> Hypertension </answer>. Question: {question}\n"""


def jsonl_to_parquet(
    input_dir: str,
    output_dir: str,
    data_source: str = "medical_qa",
    template_type: str = "base",
    ability: str = "medical-reasoning",
    train_file: str = None,
    val_file: str = None,
) -> None:
    """Convert MedicalGPT JSONL data to veRL Parquet format.

    Args:
        input_dir: Directory containing JSONL files (question/answer format)
        output_dir: Directory to write parquet files
        data_source: Data source identifier for reward function selection
        template_type: Prompt template type ('base' for generic)
        ability: Ability tag for the dataset
        train_file: Specific training file name (default: auto-detect first .jsonl)
        val_file: Specific validation file name (default: use same as train)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find JSONL files
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {input_dir}")

    if train_file is None:
        train_file = jsonl_files[0]

    # Process training data
    train_records = _process_jsonl(
        os.path.join(input_dir, train_file),
        data_source=data_source,
        ability=ability,
        split="train",
    )

    train_df = pd.DataFrame(train_records)
    train_path = os.path.join(output_dir, "train.parquet")
    train_df.to_parquet(train_path)
    print(f"Training data: {len(train_records)} records -> {train_path}")

    # Process validation data (use same file if val_file not specified)
    val_input = os.path.join(input_dir, val_file) if val_file else os.path.join(input_dir, train_file)
    val_records = _process_jsonl(
        val_input,
        data_source=data_source,
        ability=ability,
        split="test",
        max_samples=100,
    )

    val_df = pd.DataFrame(val_records)
    val_path = os.path.join(output_dir, "test.parquet")
    val_df.to_parquet(val_path)
    print(f"Validation data: {len(val_records)} records -> {val_path}")


def _process_jsonl(
    filepath: str,
    data_source: str,
    ability: str,
    split: str,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Process a single JSONL file into veRL format records."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            # Handle different JSONL formats
            question = item.get("question", item.get("input", item.get("prompt", "")))
            answer = item.get("answer", item.get("output", item.get("response", "")))

            if not question:
                continue

            # Ensure question ends with appropriate punctuation
            question = question.strip()
            if question[-1] not in "。？！?":
                question += "？"

            prompt_text = MEDICAL_SEARCH_PROMPT.format(question=question)

            # Handle multiple golden answers
            if isinstance(answer, list):
                target = answer
            else:
                target = [answer]

            record = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": ability,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": target},
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            records.append(record)

    return records


def parquet_to_jsonl(parquet_path: str, output_path: str) -> None:
    """Convert veRL Parquet format back to MedicalGPT JSONL format."""
    df = pd.read_parquet(parquet_path)
    records = []

    for _, row in df.iterrows():
        # Extract question from prompt
        prompt = row.get("prompt", [])
        question = ""
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg["content"]
                # Extract question from template
                match = content.rfind("Question: ")
                if match >= 0:
                    question = content[match + len("Question: "):].strip()
                else:
                    question = content

        # Extract answer from ground truth
        ground_truth = row.get("reward_model", {}).get("ground_truth", {})
        target = ground_truth.get("target", "")
        answer = target[0] if isinstance(target, list) and target else target

        if question:
            records.append({"question": question, "answer": answer})

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(records)} records -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert between MedicalGPT JSONL and veRL Parquet formats")
    parser.add_argument("--mode", choices=["jsonl2parquet", "parquet2jsonl"], required=True)
    parser.add_argument("--input_dir", help="Input directory (for jsonl2parquet)")
    parser.add_argument("--input_file", help="Input file (for parquet2jsonl)")
    parser.add_argument("--output_dir", help="Output directory (for jsonl2parquet)")
    parser.add_argument("--output_file", help="Output file (for parquet2jsonl)")
    parser.add_argument("--data_source", default="medical_qa")
    parser.add_argument("--ability", default="medical-reasoning")

    args = parser.parse_args()

    if args.mode == "jsonl2parquet":
        jsonl_to_parquet(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            data_source=args.data_source,
            ability=args.ability,
        )
    else:
        parquet_to_jsonl(
            parquet_path=args.input_file,
            output_path=args.output_file,
        )
