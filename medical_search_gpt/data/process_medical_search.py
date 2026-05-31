# -*- coding: utf-8 -*-
"""
Process medical QA datasets into search-augmented RL training format (Parquet).

Supports:
- MedQA (USMLE-style questions)
- PubMedQA (yes/no/maybe QA on PubMed abstracts)
- Custom medical QA JSONL datasets
"""

import json
import os
import argparse
from typing import List, Dict, Optional

import pandas as pd

MEDICAL_SEARCH_PROMPT = """Answer the given medical question. \
You must conduct reasoning inside <think/> </think/> first every time you get new information. \
After reasoning, if you find you lack some medical knowledge, \
you can call a medical search engine by <search> query </search> and it will return the top searched \
medical literature results between <information> and </information>. \
You can search as many times as you want. \
If you find no further external knowledge needed, you can directly provide the answer \
inside <answer> and </answer>, without detailed illustrations. \
For example, <answer> Hypertension </answer>. Question: {question}\n"""


def process_medqa(dataset_path: str, output_dir: str, split: str = "train") -> None:
    """Process MedQA dataset (JSONL with question/options/answer)."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            question = item.get("question", "")

            # Format options into question
            options = item.get("options", {})
            if options:
                option_str = " ".join(f"{k}. {v}" for k, v in options.items())
                question = f"{question}\nOptions: {option_str}"

            answer = item.get("answer", "")

            prompt_text = MEDICAL_SEARCH_PROMPT.format(question=question)
            records.append({
                "data_source": "medical_qa",
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": "medical-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": [answer]},
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "source": "medqa",
                },
            })

    output_path = os.path.join(output_dir, f"{split}.parquet")
    pd.DataFrame(records).to_parquet(output_path)
    print(f"MedQA {split}: {len(records)} records -> {output_path}")


def process_pubmedqa(dataset_path: str, output_dir: str, split: str = "train") -> None:
    """Process PubMedQA dataset (JSONL with question/context/answer)."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            question = item.get("question", "")
            answer = item.get("answer", item.get("final_decision", ""))

            # PubMedQA answers are yes/no/maybe
            if isinstance(answer, dict):
                answer = answer.get("decision", str(answer))

            prompt_text = MEDICAL_SEARCH_PROMPT.format(question=question)
            records.append({
                "data_source": "medical_qa",
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": "medical-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": [answer]},
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "source": "pubmedqa",
                },
            })

    output_path = os.path.join(output_dir, f"{split}.parquet")
    pd.DataFrame(records).to_parquet(output_path)
    print(f"PubMedQA {split}: {len(records)} records -> {output_path}")


def process_custom_jsonl(
    input_path: str, output_dir: str, split: str = "train",
    question_key: str = "question", answer_key: str = "answer",
) -> None:
    """Process a custom medical QA JSONL dataset."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            question = item.get(question_key, "")
            answer = item.get(answer_key, "")
            if not question:
                continue

            prompt_text = MEDICAL_SEARCH_PROMPT.format(question=question)
            records.append({
                "data_source": "medical_qa",
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": "medical-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": [answer] if not isinstance(answer, list) else answer},
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "source": "custom",
                },
            })

    output_path = os.path.join(output_dir, f"{split}.parquet")
    pd.DataFrame(records).to_parquet(output_path)
    print(f"Custom {split}: {len(records)} records -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process medical QA datasets for search-augmented RL training")
    parser.add_argument("--dataset_type", choices=["medqa", "pubmedqa", "custom"], required=True)
    parser.add_argument("--input_path", required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory for parquet files")
    parser.add_argument("--split", default="train")
    parser.add_argument("--question_key", default="question", help="Key for question field (custom mode)")
    parser.add_argument("--answer_key", default="answer", help="Key for answer field (custom mode)")

    args = parser.parse_args()

    if args.dataset_type == "medqa":
        process_medqa(args.input_path, args.output_dir, args.split)
    elif args.dataset_type == "pubmedqa":
        process_pubmedqa(args.input_path, args.output_dir, args.split)
    elif args.dataset_type == "custom":
        process_custom_jsonl(args.input_path, args.output_dir, args.split, args.question_key, args.answer_key)
