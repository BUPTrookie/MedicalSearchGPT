# -*- coding: utf-8 -*-
"""
Search-augmented inference for medical QA.
Models can reason through medical questions and call a medical search engine
when they need additional knowledge.

Usage:
    # Start search server first:
    python -m medical_search_gpt.search.retrieval_server \
        --index_path indexes/pubmed_bm25 --corpus_path data/pubmed.jsonl --topk 5

    # Run inference:
    CUDA_VISIBLE_DEVICES=0 python -m medical_search_gpt.serving.search_inference \
        --model_path ./outputs-search-rl \
        --search_url http://127.0.0.1:8000/retrieve \
        --interactive
"""

import argparse
import re

import requests
import torch
import transformers


MEDICAL_SEARCH_PROMPT = """Answer the given medical question. \
You must conduct reasoning inside <think/> </think/> first every time you get new information. \
After reasoning, if you find you lack some medical knowledge, \
you can call a medical search engine by <search> query </search> and it will return the top searched \
medical literature results between <information> and </information>. \
You can search as many times as you want. \
If you find no further external knowledge needed, you can directly provide the answer \
inside <answer> and </answer>, without detailed illustrations. \
For example, <answer> Hypertension </answer>. Question: {question}\n"""


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [
            tokenizer.encode(t, add_special_tokens=False) for t in target_sequences
        ]
        self.target_lengths = [len(t) for t in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for i, target in enumerate(self.target_ids):
            target_tensor = torch.as_tensor(target, device=input_ids.device)
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target_tensor):
                return True
        return False


def get_search_query(text: str) -> str | None:
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def search(query: str, search_url: str, topk: int = 5) -> str:
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    results = requests.post(search_url, json=payload).json()["result"]

    format_ref = ""
    for idx, doc_item in enumerate(results[0]):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_ref += f"[{idx + 1}] {title}\n{text}\n"
    return format_ref


def run_inference(
    question: str,
    model,
    tokenizer,
    search_url: str,
    topk: int = 5,
    max_turns: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
):
    device = next(model.parameters()).device

    question = question.strip()
    if question[-1] not in "。？！?":
        question += "？"

    prompt_text = MEDICAL_SEARCH_PROMPT.format(question=question)

    if tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )

    # EOS tokens (Qwen2.5 series)
    curr_eos = [151645, 151643]
    search_template = "\n\n{output_text}<information>{search_results}</information>\n\n"

    target_sequences = [
        "</search>", " </search>", "</search>\n",
        " </search>\n", "</search>\n\n", " </search>\n\n",
    ]
    stopping_criteria = transformers.StoppingCriteriaList(
        [StopOnSequence(target_sequences, tokenizer)]
    )

    prompt = prompt_text
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    print(prompt)

    for turn in range(max_turns):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
        )

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Check if generation ended naturally (EOS)
        if outputs[0][-1].item() in curr_eos:
            print(output_text)
            break

        # Extract search query and perform search
        tmp_query = get_search_query(
            tokenizer.decode(outputs[0], skip_special_tokens=True)
        )
        if tmp_query:
            search_results = search(tmp_query, search_url, topk)
        else:
            search_results = ""

        search_text = search_template.format(
            output_text=output_text, search_results=search_results
        )
        prompt += search_text
        print(search_text)

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Medical search-augmented inference")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--search_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--question", type=str, default=None)

    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    if args.interactive:
        print("Medical Search-Augmented Inference (type 'quit' to exit)")
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if question:
                run_inference(
                    question, model, tokenizer,
                    args.search_url, args.topk,
                    args.max_turns, args.temperature, args.max_new_tokens,
                )
    elif args.question:
        run_inference(
            args.question, model, tokenizer,
            args.search_url, args.topk,
            args.max_turns, args.temperature, args.max_new_tokens,
        )
    else:
        print("Please provide --question or use --interactive mode")


if __name__ == "__main__":
    main()
