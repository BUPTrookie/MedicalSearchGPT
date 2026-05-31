# -*- coding: utf-8 -*-
"""
Medical QA reward scoring functions for search-augmented RL training.
Provides exact match (with medical normalization) and token-level F1 scoring.
"""

import re
import string
import random
from typing import Dict, List


# Common medical abbreviations and their expansions
MEDICAL_ABBREVIATIONS = {
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "chf": "congestive heart failure",
    "copd": "chronic obstructive pulmonary disease",
    "mi": "myocardial infarction",
    "ckd": "chronic kidney disease",
    "cad": "coronary artery disease",
    "tia": "transient ischemic attack",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "uri": "upper respiratory infection",
    "uti": "urinary tract infection",
    "sob": "shortness of breath",
    "ecg": "electrocardiogram",
    "ekg": "electrocardiogram",
    "mri": "magnetic resonance imaging",
    "ct": "computed tomography",
    "nsaid": "nonsteroidal anti-inflammatory drug",
    "ace": "angiotensin converting enzyme",
    "arb": "angiotensin receptor blocker",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "prn": "as needed",
    "po": "by mouth",
    "iv": "intravenous",
    "im": "intramuscular",
}


def normalize_medical_answer(s: str) -> str:
    """Normalize a medical answer for comparison."""
    s = s.lower().strip()

    # Expand common abbreviations
    words = s.split()
    expanded = []
    for word in words:
        expanded.append(MEDICAL_ABBREVIATIONS.get(word, word))
    s = " ".join(expanded)

    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Fix whitespace
    s = " ".join(s.split())

    return s


def extract_solution(solution_str: str):
    """Extract the answer from the solution string (last <answer>...</answer> tag)."""
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if len(matches) <= 1:
        return None

    return matches[-1].group(1).strip()


def medical_em_check(prediction: str, golden_answers: List[str]) -> float:
    """Check exact match with medical normalization."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_pred = normalize_medical_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_medical_answer(golden_answer) == normalized_pred:
            return 1.0
    return 0.0


def medical_f1_score(prediction: str, golden_answer: str) -> float:
    """Compute token-level F1 score between prediction and golden answer."""
    pred_tokens = set(normalize_medical_answer(prediction).split())
    gold_tokens = set(normalize_medical_answer(golden_answer).split())

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = pred_tokens & gold_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_score_medical_em(
    solution_str: str, ground_truth: Dict, method: str = "strict",
    format_score: float = 0.0, score: float = 1.0
) -> float:
    """Exact match scoring for medical QA with answer normalization."""
    answer = extract_solution(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"[Medical EM] Golden answers: {ground_truth['target']}")
        print(f"[Medical EM] Extracted answer: {answer}")
        print(f"[Medical EM] Solution string: {solution_str}")

    if answer is None:
        return 0.0

    if medical_em_check(answer, ground_truth["target"]):
        return score
    return format_score


def compute_score_medical_f1(
    solution_str: str, ground_truth: Dict, method: str = "strict",
    format_score: float = 0.0, score: float = 1.0
) -> float:
    """F1-based scoring for medical QA. More forgiving than exact match."""
    answer = extract_solution(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"[Medical F1] Golden answers: {ground_truth['target']}")
        print(f"[Medical F1] Extracted answer: {answer}")
        print(f"[Medical F1] Solution string: {solution_str}")

    if answer is None:
        return 0.0

    targets = ground_truth["target"]
    if isinstance(targets, str):
        targets = [targets]

    # Take the best F1 score across all golden answers
    best_f1 = max(medical_f1_score(answer, t) for t in targets)

    # Scale: if best_f1 > threshold, give full score; otherwise give format_score
    if best_f1 >= 0.5:
        return score * best_f1
    return format_score
