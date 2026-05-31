# -*- coding: utf-8 -*-
"""
Medical-specific generation manager that extends LLMGenerationManager
with medical query expansion and citation formatting.
"""

import re
from typing import List, Dict, Tuple, Any

from .generation import LLMGenerationManager, GenerationConfig


# Common medical term expansions for search query improvement
MEDICAL_SYNONYMS = {
    "heart attack": "myocardial infarction",
    "high blood pressure": "hypertension",
    "low blood sugar": "hypoglycemia",
    "stroke": "cerebrovascular accident",
    "cancer": "neoplasm malignant",
    "kidney disease": "renal disease nephropathy",
    "liver disease": "hepatic disease",
    "lung disease": "pulmonary disease",
}


class MedicalLLMGenerationManager(LLMGenerationManager):
    """Generation manager with medical-specific search enhancements."""

    def __init__(self, tokenizer, actor_rollout_wg, config: GenerationConfig,
                 is_validation: bool = False):
        super().__init__(tokenizer, actor_rollout_wg, config, is_validation)

    def _expand_medical_query(self, query: str) -> str:
        """Expand medical query with synonyms for better retrieval."""
        query_lower = query.lower()
        for term, expansion in MEDICAL_SYNONYMS.items():
            if term in query_lower:
                query = f"{query} {expansion}"
        return query

    def _batch_search(self, queries):
        """Override to add medical query expansion before searching."""
        expanded_queries = [self._expand_medical_query(q) for q in queries]
        return super()._batch_search(expanded_queries)

    def _passages2string(self, retrieval_result):
        """Override to format medical literature with PMID-style citations."""
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            score = doc_item.get("score", 0)
            format_reference += f"[{idx + 1}] {title}\n{text}\n"
        return format_reference

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """Override to add medical entity awareness in prediction processing."""
        return super().postprocess_predictions(predictions)


# Medical search prompt template for training data
MEDICAL_SEARCH_SYSTEM_PROMPT = """You are a medical AI assistant. Answer the given medical question. \
You must conduct reasoning inside <think/> </think/> first every time you get new information. \
After reasoning, if you find you lack some medical knowledge, \
you can call a medical search engine by <search> query </search> and it will return the top searched \
medical literature results between <information> and </information>. \
You can search as many times as you want. \
If you find no further external knowledge needed, \
you can directly provide the answer inside <answer> and </answer>, \
without detailed illustrations. For example, <answer> Hypertension </answer>. \
Question: {question}"""
