# -*- coding: utf-8 -*-
"""
Medical corpora connectors for building search indexes.
Supports PubMed abstracts, medical knowledge bases, and custom medical corpora.
"""

import json
import os
import re
import time
from typing import List, Dict, Optional
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None


class PubMedCorpusBuilder:
    """Fetch articles from PubMed E-utilities API and convert to JSONL format."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        self.api_key = api_key
        self.email = email

    def search_articles(self, query: str, max_results: int = 10000, retstart: int = 0) -> List[str]:
        """Search PubMed and return a list of PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 10000),
            "retstart": retstart,
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email

        resp = requests.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def fetch_articles(self, pmids: List[str], batch_size: int = 100) -> List[Dict]:
        """Fetch article details for a list of PMIDs."""
        articles = []
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            resp = requests.get(f"{self.BASE_URL}/efetch.fcgi", params=params)
            resp.raise_for_status()

            articles.extend(self._parse_xml_articles(resp.text))
            if i + batch_size < len(pmids):
                time.sleep(0.34)  # Respect rate limits (3 req/s without API key)

        return articles

    def _parse_xml_articles(self, xml_text: str) -> List[Dict]:
        """Parse PubMed XML response into article dicts."""
        articles = []
        for article_block in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml_text, re.DOTALL):
            title_match = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", article_block, re.DOTALL)
            abstract_match = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", article_block, re.DOTALL)
            pmid_match = re.search(r"<PMID[^>]*>(\d+)</PMID>", article_block)

            title = title_match.group(1).strip() if title_match else ""
            # Clean XML tags from abstract
            abstract = " ".join(
                re.sub(r"<[^>]+>", "", a).strip() for a in abstract_match
            )
            pmid = pmid_match.group(1) if pmid_match else ""

            if title or abstract:
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "contents": f"{title}\n{abstract}",
                })
        return articles

    def build_corpus(
        self, query: str, max_results: int = 10000, output_path: str = "pubmed_corpus.jsonl"
    ) -> str:
        """Build a JSONL corpus file from PubMed search results."""
        pmids = self.search_articles(query, max_results)
        print(f"Found {len(pmids)} articles for query: {query}")

        articles = self.fetch_articles(pmids)
        print(f"Fetched {len(articles)} articles")

        with open(output_path, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps({"contents": article["contents"]}, ensure_ascii=False) + "\n")

        print(f"Corpus saved to {output_path}")
        return output_path


class MedicalCorpusPreprocessor:
    """Preprocess medical text for search index building."""

    MEDICAL_ABBREVIATIONS = {
        "HTN": "hypertension",
        "DM": "diabetes mellitus",
        "CHF": "congestive heart failure",
        "COPD": "chronic obstructive pulmonary disease",
        "MI": "myocardial infarction",
        "CKD": "chronic kidney disease",
        "CAD": "coronary artery disease",
        "TIA": "transient ischemic attack",
        "DVT": "deep vein thrombosis",
        "PE": "pulmonary embolism",
        "URI": "upper respiratory infection",
        "UTI": "urinary tract infection",
        "SOB": "shortness of breath",
        "N/V": "nausea and vomiting",
        "CBC": "complete blood count",
        "BMP": "basic metabolic panel",
        "ECG": "electrocardiogram",
        "MRI": "magnetic resonance imaging",
        "CT": "computed tomography",
    }

    def normalize_text(self, text: str) -> str:
        """Normalize medical text: lowercase, expand abbreviations."""
        text = text.lower().strip()
        for abbr, full in self.MEDICAL_ABBREVIATIONS.items():
            text = re.sub(rf"\b{abbr.lower()}\b", full, text)
        return text

    def process_jsonl(self, input_path: str, output_path: str, normalize: bool = True):
        """Process a JSONL corpus file with optional text normalization."""
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                item = json.loads(line.strip())
                if normalize and "contents" in item:
                    item["contents"] = self.normalize_text(item["contents"])
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_custom_corpus(
    input_path: str, output_path: str, title_field: str = "title", text_field: str = "text"
):
    """Convert a custom medical corpus to the JSONL format expected by index_builder.py.

    Input can be JSONL with arbitrary fields. Output is JSONL with 'contents' field.
    """
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line.strip())
            title = item.get(title_field, "")
            text = item.get(text_field, item.get("abstract", item.get("content", "")))
            contents = f"{title}\n{text}" if title else text
            fout.write(json.dumps({"contents": contents}, ensure_ascii=False) + "\n")
