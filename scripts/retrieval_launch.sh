#!/bin/bash
# Launch medical search retrieval server
# Usage: bash scripts/retrieval_launch.sh

file_path=${CORPUS_PATH:-"./indexes/pubmed"}
index_file=$file_path/bm25_index
corpus_file=$file_path/pubmed_abstracts.jsonl
retriever_name=${RETRIEVER_NAME:-"bm25"}
retriever_path=${RETRIEVER_MODEL:-""}
topk=${RETRIEVER_TOPK:-5}

python -m medical_search_gpt.search.retrieval_server --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk $topk \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path
