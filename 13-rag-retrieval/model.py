"""最小 RAG scaffold：可运行的召回、证据、prompt；不伪装成 LLM。"""
import argparse
import sys
from pathlib import Path

# 允许从任意工作目录运行 python 13-rag-retrieval/demo.py。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.retrieval import BM25

CORPUS = [
    {"id": "D1", "text": "Transformer uses self-attention to process token relations."},
    {"id": "D2", "text": "BERT uses bidirectional masked language modeling."},
    {"id": "D3", "text": "GPT uses a causal language modeling objective."},
]

def retrieve(query, k=2):
    bm25 = BM25([doc["text"] for doc in CORPUS])
    return [(CORPUS[i], bm25.score(query, i)) for i in bm25.search(query)[:k]]

def build_prompt(query, hits):
    evidence = "\n".join(
        f'[{doc["id"]}] {doc["text"]}' for doc, _ in hits
    )
    return ("请仅使用下列证据回答。若证据不足请说明，并标明 [D#] 引用。\n"
            f"证据:\n{evidence}\n问题: {query}\n回答:")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="What does Transformer use?")
    args = parser.parse_args()
    hits = retrieve(args.query)
    print("RESULT:\n" + build_prompt(args.query, hits))
