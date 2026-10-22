"""BM25 小型检索集合；query/relevant 显式分离以计算真正排名指标。"""
DOCS = [
    {"id": "d0", "text": "transformers use self attention for language representation"},
    {"id": "d1", "text": "bm25 ranks documents with term frequency and inverse document frequency"},
    {"id": "d2", "text": "lstm uses gates to remember sequence information"},
    {"id": "d3", "text": "word2vec learns dense word embeddings from contexts"},
    {"id": "d4", "text": "attention relates query and key and value vectors"},
]
VALID = [
    {"query": "bm25 term frequency", "relevant": ["d1"]},
    {"query": "lstm sequence gates", "relevant": ["d2"]},
    {"query": "word2vec word embeddings", "relevant": ["d3"]},
    {"query": "transformer self attention", "relevant": ["d0"]},
]
