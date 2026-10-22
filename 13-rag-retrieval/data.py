"""检索知识库独立于测试问题，带明确出处与 ground-truth 证据。"""
DOCS=[
    {"id":"D1","text":"Transformer uses self-attention to connect input tokens."},
    {"id":"D2","text":"BERT trains with masked language modeling over bidirectional context."},
    {"id":"D3","text":"GPT learns to predict the next token using a causal mask."},
    {"id":"D4","text":"BM25 ranks documents by term frequency, IDF, and length normalization."},
]
VALID=[
    {"question":"What does Transformer use to connect tokens?","evidence":["D1"]},
    {"question":"Which model uses masked language modeling?","evidence":["D2"]},
    {"question":"What does GPT predict?","evidence":["D3"]},
    {"question":"How does BM25 rank documents?","evidence":["D4"]},
    {"question":"What is the weather tomorrow?","evidence":[]},
]
