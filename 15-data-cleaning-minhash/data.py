"""真实管线要求持久 doc_id 和来源元信息，不只是一堆字符串。"""
RAW=[
    {"id":"a","source":"site-a","quality":0.97,"text":"NLP models learn from a large text corpus."},
    {"id":"b","source":"mirror","quality":0.65,"text":"NLP models learn from a large text corpus."},
    {"id":"c","source":"site-b","quality":0.90,"text":"NLP models learn from a large text corpora."},
    {"id":"d","source":"journal","quality":0.95,"text":"Attention connects different tokens in a sequence."},
    {"id":"e","source":"spam","quality":0.12,"text":"Free free free free buy now."},
    {"id":"f","source":"tutorial","quality":0.87,"text":"Transformers model language using attention across tokens."},
]
