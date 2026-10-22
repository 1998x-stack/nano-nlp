"""BM25 from scratch：不使用 sklearn 或搜索引擎。"""
import math
import re
from collections import Counter

DOCS = [
    "transformer uses attention to model language",
    "bm25 retrieves documents using term frequency",
    "attention helps a language model focus on tokens",
]

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())

class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.terms = [Counter(tokenize(d)) for d in documents]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(lambda x: sum(x.values()), self.terms)) / len(documents)
        self.df = Counter(w for tf in self.terms for w in tf)

    def score(self, query, index):
        tf = self.terms[index]
        dl = sum(tf.values())
        K = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        result = 0.0
        for word in set(tokenize(query)):
            freq = tf[word]
            if not freq:
                continue
            # 用平滑 IDF 避免稀有词产生负权重。
            idf = math.log(1 + (len(self.terms) - self.df[word] + 0.5) / (self.df[word] + 0.5))
            result += idf * freq * (self.k1 + 1) / (freq + K)
        return result

    def search(self, query):
        return sorted(range(len(self.documents)), key=lambda i: self.score(query, i), reverse=True)

if __name__ == "__main__":
    engine = BM25(DOCS)
    print("RESULT:", [(i, round(engine.score("attention language", i), 3))
                      for i in engine.search("attention language")])
