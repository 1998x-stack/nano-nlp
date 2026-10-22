"""共享 BM25：支持 RAG 章节复用；保持标准库依赖为零。"""
import math
import re
from collections import Counter

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        if not documents:
            raise ValueError("documents must not be empty")
        self.docs = documents
        self.tokens = [Counter(tokenize(x)) for x in documents]
        self.df = Counter(token for counter in self.tokens for token in counter)
        self.avgdl = sum(map(lambda row: sum(row.values()), self.tokens)) / len(documents)
        self.k1, self.b = k1, b

    def score(self, query, index):
        tf = self.tokens[index]
        length = sum(tf.values())
        avg = self.avgdl if self.avgdl > 0 else 1
        K = self.k1 * (1 - self.b + self.b * length / avg)
        score = 0.0
        for term in set(tokenize(query)):
            count = tf[term]
            if count == 0:
                continue
            idf = math.log(1 + (len(self.docs) - self.df[term] + .5) / (self.df[term] + .5))
            score += idf * (count * (self.k1 + 1)) / (count + K)
        return score

    def search(self, query):
        return sorted(range(len(self.docs)), key=lambda i: (-self.score(query, i), i))
