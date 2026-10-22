"""最小 bigram 语言模型：计数、平滑、概率与预测一目了然。"""
from collections import Counter, defaultdict

CORPUS = [
    "the cat likes fish",
    "the dog likes fish",
    "the cat likes milk",
    "the dog eats fish",
]

class BigramLM:
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.vocab = {"<eos>"}

    def fit(self, sentences):
        for sentence in sentences:
            words = sentence.lower().split() + ["<eos>"]
            self.vocab.update(words)
            for prev, word in zip(["<bos>"] + words[:-1], words):
                self.counts[prev][word] += 1
        return self

    def probability(self, previous, next_word):
        """Laplace 平滑：未观察到的搭配也不为 0。"""
        row = self.counts[previous]
        return (row[next_word] + 1) / (sum(row.values()) + len(self.vocab))

    def predict(self, previous):
        return sorted(
            ((w, self.probability(previous, w)) for w in self.vocab),
            key=lambda pair: (-pair[1], pair[0]),
        )

if __name__ == "__main__":
    lm = BigramLM().fit(CORPUS)
    print("RESULT:", lm.predict("likes")[:3])
