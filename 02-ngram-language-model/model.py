"""带显式未知词状态的 Bigram LM。

关键概率语义：
* 输出空间 V 包含 <eos>、<unk> 与所有训练词。
* 任意未见的下一个词先映射到 <unk>，而不是为其凭空添加一个新词。
* P(w|h) 在固定 V 上加一平滑，所以对所有 w∈V 求和严格为 1。
* 未知历史词同样映射到 <unk>；查询期间不修改训练计数。
"""
from collections import Counter, defaultdict

BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"

CORPUS = [
    "the cat likes fish",
    "the dog likes fish",
    "the cat likes milk",
    "the dog eats fish",
]

class BigramLM:
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.vocab = {EOS, UNK}

    def fit(self, sentences):
        """重置并从训练语料重建频数表，不读取任何 held-out 词。"""
        self.counts = defaultdict(Counter)
        self.vocab = {EOS, UNK}
        rows = [sentence.lower().split() for sentence in sentences]
        self.vocab.update(token for row in rows for token in row)
        for words in rows:
            target = words + [EOS]
            for previous, word in zip([BOS] + target[:-1], target):
                self.counts[previous][word] += 1
        return self

    def normalize_word(self, word):
        """BOS 是条件上下文，不是预测输出；未知词统一到 <unk>。"""
        word = word.lower()
        if word == BOS:
            return BOS
        return word if word in self.vocab else UNK

    def probability(self, previous, next_word):
        """P(next|previous)，加一平滑定义在固定输出词表上。"""
        prev = self.normalize_word(previous)
        next_token = self.normalize_word(next_word)
        if next_token == BOS:
            raise ValueError("BOS 不应是预测目标")
        row = self.counts.get(prev, Counter())
        return (row.get(next_token, 0) + 1) / (sum(row.values()) + len(self.vocab))

    def predict(self, previous):
        return sorted(
            ((w, self.probability(previous, w)) for w in self.vocab),
            key=lambda pair: (-pair[1], pair[0]),
        )

if __name__ == "__main__":
    lm = BigramLM().fit(CORPUS)
    print("RESULT:", lm.predict("likes")[:3])
