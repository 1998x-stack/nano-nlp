"""极简 Skip-gram negative sampling。pip install torch。"""
import argparse
import torch
from torch import nn
from torch.nn import functional as F

TEXT = "cat likes milk dog likes bones cat likes fish dog likes fish cat eats fish".split()

def make_pairs(tokens, window=2):
    words = sorted(set(tokens))
    stoi = {w: i for i, w in enumerate(words)}
    ids = [stoi[w] for w in tokens]
    pairs = [(ids[i], ids[j]) for i in range(len(ids))
             for j in range(max(0, i-window), min(len(ids), i+window+1)) if i != j]
    return words, torch.tensor(pairs, dtype=torch.long)

class SkipGram(nn.Module):
    def __init__(self, vocab, dim=12):
        super().__init__()
        self.center = nn.Embedding(vocab, dim)
        self.context = nn.Embedding(vocab, dim)

    def forward(self, center, positive, negatives):
        v = self.center(center)
        pos = (v * self.context(positive)).sum(-1)
        neg = (v[:, None, :] * self.context(negatives)).sum(-1)
        # softplus(-x) 等价于 -log sigmoid(x)，数值更稳定。
        return (F.softplus(-pos) + F.softplus(neg).sum(-1)).mean()

def train(steps=30):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    words, pairs = make_pairs(TEXT)
    model = SkipGram(len(words))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    for _ in range(steps):
        negatives = torch.randint(0, len(words), (len(pairs), 3))
        loss = model(pairs[:, 0], pairs[:, 1], negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        embeddings = F.normalize(model.center.weight, dim=-1)
        cat = words.index("cat")
        nearest = torch.argsort(embeddings @ embeddings[cat], descending=True)[:4]
    return float(loss.item()), [words[i] for i in nearest.tolist()]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()
    print("RESULT:", train(args.steps))
