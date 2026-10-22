"""nanoGPT 风格的小型 LSTM 字符语言模型。"""
import argparse
import torch
from torch import nn
from torch.nn import functional as F

TEXT = ("to be or not to be that is the question\n" * 12)
CHARS = sorted(set(TEXT))
STOI = {c: i for i, c in enumerate(CHARS)}

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, dim=24):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.rnn = nn.LSTM(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x, targets=None):
        h, _ = self.rnn(self.embed(x))  # 保留每个时间步的隐状态
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

def train(steps=25):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    ids = torch.tensor([STOI[c] for c in TEXT], dtype=torch.long)
    model = LSTMLM(len(CHARS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015)
    for i in range(steps):
        offset = (i * 7) % (len(ids) - 17)
        x = ids[offset:offset+16][None, :]
        y = ids[offset+1:offset+17][None, :]  # next-token shift
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, float(loss.item())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=25)
    model, loss = train(p.parse_args().steps)
    print("RESULT:", {"loss": round(loss, 4), "parameters": sum(p.numel() for p in model.parameters())})
