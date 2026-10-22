"""nanoGPT 风格的小型 LSTM 字符语言模型。"""
import argparse
from pathlib import Path
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

def train(steps=25, text=TEXT):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    if len(text) < 18:
        raise ValueError('文本至少需要 18 个字符')
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    model = LSTMLM(len(chars))
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
    p.add_argument("--text-file", type=Path, help="可选本地 UTF-8 文本")
    args = p.parse_args()
    text = args.text_file.read_text(encoding="utf-8") if args.text_file else TEXT
    model, loss = train(args.steps, text)
    print("RESULT:", {"loss": round(loss, 4), "parameters": sum(p.numel() for p in model.parameters())})
