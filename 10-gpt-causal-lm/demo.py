"""nanoGPT-inspired，单文件透明实现 Decoder-only Transformer。"""
from dataclasses import dataclass
import argparse
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F

TEXT = ("to be or not to be\n" * 25)
CHARS = sorted(set(TEXT))
STOI = {c: i for i, c in enumerate(CHARS)}

@dataclass
class Config:
    block_size: int = 16
    n_embd: int = 32
    n_head: int = 4

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.ff = nn.Sequential(nn.Linear(c.n_embd, c.n_embd * 4),
                                nn.GELU(), nn.Linear(c.n_embd * 4, c.n_embd))
        self.heads = c.n_head

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        def split(z):
            return z.reshape(B, T, self.heads, C // self.heads).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), -float("inf"))
        y = att.softmax(-1) @ v
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        x = x + self.proj(y)    # pre-norm residual attention
        return x + self.ff(self.ln2(x))  # pre-norm residual MLP

class TinyGPT(nn.Module):
    def __init__(self, vocab, config=Config()):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(vocab, config.n_embd)
        self.position = nn.Embedding(config.block_size, config.n_embd)
        self.block = Block(config)
        self.norm = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, vocab, bias=False)

    def forward(self, tokens, targets=None):
        T = tokens.size(1)
        assert T <= self.config.block_size
        x = self.token(tokens) + self.position(torch.arange(T, device=tokens.device))
        logits = self.head(self.norm(self.block(x)))
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten()) if targets is not None else None
        return logits, loss

def train(steps=25, text=TEXT):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    if len(text) < 18:
        raise ValueError('文本至少需要 18 个字符')
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[ch] for ch in text])
    model = TinyGPT(len(chars))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008)
    for i in range(steps):
        j = (i * 9) % (len(ids) - 17)
        x, y = ids[j:j+16][None, :], ids[j+1:j+17][None, :]
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return round(loss.item(), 4)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--text-file", type=Path, help="可选本地 UTF-8 文本")
    args = p.parse_args()
    text = args.text_file.read_text(encoding="utf-8") if args.text_file else TEXT
    print("RESULT:", train(args.steps, text))
