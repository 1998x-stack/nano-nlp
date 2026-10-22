"""最小 Encoder–Decoder + cross-attention：玩具数字翻译。"""
import argparse
import torch
from torch import nn
from torch.nn import functional as F

# 源语言 1,2,3；目标语言 4,5,6；0 为起始符。
SOURCE = torch.tensor([[1, 2, 3], [3, 2, 1], [2, 1, 3]])
TARGET = SOURCE + 3

class TinyTranslator(nn.Module):
    def __init__(self, vocab=7, dim=24):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.encoder = nn.GRU(dim, dim, batch_first=True)
        self.decoder = nn.GRU(dim * 2, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab)

    def forward(self, source, target):
        memory, hidden = self.encoder(self.embed(source))
        current = torch.zeros(source.size(0), 1, dtype=torch.long)
        logits, attention_maps = [], []
        for t in range(target.size(1)):
            query = hidden.transpose(0, 1)  # [batch, 1, dim]
            scores = query @ memory.transpose(1, 2)
            weights = scores.softmax(-1)  # [batch, 1, src_len]
            context = weights @ memory   # 动态关注源端所有位置
            decoder_input = torch.cat([self.embed(current), context], dim=-1)
            output, hidden = self.decoder(decoder_input, hidden)
            logits.append(self.head(output))
            attention_maps.append(weights)
            current = target[:, t:t+1]  # teacher forcing
        return torch.cat(logits, 1), torch.cat(attention_maps, 1)

def train(steps=25):
    torch.manual_seed(13)
    torch.set_num_threads(1)
    model = TinyTranslator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(steps):
        logits, attention = model(SOURCE, TARGET)
        loss = F.cross_entropy(logits.flatten(0, 1), TARGET.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return round(loss.item(), 4), list(attention.shape)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=25)
    print("RESULT:", train(p.parse_args().steps))
