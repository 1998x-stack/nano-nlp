"""迷你 BERT-style masked LM：区分双向训练和因果生成。"""
import argparse
import torch
from torch import nn
from torch.nn import functional as F

# 0=[MASK], 1=the, 2=cat, 3=eats, 4=fish, 5=dog
DATA = torch.tensor([[1, 2, 3, 4], [1, 5, 3, 4]], dtype=torch.long)

class TinyMLM(nn.Module):
    def __init__(self, vocab=6, dim=24):
        super().__init__()
        self.token = nn.Embedding(vocab, dim)
        self.position = nn.Embedding(16, dim)
        block = nn.TransformerEncoderLayer(dim, 4, 48, batch_first=True, dropout=0)
        self.encoder = nn.TransformerEncoder(block, 1)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x):
        position = torch.arange(x.size(1), device=x.device)
        return self.head(self.encoder(self.token(x) + self.position(position)))

def train(steps=25):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = TinyMLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for i in range(steps):
        input_ids = DATA.clone()
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[:, i % DATA.size(1)] = True
        input_ids[mask] = 0
        logits = model(input_ids)
        # MLM 只计算被遮挡位置；不能把未遮挡 token 纳入 loss。
        loss = F.cross_entropy(logits[mask], DATA[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.item())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=25)
    print("RESULT:", round(train(p.parse_args().steps), 4))
