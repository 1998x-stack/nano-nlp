"""T5 核心教学：span corruption + encoder-decoder teacher forcing。"""
import argparse
import torch
from torch import nn
from torch.nn import functional as F

# token: 0=<bos>,1=the,2=cat,3=sat,4=on,5=mat,6=and,7=<extra_id_0>,
#        8=<extra_id_1>,9=<eos>
SOURCE = torch.tensor([[1, 7, 3, 4, 8]], dtype=torch.long)
TARGET = torch.tensor([[7, 2, 8, 5, 9]], dtype=torch.long)
DECODER_INPUT = torch.tensor([[0, 7, 2, 8, 5]], dtype=torch.long)

class TinyT5(nn.Module):
    def __init__(self, vocab=10, dim=32):
        super().__init__()
        self.token = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(16, dim)
        # PyTorch Transformer 同时具有 encoder self-attn 和 decoder cross-attn。
        self.transformer = nn.Transformer(d_model=dim, nhead=4,
            num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=64, dropout=0.0, batch_first=True)
        self.head = nn.Linear(dim, vocab)

    def forward(self, src, dec):
        ps = torch.arange(src.size(1))
        pt = torch.arange(dec.size(1))
        src_embed = self.token(src) + self.pos(ps)
        tgt_embed = self.token(dec) + self.pos(pt)
        # 自回归 decoder 不允许看到右边将要输出的目标词。
        causal = nn.Transformer.generate_square_subsequent_mask(dec.size(1))
        hidden = self.transformer(src_embed, tgt_embed, tgt_mask=causal)
        return self.head(hidden)

def train(steps=20):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = TinyT5()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(steps):
        logits = model(SOURCE, DECODER_INPUT)
        loss = F.cross_entropy(logits.reshape(-1, 10), TARGET.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.item())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    print("RESULT:", round(train(p.parse_args().steps), 4))
