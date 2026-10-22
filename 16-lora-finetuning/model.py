"""LoRA 的四则运算：冻结 W，训练增量 B@A。"""
import argparse
import torch
from torch import nn
from torch.nn import functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_dim=8, out_dim=4, rank=2, alpha=4):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim, bias=False)
        for p in self.base.parameters():
            p.requires_grad_(False)  # 冻结预训练权重
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)  # 初始化时恰好等于基座模型
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scale

def train(steps=25):
    torch.manual_seed(42)
    model = LoRALinear()
    x = torch.randn(32, 8)
    target = torch.stack([x[:, 0] * 2, x[:, 1] * (-3),
                          x[:, 2] + x[:, 3], x[:, 4]], dim=1)
    optimizer = torch.optim.AdamW(
        list(model.A.parameters()) + list(model.B.parameters()), lr=0.08
    )
    for _ in range(steps):
        loss = F.mse_loss(model(x), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "loss": round(loss.item(), 4)}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=25)
    print("RESULT:", train(p.parse_args().steps))
