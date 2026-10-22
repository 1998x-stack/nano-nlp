"""透明的 Multi-Head Causal Self-Attention，适合逐行读。"""
import math
import torch
from torch import nn

class CausalSelfAttention(nn.Module):
    def __init__(self, dim=16, heads=2):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # [B, T, C] → [B, H, T, D]
        def split(t):
            return t.reshape(B, T, self.heads, C // self.heads).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        # True 表示未来位置，必须不能参加 softmax。
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = scores.softmax(dim=-1)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(y), weights

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 5, 16)
    y, attention = CausalSelfAttention()(x)
    assert torch.all(attention[..., torch.triu_indices(5, 5, 1)[0],
                               torch.triu_indices(5, 5, 1)[1]] == 0)
    print("RESULT:", {"output": list(y.shape), "weights": list(attention.shape),
                      "first_row": attention[0, 0, 0].tolist()})
