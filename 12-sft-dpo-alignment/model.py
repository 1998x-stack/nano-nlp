"""SFT + DPO 核心公式：单 token 回答任务，非完整聊天训练。"""
import argparse
import copy
import torch
from torch import nn
from torch.nn import functional as F

# 问题 token 0、1；答案候选 token 0/1：每道题均有 chosen/rejected。
PROMPTS = torch.tensor([0, 1])
CHOSEN = torch.tensor([1, 0])
REJECTED = 1 - CHOSEN

class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(2, 8)
        self.head = nn.Linear(8, 2)

    def forward(self, prompt):
        return self.head(self.embed(prompt)).log_softmax(-1)

def train(steps=20, beta=0.5):
    torch.manual_seed(42)
    policy = TinyPolicy()
    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.08)
    for step in range(steps):
        logp = policy(PROMPTS)
        if step < steps // 2:
            # SFT：只使用示范答案。
            loss = F.nll_loss(logp, CHOSEN)
        else:
            with torch.no_grad():
                ref = reference(PROMPTS)
            p_good = logp.gather(1, CHOSEN[:, None]).squeeze(1)
            p_bad = logp.gather(1, REJECTED[:, None]).squeeze(1)
            r_good = ref.gather(1, CHOSEN[:, None]).squeeze(1)
            r_bad = ref.gather(1, REJECTED[:, None]).squeeze(1)
            gap = (p_good - p_bad) - (r_good - r_bad)
            loss = -F.logsigmoid(beta * gap).mean()  # DPO 目标
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return policy(PROMPTS).argmax(-1).tolist(), round(loss.item(), 4)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    print("RESULT:", train(p.parse_args().steps))
