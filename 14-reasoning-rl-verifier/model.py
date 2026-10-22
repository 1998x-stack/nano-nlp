"""可验证任务的组内优势策略梯度；教学化简，不是完整 GRPO。"""
import argparse
import torch
from torch import nn
from torch.distributions import Categorical

def verify(op, a, b, answer):
    """答案是否满足外部确定性规则；reward 不由同一个模型打分。"""
    gold = a + b if op == 0 else a * b
    return float(answer == gold)

class OpPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2, 2))  # 两种题型，各选 + / ×

    def forward(self, task):
        return Categorical(logits=self.logits[task])

def train(steps=40, group=8):
    torch.manual_seed(13)
    policy = OpPolicy()
    optim = torch.optim.AdamW(policy.parameters(), lr=0.12)
    for _ in range(steps):
        losses = []
        for task in (0, 1):
            dist = policy(task)
            actions = dist.sample((group,))
            candidate_answers = [3 + 4 if x.item() == 0 else 3 * 4 for x in actions]
            rewards = torch.tensor([verify(task, 3, 4, x) for x in candidate_answers])
            # 组内零方差时 A=0；因此极端全对/全错组无梯度。
            advantage = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
            losses.append(-(advantage.detach() * dist.log_prob(actions)).mean())
        loss = torch.stack(losses).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    return policy.logits.argmax(-1).tolist(), round(loss.item(), 4)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=40)
    print("RESULT:", train(p.parse_args().steps))
