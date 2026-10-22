"""Transformer Attention 在真实分类任务上完整训练、推断和验证。"""
import torch
from torch import nn
from torch.nn import functional as F
from loguru import logger
from data import TRAIN, VALID
from model import CausalSelfAttention

class AttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.token=nn.Embedding(4,16)
        self.pos=nn.Embedding(4,16)
        self.attn=CausalSelfAttention(16,2)
        self.norm=nn.LayerNorm(16)
        self.head=nn.Linear(16,2)

    def forward(self,x):
        positions=torch.arange(x.size(1),device=x.device)
        h=self.token(x)+self.pos(positions)
        attended,weights=self.attn(h)
        # 最后一个位置可以在 causal mask 下聚合所有前序 token。
        logits=self.head(self.norm(h+attended)[:,-1])
        return logits,weights

def run(*,steps=30,seed=42,text_file=None):
    torch.manual_seed(seed);torch.set_num_threads(1)
    x=torch.tensor([a for a,b in TRAIN]);y=torch.tensor([b for a,b in TRAIN])
    model=AttentionClassifier();optim=torch.optim.AdamW(model.parameters(),lr=.035)
    for t in range(steps):
        logits,_=model(x)
        loss=F.cross_entropy(logits,y)
        optim.zero_grad();loss.backward();optim.step()
        if t in (0,steps-1):logger.info("attention class step={} loss={:.4f}",t+1,loss.item())
    valid=torch.tensor([a for a,b in VALID]);truth=torch.tensor([b for a,b in VALID])
    model.eval()
    with torch.no_grad():
        logits,weights=model(valid)
        preds=logits.argmax(-1)
        acc=(preds==truth).float().mean().item()
        # 看末尾分类 token 对前面词的可见注意力。
        attention_last=weights.mean(1)[:,-1].tolist()
    rows=[{"tokens":x,"gold":y,"pred":int(p),"attention_last":list(map(lambda v:round(v,4),a))}
          for (x,y),p,a in zip(VALID,preds.tolist(),attention_last)]
    return {"task":"attention_classification","train_samples":len(TRAIN),
            "valid_samples":len(VALID),"val_accuracy":round(acc,5)},rows
