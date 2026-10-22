"""Transformer Attention 在真实分类任务上完整训练、推断和验证。"""
import torch
from torch import nn
from torch.nn import functional as F
from loguru import logger
from utils.v2io import save_torch_state, record_step
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
    train,valid=TRAIN,VALID
    if text_file:
        from utils.data import read_jsonl,nonoverlap_split
        rows=read_jsonl(text_file)
        if len(rows)<4 or not all(isinstance(r.get("tokens"),list) and len(r["tokens"])==4
                   and all(type(t) is int and 0<=t<4 for t in r["tokens"])
                   and r.get("label") in (0,1) for r in rows):
            raise ValueError("Attention JSONL: tokens长度4范围0..3, label 0/1")
        a,b=nonoverlap_split(rows)
        train=[(r["tokens"],r["label"]) for r in a]
        valid=[(r["tokens"],r["label"]) for r in b]
    x=torch.tensor([a for a,b in train]);y=torch.tensor([b for a,b in train])
    model=AttentionClassifier();optim=torch.optim.AdamW(model.parameters(),lr=.035)
    for t in range(steps):
        logits,_=model(x)
        loss=F.cross_entropy(logits,y)
        optim.zero_grad();loss.backward();optim.step(); record_step(loss=float(loss.item()))
        if t in (0,steps-1):logger.info("attention class step={} loss={:.4f}",t+1,loss.item())
    valid_x=torch.tensor([a for a,b in valid]);truth=torch.tensor([b for a,b in valid])
    model.eval()
    with torch.no_grad():
        logits,weights=model(valid_x)
        preds=logits.argmax(-1)
        acc=(preds==truth).float().mean().item()
        # 看末尾分类 token 对前面词的可见注意力。
        attention_last=weights.mean(1)[:,-1].tolist()
    rows=[{"tokens":x,"gold":y,"pred":int(p),"attention_last":list(map(lambda v:round(v,4),a))}
          for (x,y),p,a in zip(valid,preds.tolist(),attention_last)]
    save_torch_state(model, {"vocab": 4, "max_length": 4, "task": "attention_classification"})

    return {"task":"attention_classification","train_samples":len(train),
            "valid_samples":len(valid),"val_accuracy":round(acc,5)},rows


def predict_saved(directory, text=None):
    """从训练时持久化的状态，在独立进程执行真实模型推理。"""
    from inference import infer
    return infer(directory, text)
