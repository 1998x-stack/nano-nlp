"""Seq2Seq 小翻译器：训练时 teacher forcing，推理时自回归预测。"""
import torch
from torch.nn import functional as F
from loguru import logger
from utils.v2io import save_torch_state, record_step
from data import TRAIN, VALID
from model import TinyTranslator

def greedy_decode(model,src,length=3):
    """不读真实 target；每一步只使用前一步自己生成的 token。"""
    memory,hidden=model.encoder(model.embed(src))
    current=torch.zeros(src.size(0),1,dtype=torch.long)
    predicted=[]
    attentions=[]
    for _ in range(length):
        query=hidden.transpose(0,1)
        weights=(query @ memory.transpose(1,2)).softmax(-1)
        context=weights @ memory
        x=torch.cat([model.embed(current),context],dim=-1)
        out,hidden=model.decoder(x,hidden)
        current=model.head(out).argmax(-1)
        predicted.append(current)
        attentions.append(weights)
    return torch.cat(predicted,dim=1),torch.cat(attentions,dim=1)

def run(*,steps=30,seed=42,text_file=None):
    torch.manual_seed(seed);torch.set_num_threads(1)
    train_pairs=[(x,[t+3 for t in x]) for x in TRAIN]
    valid_pairs=[(x,[t+3 for t in x]) for x in VALID]
    if text_file:
        from utils.data import read_jsonl,nonoverlap_split
        rows=read_jsonl(text_file)
        if len(rows)<4 or not all(isinstance(r.get("source"),list) and
                isinstance(r.get("target"),list) and len(r["source"])==len(r["target"])==3
                and all(type(t) is int and 1<=t<=3 for t in r["source"])
                and all(type(t) is int and 4<=t<=6 for t in r["target"]) for r in rows):
            raise ValueError("Seq2Seq JSONL: source长度3 (1..3), target长度3 (4..6)")
        a,b=nonoverlap_split(rows)
        train_pairs=[(r["source"],r["target"]) for r in a]
        valid_pairs=[(r["source"],r["target"]) for r in b]
    src=torch.tensor([a for a,b in train_pairs]);tgt=torch.tensor([b for a,b in train_pairs])
    model=TinyTranslator()
    optim=torch.optim.AdamW(model.parameters(),lr=.025)
    for t in range(steps):
        logits,_=model(src,tgt)
        loss=F.cross_entropy(logits.reshape(-1,7),tgt.reshape(-1))
        optim.zero_grad();loss.backward();optim.step(); record_step(loss=float(loss.item()))
        if t in (0,steps-1):logger.info("translation train step {} loss {:.4f}",t+1,loss.item())
    model.eval()
    with torch.no_grad():
        valid=torch.tensor([a for a,b in valid_pairs])
        pred,att=greedy_decode(model,valid)
        truth=torch.tensor([b for a,b in valid_pairs])
        token_acc=(pred==truth).float().mean().item()
        seq_acc=(pred==truth).all(-1).float().mean().item()
    rows=[{"source":s,"target":g,"pred":p,
           "attention": [[round(float(v),4) for v in row] for row in weights]}
          for (s,_),g,p,weights in zip(valid_pairs,truth.tolist(),pred.tolist(),att.tolist())]
    save_torch_state(model, {"vocab": 7, "dim": 24, "task": "toy_digit_translation"})

    return {"task":"sequence_translation","train_pairs":len(train_pairs),"valid_pairs":len(valid_pairs),
            "val_token_accuracy":round(token_acc,5),"val_exact_match":round(seq_acc,5)},rows


def predict_saved(directory, text=None):
    """从训练时持久化的状态，在独立进程执行真实模型推理。"""
    from inference import infer
    return infer(directory, text)
