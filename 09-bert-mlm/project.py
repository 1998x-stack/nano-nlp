"""BERT-style MLM：只在被选中的位置训练/评价，验证词汇重建准确率。"""
import torch
from torch.nn import functional as F
from loguru import logger
from utils.v2io import save_torch_state, record_step
from data import TRAIN, VALID
from model import TinyMLM

def mask_at(batch,position):
    masked=batch.clone()
    mask=torch.zeros_like(masked,dtype=torch.bool)
    mask[:,position]=True
    masked[mask]=0
    return masked,mask

def run(*,steps=30,seed=42,text_file=None):
    torch.manual_seed(seed);torch.set_num_threads(1)
    train,valid=TRAIN,VALID
    if text_file:
        from utils.data import read_jsonl,nonoverlap_split
        rows=read_jsonl(text_file)
        lengths={len(r.get("tokens",[])) for r in rows}
        if len(rows)<4 or len(lengths)!=1 or not 1<=next(iter(lengths))<=16 or not all(
                isinstance(r.get("tokens"),list) and
                all(type(x) is int and 1<=x<=5 for x in r["tokens"]) for r in rows):
            raise ValueError("MLM JSONL: 固定长度1..16的tokens, token范围1..5")
        a,b=nonoverlap_split(rows)
        train=[r["tokens"] for r in a]
        valid=[r["tokens"] for r in b]
    batch=torch.tensor(train)
    model=TinyMLM()
    optim=torch.optim.AdamW(model.parameters(),lr=.025)
    for t in range(steps):
        inputs,mask=mask_at(batch,t%batch.size(1))
        logits=model(inputs)
        loss=F.cross_entropy(logits[mask],batch[mask])
        optim.zero_grad();loss.backward();optim.step(); record_step(loss=float(loss.item()))
        if t in (0,steps-1):logger.info("MLM train step={} masked_CE={:.5f}",t+1,loss.item())
    model.eval();rows=[];correct=total=0
    with torch.no_grad():
        valid_batch=torch.tensor(valid)
        for position in range(valid_batch.size(1)):
            inputs,mask=mask_at(valid_batch,position)
            pred=model(inputs)[mask].argmax(-1)
            truth=valid_batch[mask]
            correct+=int((pred==truth).sum());total+=len(truth)
            for seq,g,p in zip(inputs.tolist(),truth.tolist(),pred.tolist()):
                rows.append({"masked_input":seq,"masked_position":position,"gold":g,"pred":p})
    save_torch_state(model, {"vocab": 6, "max_length": 16, "mask_id": 0})

    return {"task":"masked_language_modeling","train_sequences":len(train),
            "valid_masked_tokens":total,"val_mask_accuracy":correct/total},rows


def predict_saved(directory, text=None):
    """从训练时持久化的状态，在独立进程执行真实模型推理。"""
    from inference import infer
    return infer(directory, text)
