"""BERT-style MLM：只在被选中的位置训练/评价，验证词汇重建准确率。"""
import torch
from torch.nn import functional as F
from loguru import logger
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
    batch=torch.tensor(TRAIN)
    model=TinyMLM()
    optim=torch.optim.AdamW(model.parameters(),lr=.025)
    for t in range(steps):
        inputs,mask=mask_at(batch,t%batch.size(1))
        logits=model(inputs)
        loss=F.cross_entropy(logits[mask],batch[mask])
        optim.zero_grad();loss.backward();optim.step()
        if t in (0,steps-1):logger.info("MLM train step={} masked_CE={:.5f}",t+1,loss.item())
    model.eval();rows=[];correct=total=0
    with torch.no_grad():
        valid=torch.tensor(VALID)
        for position in range(valid.size(1)):
            inputs,mask=mask_at(valid,position)
            pred=model(inputs)[mask].argmax(-1)
            truth=valid[mask]
            correct+=int((pred==truth).sum());total+=len(truth)
            for seq,g,p in zip(inputs.tolist(),truth.tolist(),pred.tolist()):
                rows.append({"masked_input":seq,"masked_position":position,"gold":g,"pred":p})
    return {"task":"masked_language_modeling","train_sequences":len(TRAIN),
            "valid_masked_tokens":total,"val_mask_accuracy":correct/total},rows
