"""Seq2Seq 小翻译器：训练时 teacher forcing，推理时自回归预测。"""
import torch
from torch.nn import functional as F
from loguru import logger
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
    src=torch.tensor(TRAIN);tgt=src+3
    model=TinyTranslator()
    optim=torch.optim.AdamW(model.parameters(),lr=.025)
    for t in range(steps):
        logits,_=model(src,tgt)
        loss=F.cross_entropy(logits.reshape(-1,7),tgt.reshape(-1))
        optim.zero_grad();loss.backward();optim.step()
        if t in (0,steps-1):logger.info("translation train step {} loss {:.4f}",t+1,loss.item())
    model.eval()
    with torch.no_grad():
        valid=torch.tensor(VALID)
        pred,att=greedy_decode(model,valid)
        truth=valid+3
        token_acc=(pred==truth).float().mean().item()
        seq_acc=(pred==truth).all(-1).float().mean().item()
    rows=[{"source":s,"target":g,"pred":p,
           "attention": [[round(float(v),4) for v in row] for row in weights]}
          for s,g,p,weights in zip(VALID,truth.tolist(),pred.tolist(),att.tolist())]
    return {"task":"sequence_translation","train_pairs":len(TRAIN),"valid_pairs":len(VALID),
            "val_token_accuracy":round(token_acc,5),"val_exact_match":round(seq_acc,5)},rows
