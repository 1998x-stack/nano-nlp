"""T5 span-to-span 任务：Teacher forcing 训练与无标签自回归解码。"""
import torch
from torch.nn import functional as F
from loguru import logger
from data import TRAIN,VALID
from model import TinyT5

def decoder_input(target):
    bos=torch.zeros(target.size(0),1,dtype=target.dtype)
    return torch.cat([bos,target[:,:-1]],dim=1)

def generate(model,src,length=5):
    current=torch.zeros(src.size(0),1,dtype=torch.long)
    outputs=[]
    for _ in range(length):
        logits=model(src,current)
        predicted=logits[:,-1].argmax(-1,keepdim=True)
        outputs.append(predicted)
        current=torch.cat([current,predicted],dim=1)
    return torch.cat(outputs,dim=1)

def run(*,steps=30,seed=42,text_file=None):
    torch.manual_seed(seed);torch.set_num_threads(1)
    src=torch.tensor([a for a,b in TRAIN]);target=torch.tensor([b for a,b in TRAIN])
    model=TinyT5();optim=torch.optim.AdamW(model.parameters(),lr=.02)
    for t in range(steps):
        logits=model(src,decoder_input(target))
        loss=F.cross_entropy(logits.reshape(-1,10),target.reshape(-1))
        optim.zero_grad();loss.backward();optim.step()
        if t in (0,steps-1):logger.info("T5 train step={} loss={:.4f}",t+1,loss.item())
    valid_src=torch.tensor([a for a,b in VALID])
    valid_target=torch.tensor([b for a,b in VALID])
    model.eval()
    with torch.no_grad():
        pred=generate(model,valid_src,valid_target.size(1))
        acc=(pred==valid_target).float().mean().item()
        seq_acc=(pred==valid_target).all(-1).float().mean().item()
    rows=[{"corrupted":s,"target":target,"generated":actual}
          for (s,target),actual in zip(VALID,pred.tolist())]
    return {"task":"span_corruption","train_pairs":len(TRAIN),
            "val_token_accuracy":round(acc,5),"val_exact_match":round(seq_acc,5)},rows
