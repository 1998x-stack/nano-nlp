"""LSTM 字符语言模型 MVP：BPTT → held-out NLL → 自回归字符生成。"""
import math
import torch
from torch.nn import functional as F
from loguru import logger
from utils.v2io import save_torch_state, record_step
from data import TEXT, BLOCK
from model import LSTMLM

def run(*,steps=30,seed=42,text_file=None):
    from utils.data import read_text
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    text=read_text(text_file,min_length=40) if text_file else TEXT
    chars=sorted(set(text))
    stoi={c:i for i,c in enumerate(chars)}
    ids=torch.tensor([stoi[ch] for ch in text],dtype=torch.long)
    cut=int(.8*len(ids))
    train,val=ids[:cut],ids[cut:]
    if len(train)<=BLOCK+1 or len(val)<=BLOCK+1:
        raise ValueError("训练和验证划分后都必须有 > BLOCK+1 个字符")
    model=LSTMLM(len(chars))
    optim=torch.optim.AdamW(model.parameters(),lr=.015)
    for t in range(steps):
        offset=(t*7)%(len(train)-BLOCK-1)
        x=train[offset:offset+BLOCK][None,:]
        y=train[offset+1:offset+BLOCK+1][None,:]
        _,loss=model(x,y)
        optim.zero_grad();loss.backward();optim.step(); record_step(loss=float(loss.item()))
        if t in (0,steps-1):
            logger.info("LSTM train step {} | CE {:.5f}",t+1,loss.item())
    model.eval()
    with torch.no_grad():
        # 仅在完整保留的验证分片上计算 NLL（不反传）。
        nll=[]
        for j in range(0,len(val)-BLOCK-1,BLOCK):
            _,v_loss=model(val[j:j+BLOCK][None,:],val[j+1:j+BLOCK+1][None,:])
            nll.append(float(v_loss.item()))
        generated_ids=ids[:1].tolist()
        for _ in range(30):
            context=torch.tensor(generated_ids[-BLOCK:])[None,:]
            logits,_=model(context)
            generated_ids.append(int(logits[0,-1].argmax()))
    sample="".join(chars[i] for i in generated_ids)
    val_loss=sum(nll)/len(nll)
    save_torch_state(model, {"chars": chars, "dim": 24, "block_size": BLOCK})

    return {"task":"character_lm","train_chars":len(train),"valid_chars":len(val),
            "vocab":len(chars),"val_nll":round(val_loss,5),
            "val_perplexity":round(math.exp(val_loss),5),
            "generated":sample},[{"prompt":sample[0],"generation":sample[1:],
                                  "validation_nll":round(x,5)} for x in nll]


def predict_saved(directory, text=None):
    """从训练时持久化的状态，在独立进程执行真实模型推理。"""
    from inference import infer
    return infer(directory, text)
