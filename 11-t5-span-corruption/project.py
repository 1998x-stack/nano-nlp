"""T5 span-to-span 任务：Teacher forcing 训练与无标签自回归解码。"""
import torch
from torch.nn import functional as F
from loguru import logger
from utils.v2io import save_torch_state, record_step
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
    """Encoder 编码损坏文本，Decoder 自回归生成 sentinel 边界及被删片段。

    核心不变量：训练输入必须按位置右移；推理只看前面已输出的目标 token。
    """
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    train,valid=TRAIN,VALID
    if text_file:
        from utils.data import read_jsonl,nonoverlap_split
        rows=read_jsonl(text_file)
        if len(rows)<4 or not all(isinstance(r.get("source"),list) and
            isinstance(r.get("target"),list) and len(r["source"])==5 and len(r["target"])==5
            and all(type(v) is int and 0<=v<10 for v in r["source"]+r["target"])
            and 7 in r["source"] and 8 in r["source"]
            and r["target"][0]==7 and r["target"][-1]==9 for r in rows):
            raise ValueError("T5 JSONL: source/target 各5个token 0..9, 含sentinel7/8, target EOS9")
        a,b=nonoverlap_split(rows)
        train=[(r["source"],r["target"]) for r in a]
        valid=[(r["source"],r["target"]) for r in b]
    src=torch.tensor([a for a,b in train])
    target=torch.tensor([b for a,b in train])
    model=TinyT5()
    optim=torch.optim.AdamW(model.parameters(),lr=.02)
    for t in range(steps):
        logits=model(src,decoder_input(target))
        loss=F.cross_entropy(logits.reshape(-1,10),target.reshape(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        record_step(loss=float(loss.item()))
        if t in (0,steps-1):logger.info("T5 train step={} loss={:.4f}",t+1,loss.item())
    valid_src=torch.tensor([a for a,b in valid])
    valid_target=torch.tensor([b for a,b in valid])
    model.eval()
    with torch.no_grad():
        pred=generate(model,valid_src,valid_target.size(1))
        acc=(pred==valid_target).float().mean().item()
        seq_acc=(pred==valid_target).all(-1).float().mean().item()
    rows=[{"corrupted":s,"target":target,"generated":actual}
          for (s,target),actual in zip(valid,pred.tolist())]
    save_torch_state(model, {"vocab": 10, "dim": 32, "task": "span_corruption"})

    return {"task":"span_corruption","train_pairs":len(train),
            "val_token_accuracy":round(acc,5),"val_exact_match":round(seq_acc,5)},rows


def predict_saved(directory, text=None):
    """从训练时持久化的状态，在独立进程执行真实模型推理。"""
    from inference import infer
    return infer(directory, text)
