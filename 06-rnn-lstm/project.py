"""LSTM 字符语言模型 MVP：BPTT → held-out NLL → 自回归字符生成。"""
import math
import torch
from torch.nn import functional as F
from loguru import logger
from utils.v2io import save_torch_state, record_step
from utils.lm_eval import encode_train_validation, heldout_token_nll
from data import TEXT, BLOCK
from model import LSTMLM

def run(*,steps=30,seed=42,text_file=None):
    """字符预测使用右移一位的标签，验证数据不得回传梯度。

    核心不变量：Tensor: ids[B,T] → emb[B,T,D] → logits[B,T,V]。
    """
    from utils.data import read_text
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    text=read_text(text_file,min_length=40) if text_file else TEXT
    chars, train, val, val_oov_tokens = encode_train_validation(text, block_size=BLOCK)
    # 统计量基于训练段构建词表：验证未见字符统一映射为 <unk>。
    if len(train)<=BLOCK+1 or len(val)<=BLOCK+1:
        raise ValueError("训练和验证划分后都必须有 > BLOCK+1 个字符")
    model=LSTMLM(len(chars))
    optim=torch.optim.AdamW(model.parameters(),lr=.015)
    for t in range(steps):
        offset=(t*7)%(len(train)-BLOCK-1)
        x=train[offset:offset+BLOCK][None,:]
        y=train[offset+1:offset+BLOCK+1][None,:]
        _,loss=model(x,y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        record_step(loss=float(loss.item()))
        if t in (0,steps-1):
            logger.info("LSTM train step {} | CE {:.5f}",t+1,loss.item())
    model.eval()
    val_loss, val_evaluated_tokens, val_chunks = heldout_token_nll(model, val, block_size=BLOCK)
    with torch.no_grad():
        generated_ids=train[:1].tolist()
        for _ in range(30):
            context=torch.tensor(generated_ids[-BLOCK:])[None,:]
            logits,_=model(context)
            generated_ids.append(int(logits[0,-1].argmax()))
    sample="".join(chars[i] for i in generated_ids)
    save_torch_state(model, {"chars": chars, "dim": 24, "block_size": BLOCK,
                             "unk_id": 0, "vocab_scope": "train_only"})

    return {"task":"character_lm","train_chars":len(train),"valid_chars":len(val),
            "vocab":len(chars),"val_nll":round(val_loss,5),
            "val_oov_tokens": val_oov_tokens, "val_evaluated_tokens": val_evaluated_tokens,
            "vocab_scope": "train_only",
            "val_perplexity":round(math.exp(val_loss),5),
            "generated":sample},[{"prompt":sample[0],"generation":sample[1:],
                                  "validation_nll":round(x,5),"validation_tokens": val_chunks[i]["tokens"]} for i,x in enumerate([row["nll"] for row in val_chunks])]


def predict_saved(directory, text=None):
    """从训练时持久化的状态，在独立进程执行真实模型推理。"""
    from inference import infer
    return infer(directory, text)
